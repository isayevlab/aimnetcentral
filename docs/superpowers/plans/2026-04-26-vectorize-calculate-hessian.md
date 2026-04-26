# Vectorize AIMNet2Calculator.calculate_hessian

> **Status:** TODO. Blocked on a missing `vmap` batching rule for the custom CUDA kernel `aimnet::conv_sv_2d_sp_bwd_bwd`. Two-PR plan below.

**Goal:** Replace the row-wise Python loop in `AIMNet2Calculator.calculate_hessian` (`aimnet/calculators/calculator.py:1135-1142`) with a vectorized autograd path. The current implementation calls `torch.autograd.grad(...)` `3N` times, dominating Hessian wall-clock for any caller — `AIMNet2ASE.get_hessian`, `AIMNet2Pysis.get_hessian`, and the new Sella analytic-Hessian callback all share this bottleneck.

**Expected speedup:** ~20–25× on caffeine (N=24, GPU). Forces-only call is ~29 ms; current Hessian is ~1193 ms; vectorized would be in the ~50 ms regime.

**Correctness anchor:** `tests/test_calculator.py::test_external_hessian_matches_internal` already compares the internal Hessian against `torch.autograd.functional.hessian` and asserts agreement within `5e-3`. Whatever vectorized rewrite lands must pass this test.

---

## Why it's blocked today

All three vectorization strategies route the second backward through `vmap`:

- `torch.autograd.functional.hessian(energy_fn, coord, vectorize=True)`
- `torch.autograd.grad(forces, coord, grad_outputs=eye, is_grads_batched=True)`
- `torch.func.hessian(energy_fn)(coord)` (with `functional_call`)

The AIMNet2 model's AEV layer (`aimnet/modules/aev.py:156`) routes through one of two paths depending on `nbops.get_nb_mode(data)`:

- `mode == 0` (3-D batched, no flattening) — pure `torch.einsum`, supports `vmap`.
- `mode > 0` (flattened with `nbmat`) — custom CUDA kernel `aimnet::conv_sv_2d_sp_*` from `aimnet/kernels/conv_sv_2d_sp_wp.py`.

The kernel registers `register_autograd` (forward + first/second backward) and `register_fake`, but **does not register a `vmap` batching rule**. Every vectorization path raises:

```
RuntimeError: Batching rule not implemented for aimnet::conv_sv_2d_sp_bwd_bwd.
We could not generate a fallback.
```

The "real" path — the one all `AIMNet2ASE.get_hessian` and `AIMNet2Pysis.get_hessian` callers exercise — is the flattened `nbmat` path. The 3-D einsum path was confirmed to also produce a malformed `(N, 3, 0, 3)` Hessian under `forces=True, hessian=True` due to a separate `mol_flatten` interaction (see the inline comment in `aimnet/calculators/aimnet2ase.py:160-164`), so it is not a viable fallback.

---

## Two-PR plan

### PR 1 — Register `vmap` rule on the custom kernel

**File:** `aimnet/kernels/conv_sv_2d_sp_wp.py`

Use `torch.library.register_vmap` to register a batching rule for `aimnet::conv_sv_2d_sp_bwd_bwd`. The kernel is over a batch of atoms; `vmap` adds an outer dim that maps to the existing leading dim. The rule should:

1. Detect which input has the `vmap` batch dim (typically the gradient probe).
2. Reshape to fold the vmap dim into the existing batch dim.
3. Call the underlying kernel.
4. Reshape the output to restore the vmap dim.

Reference: [PyTorch Custom Operators tutorial — vmap support](https://pytorch.org/tutorials/advanced/python_custom_ops.html#adding-vmap-support-to-an-operator).

**Test gate:** add a unit test in `tests/test_calculator.py` (or a new `tests/test_kernels.py`) that exercises `torch.func.hessian(energy_fn, coord)` on a small molecule (water, 3 atoms — small enough for fast CI on CPU) and asserts the result is finite, shape `(3, 3, 3, 3)`, and within `5e-3` of the loop-based Hessian.

The existing regression at `tests/test_calculator.py:300-318` becomes the integration test for this kernel change — it should keep passing.

### PR 2 — Swap `calculate_hessian` to use vectorized autograd

Once PR 1 lands, replace `aimnet/calculators/calculator.py:1135-1142`:

```python
@staticmethod
def calculate_hessian(forces: Tensor, coord: Tensor) -> Tensor:
    hessian = -torch.stack([
        torch.autograd.grad(_f, coord, retain_graph=True)[0] for _f in forces.flatten().unbind()
    ]).view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]
    return hessian
```

with the `torch.func.vmap`-over-vjp-closure form:

```python
@staticmethod
def calculate_hessian(forces: Tensor, coord: Tensor) -> Tensor:
    n = forces.numel()
    eye = torch.eye(n, device=forces.device, dtype=forces.dtype)

    def vjp(go):
        return torch.autograd.grad(
            forces.flatten(),
            coord,
            grad_outputs=go,
            retain_graph=True,
            allow_unused=True,
        )[0]

    hessian = -torch.func.vmap(vjp, 0)(eye)
    # hessian shape: (n, N+1, 3) → reshape, slice off padding rows/cols
    return hessian.view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]
```

**Why not `is_grads_batched=True` or `autograd.functional.hessian(vectorize=True)`:** Both go through the legacy C++ batching dispatch, which does NOT consult `torch.library.register_vmap` rules. They will continue to raise `RuntimeError: Batching rule not implemented` even after PR 1 registers rules on the kernel. Only `torch.func.vmap` (and `torch.vmap`, its alias) dispatch through the functorch path that consults `register_vmap`. Verified empirically during PR 1 development. See `docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian-pr-a-vmap-rule.md` for details.

**Acceptance**:

- `pytest tests/test_calculator.py::test_external_hessian_matches_internal` — passes (proves correctness vs. analytical reference).
- `pytest tests/test_calculator.py -k hessian` — passes (broader Hessian suite).
- `pytest tests/test_ase.py::TestHessian` — passes (ASE wrapper tests).
- Benchmark on caffeine N=24 GPU: new path measurably faster than old (target ~10× minimum to justify the change).

---

## What this unlocks

- `AIMNet2ASE.get_hessian` becomes ~20× cheaper → analytic-Hessian Sella TS searches become viable for moderate-sized systems (50–100 atoms) on a single GPU.
- `AIMNet2Pysis.get_hessian` benefits identically → faster IRC, faster pysisyphus TS.
- The OOM-warning threshold in `AIMNet2ASE.get_hessian` (currently `len(atoms) > 100`) can be revisited — vectorized backward holds less peak memory than the row-wise loop.

## Out of scope

- Changing the public Hessian shape contract. The output is still `(N, 3, N, 3)` Cartesian eV/Å². Wrapper code in `aimnet2ase.py` and `aimnet2pysis.py` stays unchanged.
- PBC Hessian. Still rejected at the wrapper. Periodic TS belongs to a separate plan.
- Multi-molecule batched Hessian. Still rejected at `aimnet/calculators/calculator.py:838-839`. Per-system batching is a separate workflow project.

## Related

- Implementation source: `aimnet/calculators/calculator.py:1131-1142`, `aimnet/modules/aev.py:156`, `aimnet/kernels/conv_sv_2d_sp_wp.py`.
- Callers: `aimnet/calculators/aimnet2ase.py:131-` (`get_hessian`), `aimnet/calculators/aimnet2pysis.py:68-74` (`get_hessian`).
- Existing regression test: `tests/test_calculator.py:300-318` (`test_external_hessian_matches_internal`).
- Sibling Sella plan (originating context): `docs/superpowers/plans/2026-04-26-sella-integration.md`.
