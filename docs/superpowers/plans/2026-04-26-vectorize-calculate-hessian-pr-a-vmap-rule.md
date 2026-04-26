# PR A: vmap rule for `aimnet::conv_sv_2d_sp_bwd_bwd`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a `torch.library.register_vmap` rule on the custom CUDA op `aimnet::conv_sv_2d_sp_bwd_bwd` so that any `vmap`-based Hessian path (`torch.func.hessian`, `torch.autograd.grad(..., is_grads_batched=True)`, `torch.autograd.functional.hessian(..., vectorize=True)`) stops raising `RuntimeError: Batching rule not implemented for aimnet::conv_sv_2d_sp_bwd_bwd`. After this PR the kernel can be exercised under `vmap`; the calculator itself is still on the row-wise loop and will be swapped in PR B.

**Architecture:** Add a single `register_vmap` rule that handles the only Hessian-relevant case: a leading `vmap` batch dim on the cotangent inputs (`grad2_a`, `grad2_g`), with the saved tensors (`grad_output`, `a`, `idx`, `g`) un-batched. Implementation strategy is a Python-level K-loop that calls the underlying kernel once per probe — each call queues async GPU launches on the same CUDA stream, so the K invocations overlap on-device. Folding K into the kernel's existing leading dim was rejected because the kernels treat the LAST row as a single padding sentinel (`padding_value = B - 1`), and stacking K copies would scatter padding rows mid-tensor. K-loop is correct, simple, and gets the bulk of the speedup (the win comes from collapsing 3N Python autograd traversals into ONE traversal that issues K kernel launches per backward node — not from batching the kernel itself). A future fully-batched kernel can replace the loop without changing the rule's contract.

**Tech Stack:** PyTorch 2.9.1 (`torch.library.register_vmap`, `torch.func.hessian`, `torch.autograd.grad(..., is_grads_batched=True)`), NVIDIA Warp 1.x kernels in `aimnet/kernels/conv_sv_2d_sp_wp.py`, pytest with `gpu` marker.

---

## Context

**Parent plan:** `docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md` (two-PR roadmap; this is PR 1, renamed PR A here).

**Why this PR is needed:** Every viable vectorized Hessian strategy routes the second backward through `vmap`. The AEV layer (`aimnet/modules/aev.py:151-170`) selects between an einsum path (CPU + `mode == 0`, vmap-compatible) and the custom kernel path (`mode > 0` and CUDA), and the kernel path is the one all real callers exercise (`AIMNet2ASE.get_hessian`, `AIMNet2Pysis.get_hessian`, the new Sella analytic-Hessian callback). The kernel registers `register_autograd` (forward + first/second backward) and `register_fake`, but **no batching rule**, so any `vmap`-based path through it raises immediately.

**What this PR does NOT do:** It does not change `AIMNet2Calculator.calculate_hessian`. The calculator continues to use the row-wise Python loop. That swap is PR B and depends on this PR landing first.

**Correctness anchor:** The existing regression `tests/test_calculator.py::test_external_hessian_matches_internal` (line 300) compares the calculator's internal Hessian against `torch.autograd.functional.hessian` (the non-vectorized form, `vectorize=False`) within `5e-3`. PR A must not regress that test. PR A also adds new tests that exercise the *vectorized* path (which today fails), proving the new rule works end-to-end.

**Op signature being decorated** (from `aimnet/kernels/conv_sv_2d_sp_wp.py:292-304`):

```
aimnet::conv_sv_2d_sp_bwd_bwd(
    grad_output: Tensor,   # (B, A, G, 4) — saved cotangent of forward output
    grad2_a:     Tensor,   # (B, A, G)    — incoming cotangent for grad_a
    grad2_g:     Tensor,   # (B, M, G, 4) — incoming cotangent for grad_g
    a:           Tensor,   # (B, A, G)    — saved
    idx:         Tensor,   # (B, M)       — saved (int64; padding rows == B-1)
    g:           Tensor,   # (B, M, G, 4) — saved
) -> [grad_grad_output, grad_a_double, grad_g_double]
```

Under `is_grads_batched=True`, the realistic `in_dims` from autograd is `(None, 0, 0, None, None, None)` — only the cotangents are batched. The rule still handles arbitrary `in_dims` for robustness (nested vmap, future use).

---

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Modify | `aimnet/kernels/conv_sv_2d_sp_wp.py` | Add `@torch.library.register_vmap("aimnet::conv_sv_2d_sp_bwd_bwd")` rule below the existing `register_autograd` block (after line 453, before the public-API section header at line 456). |
| Modify | `tests/test_conv_sv_2d_sp.py` | Add a kernel-level direct vmap test (failure pin + correctness check) inside `class TestConvSV2dSP`. Existing pattern in this file uses GPU-only fixtures via `pytestmark = pytest.mark.gpu`; reuse them. |
| Modify | `tests/test_calculator_gpu.py` | Add an end-to-end class `TestVectorizedHessian` covering both `torch.func.hessian` and `is_grads_batched=True` over `AIMNet2Calculator` on a small molecule. Module is already `pytestmark = pytest.mark.gpu`. |

No new files. No changes to `aimnet/calculators/calculator.py`, `aimnet/modules/aev.py`, or any wrappers.

---

## Tasks

### Task 1: Pin the failure mode with a kernel-level vmap test (will fail before the rule is registered)

**Files:**
- Modify: `tests/test_conv_sv_2d_sp.py` — append a method to `class TestConvSV2dSP` after `test_padding_behavior` (currently ends around line 316).

- [ ] **Step 1: Add the failing test**

Append this method inside `class TestConvSV2dSP` (the `pytestmark = pytest.mark.gpu` at module top makes it GPU-only; the `WARP_AVAILABLE` skip on the class makes it warp-gated):

```python
    def test_vmap_bwd_bwd_kernel_rule(self, test_data_small_cuda):
        """vmap over the second-backward kernel must dispatch via the registered rule.

        Before PR A this raises RuntimeError("Batching rule not implemented for
        aimnet::conv_sv_2d_sp_bwd_bwd"). After PR A it produces a (K, ...) batched
        result whose k-th slice equals the un-vmapped call with the k-th probe.
        """
        a, idx, g = test_data_small_cuda  # B=2, A=4, G=3, M=4
        B, A, G = a.shape
        _, M = idx.shape
        K = 5
        device = a.device
        dtype = a.dtype

        # Build inputs for conv_sv_2d_sp_bwd_bwd directly. None of these are
        # required to come from a real autograd trace for this unit test —
        # we just need shape-consistent tensors that exercise the kernel.
        grad_output = torch.randn(B, A, G, 4, device=device, dtype=dtype)
        grad2_a = torch.randn(K, B, A, G, device=device, dtype=dtype)
        grad2_g = torch.randn(K, B, M, G, 4, device=device, dtype=dtype)
        a_in = a.detach()
        g_in = g.detach()

        def call_kernel(g2a, g2g):
            return torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd(
                grad_output, g2a, g2g, a_in, idx, g_in
            )

        batched = torch.vmap(
            call_kernel,
            in_dims=(0, 0),
        )(grad2_a, grad2_g)

        # batched is a list of 3 tensors, each with leading K dim
        assert len(batched) == 3
        assert batched[0].shape == (K, B, A, G, 4)
        assert batched[1].shape == (K, B, A, G)
        assert batched[2].shape == (K, B, M, G, 4)

        # Spot-check k=0 against the un-vmapped call
        ref = call_kernel(grad2_a[0], grad2_g[0])
        assert torch.allclose(batched[0][0], ref[0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(batched[1][0], ref[1], atol=1e-5, rtol=1e-4)
        assert torch.allclose(batched[2][0], ref[2], atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: Run the test and confirm it fails for the expected reason**

Run: `pytest tests/test_conv_sv_2d_sp.py::TestConvSV2dSP::test_vmap_bwd_bwd_kernel_rule -v`

Expected: FAIL with a `RuntimeError` whose message contains `Batching rule not implemented for aimnet::conv_sv_2d_sp_bwd_bwd`. If the failure is anything else (shape mismatch, CUDA error, missing op), STOP and investigate before proceeding — the test is wrong, not the kernel.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_conv_sv_2d_sp.py
git commit -m "test(kernels): pin missing vmap rule for conv_sv_2d_sp_bwd_bwd"
```

The failing test is the pin. Do NOT mark it `xfail` — the next task makes it pass.

---

### Task 2: Implement the `register_vmap` rule (K-loop)

**Files:**
- Modify: `aimnet/kernels/conv_sv_2d_sp_wp.py` — insert a new block between the existing `register_autograd` calls (currently ending at line 453) and the `# ============================================================================= \n # Public API` separator (currently line 456).

- [ ] **Step 1: Add the rule**

Insert exactly this block after line 453 (`register_autograd("aimnet::conv_sv_2d_sp_bwd", ...)`) and before the `# Public API` banner:

```python
# =============================================================================
# vmap Registration
# =============================================================================


@torch.library.register_vmap("aimnet::conv_sv_2d_sp_bwd_bwd")
def _vmap_conv_sv_2d_sp_bwd_bwd(info, in_dims, grad_output, grad2_a, grad2_g, a, idx, g):
    """vmap rule for the double-backward primitive.

    Used by Hessian-via-vmap paths (torch.func.hessian, is_grads_batched=True).
    The realistic in_dims is (None, 0, 0, None, None, None) — only the cotangents
    flowing into the second backward carry the vmap batch dim. The rule still
    handles arbitrary in_dims for robustness.

    Strategy: K-loop. Folding K into the kernel's leading B dim is unsafe because
    the kernels rely on a single padding-row sentinel at index B-1; stacking K
    copies would scatter padding rows. The K calls queue async on the CUDA
    stream, so the loop's per-call cost is dominated by GPU work, not Python.
    """
    K = info.batch_size

    def _slice(t: Tensor, d: int | None, k: int) -> Tensor:
        if d is None:
            return t
        return t.movedim(d, 0)[k]

    out0: list[Tensor] = []
    out1: list[Tensor] = []
    out2: list[Tensor] = []
    for k in range(K):
        outs = torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd(
            _slice(grad_output, in_dims[0], k),
            _slice(grad2_a, in_dims[1], k),
            _slice(grad2_g, in_dims[2], k),
            _slice(a, in_dims[3], k),
            _slice(idx, in_dims[4], k),
            _slice(g, in_dims[5], k),
        )
        out0.append(outs[0])
        out1.append(outs[1])
        out2.append(outs[2])

    return (
        [torch.stack(out0, dim=0), torch.stack(out1, dim=0), torch.stack(out2, dim=0)],
        [0, 0, 0],
    )
```

Notes for the implementer:
- Do NOT add a vmap rule for `aimnet::conv_sv_2d_sp_fwd` or `aimnet::conv_sv_2d_sp_bwd` here. Only `bwd_bwd` is hit by the Hessian path. If a later test fails with a similar `Batching rule not implemented` error naming a different op, add the analogous rule then — not preemptively.
- The output structure `[T, T, T]` matches the op's `register_fake` return list (line 386-394). Returning a tuple instead of a list will fail tree-flattening — keep it a list.
- Type-checking: the file already has `# type: ignore` at line 22, so no extra annotations are required.

- [ ] **Step 2: Run Task 1's test and confirm it now passes**

Run: `pytest tests/test_conv_sv_2d_sp.py::TestConvSV2dSP::test_vmap_bwd_bwd_kernel_rule -v`

Expected: PASS, all four allclose assertions green.

- [ ] **Step 3: Run the full kernel test module to confirm no regression**

Run: `pytest tests/test_conv_sv_2d_sp.py -v -m gpu`

Expected: every existing test still PASS (forward, backward, double-backward, shapes, op-registration, padding) plus the new vmap test. If any existing test regresses, STOP — the kernel registration site change has reordered something it shouldn't have.

- [ ] **Step 4: Commit the rule**

```bash
git add aimnet/kernels/conv_sv_2d_sp_wp.py
git commit -m "feat(kernels): register vmap rule for conv_sv_2d_sp_bwd_bwd"
```

---

### Task 3: End-to-end `torch.func.hessian` test on water

**Files:**
- Modify: `tests/test_calculator_gpu.py` — append a new class `TestVectorizedHessian` at the bottom of the file.

- [ ] **Step 1: Add the failing-then-passing test**

Append this class at the end of `tests/test_calculator_gpu.py` (it's GPU-only via the module-level `pytestmark`):

```python
class TestVectorizedHessian:
    """End-to-end Hessian via vmap-based paths through the kernel.

    These tests exercise the vmap rule on aimnet::conv_sv_2d_sp_bwd_bwd added in
    PR A. They do NOT touch AIMNet2Calculator.calculate_hessian (still the loop
    path); they call torch.func.hessian / torch.autograd.grad directly on a
    closure that uses the calculator for energy.
    """

    def _water_inputs(self, device):
        coords = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        nums = torch.tensor([[8, 1, 1]], device=device)
        charge = torch.tensor([0.0], device=device)
        return coords, nums, charge

    def test_func_hessian_matches_internal(self):
        """torch.func.hessian on the energy closure matches the calculator's loop hessian."""
        device = torch.device("cuda")
        calc = AIMNet2Calculator("aimnet2", device=device, nb_threshold=0)
        coords, nums, charge = self._water_inputs(device)

        H_internal = calc(
            {"coord": coords.unsqueeze(0).clone(), "numbers": nums, "charge": charge},
            hessian=True,
        )["hessian"]
        # H_internal shape: (N, 3, N, 3) with N=3 → (3, 3, 3, 3)
        assert H_internal.shape == (3, 3, 3, 3)

        def energy_fn(x):
            out = calc({"coord": x.unsqueeze(0), "numbers": nums, "charge": charge}, forces=False)
            return out["energy"][0]

        H_func = torch.func.hessian(energy_fn)(coords)
        assert H_func.shape == (3, 3, 3, 3)
        assert torch.isfinite(H_func).all()
        assert (H_internal - H_func).abs().max().item() < 5e-3
```

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_calculator_gpu.py::TestVectorizedHessian::test_func_hessian_matches_internal -v -m gpu`

Expected: PASS. Tolerance `5e-3` matches the existing internal/external Hessian regression at `tests/test_calculator.py:318`.

If this test fails with a *different* op missing a vmap rule (e.g. `Batching rule not implemented for aimnet::conv_sv_2d_sp_bwd`), that's the signal mentioned in Task 2 Step 1 — add the analogous K-loop rule for that op in `conv_sv_2d_sp_wp.py`, then re-run. The shape of that contingent rule is identical: `(info, in_dims, grad_output, a, idx, g)` for `bwd`, returning `[grad_a, grad_g]` with `out_dims [0, 0]`. Do not add it speculatively — only if this step demands it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_calculator_gpu.py
git commit -m "test(calculator): torch.func.hessian end-to-end on water (CUDA)"
```

---

### Task 4: End-to-end `is_grads_batched=True` test on water

**Files:**
- Modify: `tests/test_calculator_gpu.py` — add a second method to `TestVectorizedHessian`.

This task verifies the OTHER vectorized path (the one PR B will actually use) so that PR B's swap is mechanically safe.

- [ ] **Step 1: Add the test**

Append inside `TestVectorizedHessian`:

```python
    def test_is_grads_batched_hessian_matches_internal(self):
        """torch.autograd.grad(..., is_grads_batched=True) hessian matches the loop hessian.

        Mirrors the proposed PR B implementation strategy. The shape juggling
        (slice off padding rows/cols) follows the existing loop in
        aimnet/calculators/calculator.py:1135-1142.
        """
        device = torch.device("cuda")
        calc = AIMNet2Calculator("aimnet2", device=device, nb_threshold=0)
        coords, nums, charge = self._water_inputs(device)

        H_internal = calc(
            {"coord": coords.unsqueeze(0).clone(), "numbers": nums, "charge": charge},
            hessian=True,
        )["hessian"]

        # Re-run the calculator in the same shape the loop expects: coord with
        # an extra padding atom row appended (the "+1 row" the loop slices off).
        coord = coords.unsqueeze(0).clone().requires_grad_(True)
        out = calc({"coord": coord, "numbers": nums, "charge": charge}, forces=True)
        forces = out["forces"]
        coord_grad = calc._saved_for_grad["coord"]  # shape (N+1, 3) with padding row

        n = forces.numel()
        eye = torch.eye(n, device=device, dtype=forces.dtype)
        H_batched = -torch.autograd.grad(
            forces.flatten(),
            coord_grad,
            grad_outputs=eye,
            retain_graph=True,
            is_grads_batched=True,
        )[0]
        # Reshape to (n, N+1, 3) -> (-1, 3, N+1, 3), drop padding row+col
        H_batched = H_batched.view(-1, 3, coord_grad.shape[0], 3)[:-1, :, :-1, :]

        assert H_batched.shape == H_internal.shape
        assert torch.isfinite(H_batched).all()
        assert (H_internal - H_batched).abs().max().item() < 5e-3
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_calculator_gpu.py::TestVectorizedHessian::test_is_grads_batched_hessian_matches_internal -v -m gpu`

Expected: PASS. If it fails with `Batching rule not implemented` for a different op, follow the contingent-rule note in Task 3 Step 2.

If it fails because `calc._saved_for_grad["coord"]` does not have a graph (e.g. `forces=True` released it), check `aimnet/calculators/calculator.py:1115-1132` — that is the path that populates `_saved_for_grad["coord"]` and computes forces. Forces are computed with `create_graph=False` by default in that block; this test needs `create_graph=True` so the second backward has a graph. If reproducing the failure shows that, the test must be amended to call the calculator with the kwargs that retain the graph (search for `_create_graph` in that file — there should be a path that exposes it; if not, this becomes a finding for PR B and the test should be marked `xfail("PR B builds the create_graph path")` rather than relaxed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_calculator_gpu.py
git commit -m "test(calculator): is_grads_batched hessian end-to-end on water (CUDA)"
```

---

### Task 5: Verify the regression suite is intact

This task runs no new code — it just confirms PR A is non-disruptive.

- [ ] **Step 1: Calculator tests (CPU + CUDA where available)**

Run: `pytest tests/test_calculator.py -v`

Expected: all PASS, including `test_external_hessian_nonzero` (line 280) and `test_external_hessian_matches_internal` (line 300). Neither test uses `vectorize=True`, so the kernel registration changes here do not affect them — but a green run is the contract.

- [ ] **Step 2: Kernel direct tests**

Run: `pytest tests/test_conv_sv_2d_sp.py -v -m gpu`

Expected: every test PASS, including the new `test_vmap_bwd_bwd_kernel_rule`.

- [ ] **Step 3: GPU calculator tests**

Run: `pytest tests/test_calculator_gpu.py -v -m gpu`

Expected: every test PASS, including the new `TestVectorizedHessian` class.

- [ ] **Step 4: Registry tests (project-rule contract per CLAUDE.md)**

Run: `pytest tests/test_model_registry.py tests/test_hf_hub.py`

Expected: PASS. PR A does not touch the registry, but the project CLAUDE.md requires this gate before any PR.

- [ ] **Step 5: If all four steps green, no further commit. Otherwise STOP.**

If any pre-existing test regresses, do not paper over it — bisect against the last green commit and identify whether the kernel file edit accidentally re-ordered registration. The `register_vmap` block must come AFTER both `register_autograd` calls (because `register_vmap` is for the op `conv_sv_2d_sp_bwd_bwd`, and the autograd of `bwd` references that same op via `torch.ops.aimnet.conv_sv_2d_sp_bwd_bwd` at line 438 — registration order matters).

---

## Acceptance criteria for PR A

All must hold before PR A can land:

1. `pytest tests/test_conv_sv_2d_sp.py -v -m gpu` → all green, including `test_vmap_bwd_bwd_kernel_rule`.
2. `pytest tests/test_calculator.py -v` → all green; the loop-based Hessian regressions are unchanged.
3. `pytest tests/test_calculator_gpu.py -v -m gpu` → all green, including both new `TestVectorizedHessian` methods.
4. `pytest tests/test_model_registry.py tests/test_hf_hub.py` → all green (CLAUDE.md gate).
5. No change to `aimnet/calculators/calculator.py`. No change to `aimnet/modules/aev.py`. No change to wrappers. The only files modified are `aimnet/kernels/conv_sv_2d_sp_wp.py`, `tests/test_conv_sv_2d_sp.py`, `tests/test_calculator_gpu.py`.

## Out of scope (PR B)

- Replacing the row-wise loop in `AIMNet2Calculator.calculate_hessian` (`aimnet/calculators/calculator.py:1135-1142`) with the vectorized form. PR B owns this swap and the caffeine N=24 GPU benchmark target (~10× minimum, ~20–25× expected).
- Optimizing the vmap rule beyond the K-loop. A fully-batched kernel that respects the padding sentinel is a possible follow-up if the K-loop dominates wall-clock for very large K (it almost certainly will not at N≤100).
- A vmap rule for `aimnet::conv_sv_2d_sp_bwd` or `aimnet::conv_sv_2d_sp_fwd`. Add only if Task 3/4 demand it (see contingent-rule note). Today, only `bwd_bwd` raises.

## Related

- Parent two-PR plan: `docs/superpowers/plans/2026-04-26-vectorize-calculate-hessian.md`
- Sibling Sella plan: `docs/superpowers/plans/2026-04-26-sella-integration.md`
- Kernel source: `aimnet/kernels/conv_sv_2d_sp_wp.py`
- Existing direct-kernel tests: `tests/test_conv_sv_2d_sp.py`
- AEV layer (path selector): `aimnet/modules/aev.py:151-170`
- Calculator (Hessian site, untouched in PR A): `aimnet/calculators/calculator.py:1131-1142`
- PyTorch reference: [Custom Operators tutorial — vmap support](https://pytorch.org/tutorials/advanced/python_custom_ops.html#adding-vmap-support-to-an-operator)
