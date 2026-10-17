"""CUDA-graph replay of repeated AIMNet2Calculator evaluations (``aimnet.calculators.cuda_graph``).

Each test compares a calculator with ``cuda_graph=True`` against the regular path (``cuda_graph=False``) of the same
model: same keys, shapes, dtypes and devices; values at float32 rounding level.

All tests in this module require CUDA and are marked with @pytest.mark.gpu.
"""

import warnings

import numpy as np
import pytest
import torch
from conftest import CAFFEINE_FILE, CIF_SPIRO, TAXOL_FILE, load_mol

from aimnet.calculators import AIMNet2Calculator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

E_ATOL = 1e-3  # eV, molecular energies of 1e3-1e5 eV in float32 atomic contributions
F_ATOL = 1e-4  # eV/A
F_RTOL = 1e-5
Q_ATOL = 1e-5  # e


def _water_cluster(n_side: int, spacing: float = 3.1, seed: int = 0) -> dict:
    """n_side**3 randomly oriented waters on a cubic grid (192 atoms for n_side=4: neighbor-list mode)."""
    rng = np.random.default_rng(seed)
    ang = np.deg2rad(104.52)
    water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [0.9572 * np.cos(ang), 0.9572 * np.sin(ang), 0.0]])
    coords = []
    for idx in np.ndindex(n_side, n_side, n_side):
        rot, r = np.linalg.qr(rng.normal(size=(3, 3)))
        rot = rot * np.sign(np.diag(r))
        coords.append(water @ rot.T + spacing * np.asarray(idx, dtype=float))
    numbers = np.tile([8, 1, 1], n_side**3)
    return {"coord": np.concatenate(coords), "numbers": numbers, "charge": 0.0}


def _jitter(coord: np.ndarray, step: int, scale: float = 0.02) -> np.ndarray:
    return coord + scale * np.random.default_rng(step).normal(size=coord.shape)


def _cuda_input(data: dict) -> dict:
    return {
        "coord": torch.as_tensor(data["coord"], dtype=torch.float32, device="cuda"),
        "numbers": torch.as_tensor(data["numbers"], dtype=torch.int64, device="cuda"),
        "charge": torch.as_tensor(data["charge"], dtype=torch.float32, device="cuda").reshape(-1),
    }


def _assert_same(out: dict, ref: dict) -> None:
    assert out.keys() == ref.keys()
    for k in ref:
        assert out[k].shape == ref[k].shape, k
        assert out[k].dtype == ref[k].dtype, k
        assert out[k].device == ref[k].device, k
    torch.testing.assert_close(out["energy"], ref["energy"], atol=E_ATOL, rtol=0.0)
    torch.testing.assert_close(out["charges"], ref["charges"], atol=Q_ATOL, rtol=0.0)
    if "forces" in ref:
        torch.testing.assert_close(out["forces"], ref["forces"], atol=F_ATOL, rtol=F_RTOL)


def _assert_close_all(out: dict, ref: dict) -> None:
    """Every key, at the regular path's run-to-run level on CUDA (atomic accumulation in D3 / neighbor lists)."""
    assert out.keys() == ref.keys()
    for k in ref:
        atol, rtol = {"energy": (E_ATOL, 0.0), "charges": (Q_ATOL, 0.0)}.get(k, (F_ATOL, F_RTOL))
        torch.testing.assert_close(out[k], ref[k], atol=atol, rtol=rtol)


def _n_engines(calc: AIMNet2Calculator) -> int:
    return len(calc._cuda_graph._engines)


@pytest.fixture(scope="module", params=["aimnet2", "aimnet2-2025"])
def calc_pair(request):
    graph = AIMNet2Calculator(request.param, cuda_graph=True)
    regular = AIMNet2Calculator(request.param, cuda_graph=False)
    assert graph._cuda_graph is not None
    assert regular._cuda_graph is None
    return graph, regular


@pytest.fixture
def calcs(calc_pair):
    graph, regular = calc_pair
    graph._cuda_graph.clear()
    return graph, regular


@pytest.fixture(scope="module")
def caffeine():
    return load_mol(CAFFEINE_FILE)


class TestReplayMatchesRegularPath:
    @pytest.mark.parametrize("source", ["caffeine", "taxol", "water"])
    def test_single_molecule(self, calcs, source):
        """Quick-start input [N, 3]: the dense engine for N <= 120 atoms, the neighbor-list engine above."""
        graph, regular = calcs
        data = {"caffeine": lambda: load_mol(CAFFEINE_FILE), "taxol": lambda: load_mol(TAXOL_FILE)}.get(
            source, lambda: _water_cluster(4)
        )()
        for step in range(4):
            x = {**data, "coord": _jitter(data["coord"], step)}
            _assert_same(graph(x, forces=True), regular(x, forces=True))
            assert _n_engines(graph) == (0 if step == 0 else 1)

    def test_batch(self, calcs, caffeine):
        graph, regular = calcs
        B = 4
        base = np.broadcast_to(caffeine["coord"], (B, *caffeine["coord"].shape))
        numbers = torch.as_tensor(np.tile(caffeine["numbers"], (B, 1)), device="cuda")
        charge = torch.zeros(B, device="cuda")
        for step in range(3):
            x = {"coord": torch.as_tensor(_jitter(base, step), device="cuda"), "numbers": numbers, "charge": charge}
            _assert_same(graph(x, forces=True), regular(x, forces=True))
        assert _n_engines(graph) == 1

    def test_energy_only(self, calcs, caffeine):
        graph, regular = calcs
        for step in range(3):
            x = {**caffeine, "coord": _jitter(caffeine["coord"], step)}
            out = graph(x)
            assert "forces" not in out
            _assert_same(out, regular(x))
        assert _n_engines(graph) == 1
        graph(caffeine, forces=True)
        graph(caffeine, forces=True)
        assert _n_engines(graph) == 2  # energy-only and energy + forces graphs are separate

    @pytest.mark.parametrize("kind", ["numpy", "list", "cpu_tensor", "cuda_tensor", "with_mult", "ase_style"])
    def test_input_types(self, calcs, caffeine, kind):
        graph, regular = calcs
        if kind == "numpy":
            x = caffeine
        elif kind == "list":
            x = {"coord": caffeine["coord"].tolist(), "numbers": caffeine["numbers"].tolist(), "charge": 0}
        elif kind == "cpu_tensor":
            x = {k: torch.as_tensor(v) for k, v in caffeine.items()}
        elif kind == "cuda_tensor":
            x = _cuda_input(caffeine)
        elif kind == "with_mult":
            x = {**caffeine, "mult": 1.0}
        else:  # the ASE wrapper: cached CUDA numbers and 0-d charge/mult, fresh unsqueezed views on every call
            t = _cuda_input(caffeine)
            t["charge"] = torch.tensor(0.0, device="cuda")
            mult = torch.tensor(1.0, device="cuda")
            x = None
        for step in range(3):
            if kind == "ase_style":
                coord = torch.as_tensor(_jitter(caffeine["coord"], step), dtype=torch.float32, device="cuda")
                y = {"coord": coord, "numbers": t["numbers"], "charge": t["charge"], "mult": mult}
                y = {k: v.unsqueeze(0) for k, v in y.items()}
            else:
                y = x
            _assert_same(graph(y, forces=True), regular(y, forces=True))
        assert _n_engines(graph) == 1

    def test_results_are_not_overwritten(self, calcs, caffeine):
        graph, _ = calcs
        outs = [graph({**caffeine, "coord": _jitter(caffeine["coord"], step)}, forces=True) for step in range(4)]
        kept = [{k: v.clone() for k, v in o.items()} for o in outs]
        graph({**caffeine, "coord": _jitter(caffeine["coord"], 99)}, forces=True)
        for o, k in zip(outs, kept, strict=True):
            for key in o:
                assert torch.equal(o[key], k[key])
        assert not torch.equal(outs[2]["forces"], outs[3]["forces"])


class TestRegularPathInputs:
    """Inputs and requests the CUDA-graph replay does not handle run on the regular path, unchanged."""

    def _check(self, graph, regular, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                _assert_close_all(graph(data, **kwargs), regular(data, **kwargs))
        assert _n_engines(graph) == 0

    def test_periodic(self, calcs):
        graph, regular = calcs
        self._check(graph, regular, load_mol(CIF_SPIRO), forces=True, stress=True)

    def test_mol_idx(self, calcs, caffeine):
        graph, regular = calcs
        n = len(caffeine["numbers"])
        data = {
            "coord": np.concatenate([caffeine["coord"], caffeine["coord"] + 20.0]),
            "numbers": np.concatenate([caffeine["numbers"]] * 2),
            "charge": [0.0, 0.0],
            "mol_idx": np.repeat([0, 1], n),
        }
        self._check(graph, regular, data, forces=True)

    def test_padded_batch(self, calcs, caffeine):
        graph, regular = calcs
        numbers = np.stack([caffeine["numbers"], caffeine["numbers"]])
        numbers[1, -1] = 0
        data = {"coord": np.stack([caffeine["coord"]] * 2), "numbers": numbers, "charge": [0.0, 0.0]}
        self._check(graph, regular, data, forces=True)

    def test_hessian(self, calcs, caffeine):
        graph, regular = calcs
        self._check(graph, regular, caffeine, forces=True, hessian=True)

    def test_coord_requires_grad(self, calcs, caffeine):
        graph, regular = calcs
        data = {**caffeine, "coord": torch.tensor(caffeine["coord"], dtype=torch.float32, requires_grad=True)}
        self._check(graph, regular, data, forces=True)


class TestLayoutAndSettingsChanges:
    def test_numbers_changed_in_place(self, calcs, caffeine):
        """The same CUDA ``numbers`` tensor edited in place is a new layout (``_version`` guard)."""
        graph, regular = calcs
        x = _cuda_input(caffeine)
        for _ in range(2):
            graph(x, forces=True)
        j = int(np.flatnonzero(caffeine["numbers"] != caffeine["numbers"][0])[0])
        x["numbers"][[0, j]] = x["numbers"][[j, 0]].clone()
        for _ in range(2):
            _assert_same(graph(x, forces=True), regular(x, forces=True))

    def test_lr_cutoff_uses_regular_path(self, caffeine):
        graph = AIMNet2Calculator("aimnet2", cuda_graph=True)
        regular = AIMNet2Calculator("aimnet2", cuda_graph=False)
        for _ in range(2):
            graph(caffeine, forces=True)
        assert _n_engines(graph) == 1
        for calc in (graph, regular):
            calc.set_lr_cutoff(4.0)  # truncates the full Coulomb: not what the engines compute
        for _ in range(3):
            _assert_close_all(graph(caffeine, forces=True), regular(caffeine, forces=True))

    def test_dsf_uses_regular_path(self, caffeine):
        graph = AIMNet2Calculator("aimnet2", cuda_graph=True)
        regular = AIMNet2Calculator("aimnet2", cuda_graph=False)
        for calc in (graph, regular):
            calc.set_lrcoulomb_method("dsf")
        for _ in range(3):
            _assert_close_all(graph(caffeine, forces=True), regular(caffeine, forces=True))
        assert _n_engines(graph) == 0

    def test_neighbor_capacity_rebuild(self, calcs):
        """A denser geometry than the captured one overflows the neighbor capacity: the engine is rebuilt."""
        graph, regular = calcs
        data = _water_cluster(4)
        for _ in range(2):
            graph(data, forces=True)
        (entry,) = graph._cuda_graph._engines.values()
        dense = _water_cluster(4, spacing=2.6)  # same molecules, closer together: ~1.7x the neighbors
        for _ in range(2):
            _assert_same(graph(dense, forces=True), regular(dense, forces=True))
        assert entry.runner.rebuilds == 1
        _assert_same(graph(data, forces=True), regular(data, forces=True))

    def test_engine_disagreeing_with_regular_path_is_not_used(self, calcs, caffeine, monkeypatch):
        """The capture-time check: an engine whose forces are 1% off is rejected, the regular path answers."""
        from aimnet.calculators.cuda_graph import dense

        graph, regular = calcs
        forward = dense.StaticDenseAIMNet2.forward

        def off_by_one_percent(self, coord):
            e, f, q = forward(self, coord)
            return e, f * 1.01, q

        monkeypatch.setattr(dense.StaticDenseAIMNet2, "forward", off_by_one_percent)
        graph._cuda_graph._warned = False
        with pytest.warns(RuntimeWarning, match="disagrees with the regular path"):
            for _ in range(3):
                _assert_same(graph(caffeine, forces=True), regular(caffeine, forces=True))
        assert _n_engines(graph) == 0

    def test_unsupported_species_still_raises(self, calcs, caffeine):
        graph, _ = calcs
        if not (graph.metadata or {}).get("implemented_species"):
            pytest.skip("model declares no implemented_species")
        bad = {**caffeine, "numbers": np.where(caffeine["numbers"] == 1, 100, caffeine["numbers"])}
        for _ in range(3):
            with pytest.raises(ValueError, match="not in this model's"):
                graph(bad, forces=True)


class TestEnablement:
    def test_disabled_by_environment(self, monkeypatch):
        monkeypatch.setenv("AIMNET_CUDA_GRAPH", "0")
        assert AIMNet2Calculator("aimnet2")._cuda_graph is None
        assert AIMNet2Calculator("aimnet2", cuda_graph=True)._cuda_graph is not None

    def test_cpu_calculator(self):
        assert AIMNet2Calculator("aimnet2", device="cpu")._cuda_graph is None

    def test_open_shell_model_uses_regular_path(self):
        assert AIMNet2Calculator("aimnet2-nse")._cuda_graph is None


class TestAEVSVDenseKernel:
    """Triton AEV basis vs the PyTorch expression of the dense engine: forward bitwise, gradients at float32 level."""

    @pytest.mark.parametrize("system", ["caffeine", "taxol", "caffeine_x3"])
    def test_matches_pytorch_expression(self, calc_pair, system):
        pytest.importorskip("triton")
        from aimnet.calculators.cuda_graph.dense import StaticDenseAIMNet2

        graph, _ = calc_pair
        mol = load_mol(TAXOL_FILE if system == "taxol" else CAFFEINE_FILE)
        B = 3 if system == "caffeine_x3" else 1
        x = torch.as_tensor(np.stack([_jitter(mol["coord"], b) for b in range(B)]), dtype=torch.float32, device="cuda")
        z = torch.as_tensor(np.tile(mol["numbers"], (B, 1)), device="cuda")
        q = torch.zeros(B, device="cuda")
        kernel = StaticDenseAIMNet2(graph, z, q)
        torch_expr = StaticDenseAIMNet2(graph, z, q, fused_aev=False)
        assert kernel._aev_kernel is not None
        assert torch_expr._aev_kernel is None
        x = x.requires_grad_(True)
        g_k = kernel._aev_bmm_layout({**kernel.static, "coord": x})["g_L"]
        g_t = torch_expr._aev_bmm_layout({**torch_expr.static, "coord": x})["g_L"]
        assert torch.equal(g_k, g_t)
        w = torch.randn_like(g_t)
        (dx_k,) = torch.autograd.grad((g_k * w).sum(), x)
        (dx_t,) = torch.autograd.grad((g_t * w).sum(), x)
        assert float((dx_k - dx_t).abs().max() / dx_t.abs().max()) < 1e-5


class TestFullCoulombKernel:
    """Warp full-Coulomb rows vs a float64 ground truth, next to the float32 math of ``LRCoulomb.coul_simple``."""

    @staticmethod
    def _system(B, N, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        T = B * N
        x = torch.rand(T + 1, 3, generator=g) * (N ** (1 / 3)) * 3.0
        x[-1] = 0
        q = torch.randn(T + 1, generator=g) * 0.3
        q[-1] = 0
        weights = torch.randn(B, generator=g, dtype=torch.float64)
        mol = torch.cat((torch.arange(B).repeat_interleave(N), torch.tensor([B - 1])))
        return x.cuda(), q.cuda(), weights.cuda(), mol.cuda()

    @staticmethod
    def _dense_rows(x, q, B, N, dtype):
        T = B * N
        e = torch.zeros(T + 1, dtype=torch.float64, device=x.device)
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        for b in range(B):
            s = slice(b * N, (b + 1) * N)
            diff = x[s][None, :, :] - x[s][:, None, :]
            d = torch.where(eye, torch.ones_like(eye, dtype=dtype), (diff.square().sum(-1) + eye.to(dtype)).sqrt())
            pair = torch.where(eye, torch.zeros_like(d), q[s, None] * q[None, s] / d)
            e = e.index_add(0, torch.arange(b * N, (b + 1) * N, device=x.device), pair.sum(-1, dtype=torch.float64))
        return e

    @staticmethod
    def _grads(fn, x, q, weights, mol, dtype):
        xx = x.clone().to(dtype).requires_grad_(True)
        qq = q.clone().to(dtype).requires_grad_(True)
        e = fn(xx, qq)
        gx, gq = torch.autograd.grad((e * weights[mol]).sum(), (xx, qq))
        return e, gx, gq

    @pytest.mark.parametrize(("B", "N"), [(1, 7), (3, 50), (2, 700), (1, 3000)])
    def test_accuracy(self, B, N):
        from aimnet.kernels.coulomb_wp import full_coulomb_rows

        x, q, w, mol = self._system(B, N)
        T = B * N
        ref64 = self._grads(lambda a, b: self._dense_rows(a, b, B, N, torch.float64), x, q, w, mol, torch.float64)
        ref32 = self._grads(lambda a, b: self._dense_rows(a, b, B, N, torch.float32), x, q, w, mol, torch.float32)
        kernel = self._grads(lambda a, b: full_coulomb_rows(a, b, N, T), x, q, w, mol, torch.float32)

        def rel(a, b):
            return float((a.double() - b).abs().max() / b.abs().max())

        for name, k, r32, r64 in zip(("e", "dR", "dq"), kernel, ref32, ref64, strict=True):
            assert rel(k, r64) <= 2.0 * max(rel(r32, r64), torch.finfo(torch.float32).eps), name
        assert float(kernel[0][-1]) == 0.0
        assert float(kernel[2][-1]) == 0.0
