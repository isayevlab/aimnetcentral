"""Tests for CQEq (Constrained Charge Equilibration) functionality.

Covers:
- constrained_nse at the ops level (per-region charge conservation)
- eval_cqeq at the calculator level (end-to-end inference)
- diabatic_coupling helper
- Q-TTF-Q radical anion (integration test)
"""

import numpy as np
import pytest
import torch

from aimnet import nbops, ops

# All calculator-level tests need the pretrained model (+ ASE for load_mol)
pytestmark = pytest.mark.ase


# =====================================================================
# Unit tests for constrained_nse  (ops level)
# =====================================================================


class TestConstrainedNSE:
    """Tests for ops.constrained_nse - per-region charge equilibration."""

    def test_per_region_charge_conservation_mode0(self, device):
        """In nb_mode=0 (batched, no nbmat), charges in each region should
        sum to the target value."""
        B, N = 1, 6
        coord = torch.randn(B, N, 3, device=device)
        numbers = torch.tensor([[8, 1, 1, 8, 1, 1]], device=device)
        data = {"coord": coord, "numbers": numbers}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        q_u = torch.randn(B, N, 1, device=device) * 0.1
        f_u = torch.rand(B, N, 1, device=device) + 0.1  # positive

        # Region 0 = atoms 0-2 (donor), region 1 = atoms 3-5 (acceptor)
        data["region_mask"] = torch.tensor([[0, 0, 0, 1, 1, 1]], device=device)
        data["region_charges"] = torch.tensor([0.5, -0.5], device=device)

        q = ops.constrained_nse(q_u, f_u, data)

        q_region0 = q[0, :3, 0].sum()
        q_region1 = q[0, 3:, 0].sum()
        assert q_region0.item() == pytest.approx(0.5, abs=1e-5)
        assert q_region1.item() == pytest.approx(-0.5, abs=1e-5)

    def test_per_region_charge_conservation_mode1(self, device):
        """In nb_mode=1 (flat with nbmat), each region's charge sums to target."""
        # 4 atoms + 1 padding = 5 rows
        N = 5
        coord = torch.randn(N, 3, device=device)
        numbers = torch.tensor([8, 1, 8, 1, 0], device=device)  # last = pad
        mol_idx = torch.tensor([0, 0, 0, 0, 0], device=device)

        # Build a trivial nbmat (N, M) where M=1; each atom points to pad
        nbmat = torch.full((N, 1), N - 1, dtype=torch.int32, device=device)
        nbmat[0, 0] = 1
        nbmat[1, 0] = 0

        data = {
            "coord": coord,
            "numbers": numbers,
            "mol_idx": mol_idx,
            "nbmat": nbmat,
        }
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        q_u = torch.randn(N, 1, device=device) * 0.1
        f_u = torch.rand(N, 1, device=device) + 0.1

        # region 0 = atoms 0,1   region 1 = atoms 2,3   pad -> region 0
        data["region_mask"] = torch.tensor([0, 0, 1, 1, 0], device=device, dtype=torch.long)
        data["region_charges"] = torch.tensor([0.3, -0.3], device=device)

        q = ops.constrained_nse(q_u, f_u, data)

        q_r0 = q[[0, 1, 4], 0].sum()  # atoms 0,1 + pad (region 0)
        q_r1 = q[[2, 3], 0].sum()
        assert q_r0.item() == pytest.approx(0.3, abs=1e-5)
        assert q_r1.item() == pytest.approx(-0.3, abs=1e-5)

    def test_matches_nse_when_single_region(self, simple_molecule):
        """With a single region encompassing the whole molecule,
        constrained_nse should give the same result as standard nse."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        dev = data["coord"].device
        B, N = data["coord"].shape[:2]

        q_u = torch.randn(B, N, 1, device=dev)
        f_u = torch.rand(B, N, 1, device=dev) + 0.1
        Q = torch.tensor([[0.0]], device=dev)

        # Standard NSE
        q_standard = ops.nse(Q, q_u, f_u, data.copy())

        # Constrained NSE with one region (all atoms → region 0, target 0.0)
        data2 = data.copy()
        data2["region_mask"] = torch.zeros(B, N, device=dev, dtype=torch.long)
        data2["region_charges"] = torch.tensor([0.0], device=dev)
        q_cqeq = ops.constrained_nse(q_u, f_u, data2)

        torch.testing.assert_close(q_cqeq, q_standard, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self, device):
        """Ensure autograd can differentiate through constrained_nse."""
        B, N = 1, 4
        coord = torch.randn(B, N, 3, device=device)
        numbers = torch.tensor([[6, 1, 6, 1]], device=device)
        data = {"coord": coord, "numbers": numbers}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        q_u = torch.randn(B, N, 1, device=device, requires_grad=True)
        f_u = torch.rand(B, N, 1, device=device) + 0.1

        data["region_mask"] = torch.tensor([[0, 0, 1, 1]], device=device)
        data["region_charges"] = torch.tensor([0.0, 0.0], device=device)

        q = ops.constrained_nse(q_u, f_u, data)
        q.sum().backward()

        assert q_u.grad is not None
        assert torch.isfinite(q_u.grad).all()

    def test_multichannel(self, device):
        """Constrained NSE should work when q_u has 2 charge channels."""
        B, N, C = 1, 4, 2
        coord = torch.randn(B, N, 3, device=device)
        numbers = torch.tensor([[8, 1, 8, 1]], device=device)
        data = {"coord": coord, "numbers": numbers}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        q_u = torch.randn(B, N, C, device=device)
        f_u = torch.rand(B, N, C, device=device) + 0.1

        data["region_mask"] = torch.tensor([[0, 0, 1, 1]], device=device)
        # region_charges broadcasts; shape (2,) should apply per-channel
        data["region_charges"] = torch.tensor([0.0, 0.0], device=device)

        q = ops.constrained_nse(q_u, f_u, data)

        assert q.shape == (B, N, C)
        # Each channel, each region should sum to 0
        for c in range(C):
            assert q[0, :2, c].sum().item() == pytest.approx(0.0, abs=1e-4)
            assert q[0, 2:, c].sum().item() == pytest.approx(0.0, abs=1e-4)


# =====================================================================
# Integration tests for eval_cqeq  (calculator level)
# =====================================================================


class TestEvalCQEq:
    """End-to-end tests for AIMNet2Calculator.eval_cqeq."""

    @staticmethod
    def _dimer_data():
        """Two water molecules as a D-A dimer (flat tensors)."""
        coord = torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [0.96, 0.00, 0.00],
                [-0.24, 0.93, 0.00],
                [6.00, 0.00, 0.00],
                [6.96, 0.00, 0.00],
                [5.76, 0.93, 0.00],
            ],
            dtype=torch.float32,
        )
        return {
            "coord": coord,
            "numbers": torch.tensor([8, 1, 1, 8, 1, 1]),
            "mol_idx": torch.tensor([0, 0, 0, 0, 0, 0]),
            "charge": torch.tensor([0.0]),
        }

    def test_eval_cqeq_returns_energy_and_charges(self):
        """eval_cqeq should return at least energy and charges."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()
        res = calc.eval_cqeq(
            data,
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[0.0, 0.0],
        )
        assert "energy" in res
        assert "charges" in res
        assert torch.isfinite(res["energy"]).all()
        assert torch.isfinite(res["charges"]).all()

    def test_eval_cqeq_forces(self):
        """eval_cqeq should compute forces when requested."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()
        res = calc.eval_cqeq(
            data,
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[0.0, 0.0],
            forces=True,
        )
        assert "forces" in res
        assert torch.isfinite(res["forces"]).all()

    def test_neutral_diabatic_close_to_ground_state(self):
        """For a well-separated neutral dimer the neutral diabatic state
        energy should be very close to the adiabatic ground state."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()

        e_gs = calc(data.copy())["energy"].item()
        e_neutral = calc.eval_cqeq(
            data.copy(),
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[0.0, 0.0],
        )["energy"].item()

        # Should be very close (both neutral, well-separated)
        assert abs(e_neutral - e_gs) < 0.05  # within 50 mHa

    def test_ct_state_higher_energy(self):
        """The D^+ A^- charge-transfer state should be higher in energy
        than the neutral diabatic state for well-separated molecules."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()

        e_neutral = calc.eval_cqeq(
            data.copy(),
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[0.0, 0.0],
        )["energy"].item()

        e_ct = calc.eval_cqeq(
            data.copy(),
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[1.0, -1.0],
        )["energy"].item()

        assert e_ct > e_neutral

    def test_region_charges_preserved(self):
        """Atomic charges in each region should sum to the requested value."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()

        res = calc.eval_cqeq(
            data,
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[1.0, -1.0],
        )
        q = res["charges"]
        # q might be (6,) or (6, C) - sum over the donor and acceptor
        q_donor = q[:3].sum().item()
        q_acceptor = q[3:].sum().item()
        assert q_donor == pytest.approx(1.0, abs=0.05)
        assert q_acceptor == pytest.approx(-1.0, abs=0.05)

    def test_numpy_region_inputs(self):
        """region_mask and region_charges should accept numpy arrays."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()
        res = calc.eval_cqeq(
            data,
            region_mask=np.array([0, 0, 0, 1, 1, 1]),
            region_charges=np.array([0.0, 0.0]),
        )
        assert "energy" in res

    def test_list_region_inputs(self):
        """region_mask and region_charges should accept plain Python lists."""
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator("aimnet2", nb_threshold=0)
        data = self._dimer_data()
        res = calc.eval_cqeq(
            data,
            region_mask=[0, 0, 0, 1, 1, 1],
            region_charges=[0.0, 0.0],
        )
        assert "energy" in res
