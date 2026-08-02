"""Tests for aimnet.nbops - neighbor operations module."""

import pytest
import torch

from aimnet import nbops


class TestSetNbMode:
    """Tests for set_nb_mode and get_nb_mode functions."""

    def test_nb_mode_0_no_nbmat(self, device):
        """Test nb_mode=0 when no neighbor matrix is provided."""
        data = {"numbers": torch.tensor([[6, 1, 1]], device=device)}
        data = nbops.set_nb_mode(data)
        assert nbops.get_nb_mode(data) == 0
        assert data["_nb_mode"].item() == 0

    def test_nb_mode_1_2d_nbmat(self, device):
        """Test nb_mode=1 when 2D neighbor matrix is provided."""
        N = 5
        nbmat = torch.randint(0, N, (N, 3), device=device)
        data = {"nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        assert nbops.get_nb_mode(data) == 1
        assert data["_nb_mode"].item() == 1

    def test_nb_mode_2_3d_nbmat(self, device):
        """Test nb_mode=2 when 3D neighbor matrix is provided."""
        B, N = 2, 5
        nbmat = torch.randint(0, N, (B, N, 3), device=device)
        data = {"nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        assert nbops.get_nb_mode(data) == 2
        assert data["_nb_mode"].item() == 2

    def test_invalid_nbmat_shape(self, device):
        """Test that invalid nbmat shape raises ValueError."""
        nbmat = torch.randint(0, 5, (2, 3, 4, 5), device=device)  # 4D tensor
        data = {"nbmat": nbmat}
        with pytest.raises(ValueError, match="Invalid neighbor matrix shape"):
            nbops.set_nb_mode(data)


class TestCalcMasks:
    """Tests for calc_masks function."""

    def test_calc_masks_mode_0_no_padding(self, simple_molecule):
        """Test mask calculation for mode 0 without padding."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Check mask shapes
        assert data["mask_i"].shape == data["numbers"].shape
        assert data["mask_ij"].shape == (1, 3, 3)  # (B, N, N)

        # No padding means mask_i should be all False
        assert not data["mask_i"].any()
        assert data["_input_padded"].item() is False

        # Diagonal should be True in mask_ij
        assert data["mask_ij"][0].diagonal().all()

    def test_calc_masks_mode_0_with_padding(self, padded_batch):
        """Test mask calculation for mode 0 with padding."""
        data = padded_batch.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # Second molecule has padding (atom with number=0)
        assert data["mask_i"][1, 2].item() is True  # padding atom
        assert data["_input_padded"].item() is True

        # mol_sizes should be correct
        assert data["mol_sizes"][0].item() == 3  # H2O
        assert data["mol_sizes"][1].item() == 2  # H2

    def test_calc_masks_mode_1(self, device):
        """Test mask calculation for mode 1 (flat format with mol_idx)."""
        # Create flat format data
        N = 5  # 4 real atoms + 1 padding
        coord = torch.rand((N, 3), device=device)
        numbers = torch.tensor([6, 1, 1, 1, 0], device=device)  # last is padding
        mol_idx = torch.tensor([0, 0, 1, 1, 2], device=device)  # 2 molecules + padding
        nbmat = torch.tensor(
            [
                [1, 4, 4],  # neighbors of atom 0
                [0, 4, 4],  # neighbors of atom 1
                [3, 4, 4],  # neighbors of atom 2
                [2, 4, 4],  # neighbors of atom 3
                [4, 4, 4],  # padding atom (neighbors itself)
            ],
            device=device,
        )

        data = {"coord": coord, "numbers": numbers, "mol_idx": mol_idx, "nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # mask_i should mark the last atom as padding
        assert data["mask_i"][-1].item() is True
        assert data["_input_padded"].item() is True

        # mask_ij should identify neighbor entries pointing to padding
        assert data["mask_ij"].shape == nbmat.shape
        assert data["mask_ij"][0, 1].item() is True  # points to padding atom 4
        assert data["mask_ij"][0, 0].item() is False  # points to real atom 1

    def test_calc_masks_mode_2(self, device):
        """Test mask calculation for mode 2 (batched with 3D nbmat)."""
        B, N = 2, 4
        coord = torch.rand((B, N, 3), device=device)
        numbers = torch.tensor([[6, 1, 1, 0], [6, 1, 0, 0]], device=device)
        # 3D nbmat: (B, N, max_neighbors)
        nbmat = torch.tensor(
            [
                [[1, 2, 8], [0, 2, 8], [0, 1, 8], [8, 8, 8]],
                [[5, 6, 8], [4, 6, 8], [8, 8, 8], [8, 8, 8]],
            ],
            device=device,
        )

        data = {"coord": coord, "numbers": numbers, "nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # mask_i should identify padding atoms (number=0)
        assert data["mask_i"][0, 3].item() is True
        assert data["mask_i"][1, 2].item() is True
        assert data["mask_i"][1, 3].item() is True
        assert data["_input_padded"].item() is True

        # mol_sizes should be correct
        assert data["mol_sizes"][0].item() == 3
        assert data["mol_sizes"][1].item() == 2

        # global sentinel and padding-target indices are masked per batch
        assert data["mask_ij"][0, 0, 2].item() is True
        assert data["mask_ij"][1, 0, 1].item() is True
        # padded center rows are fully masked
        assert data["mask_ij"][1, 2].all()

    def test_calc_masks_mode_2_masks_local_and_global_padding(self, device):
        numbers = torch.tensor([[6, 1, 1, 0], [6, 1, 0, 0]], device=device)
        nbmat_primary = torch.tensor(
            [
                [[1, 3, 8], [0, 3, 8], [0, 1, 8], [8, 8, 8]],
                [[5, 6, 8], [4, 6, 8], [8, 8, 8], [8, 8, 8]],
            ],
            device=device,
        )
        nbmat_lr = torch.tensor(
            [
                [[1, 2, 8], [0, 2, 8], [0, 1, 8], [8, 8, 8]],
                [[5, 7, 8], [4, 7, 8], [8, 8, 8], [8, 8, 8]],
            ],
            device=device,
        )

        data = {"numbers": numbers, "nbmat": nbmat_primary, "nbmat_lr": nbmat_lr}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        assert data["mask_ij"][0, 0, 1].item() is True
        assert data["mask_ij"][1, 0, 1].item() is True
        assert data["mask_ij_lr"][0, 0, 2].item() is True
        assert data["mask_ij_lr"][1, 0, 1].item() is True
        assert data["mask_ij"][0, 3].all()
        assert data["mask_ij_lr"][1, 2].all()


class TestMaskIj:
    """Tests for mask_ij_ function."""

    def test_mask_ij_inplace(self, device):
        """Test in-place masking of pairwise tensor."""
        data = {
            "numbers": torch.tensor([[6, 1, 1]], device=device),
            "_nb_mode": torch.tensor(0),
        }
        data = nbops.calc_masks(data)

        x = torch.ones((1, 3, 3), device=device)
        nbops.mask_ij_(x, data, mask_value=0.0, inplace=True)

        # Diagonal should be masked
        assert x[0].diagonal().sum().item() == 0.0
        # Off-diagonal should be unchanged
        assert x[0, 0, 1].item() == 1.0

    def test_mask_ij_not_inplace(self, device):
        """Test non-inplace masking returns new tensor."""
        data = {
            "numbers": torch.tensor([[6, 1, 1]], device=device),
            "_nb_mode": torch.tensor(0),
        }
        data = nbops.calc_masks(data)

        x_orig = torch.ones((1, 3, 3), device=device)
        x_new = nbops.mask_ij_(x_orig, data, mask_value=0.0, inplace=False)

        # Original should be unchanged
        assert x_orig[0].diagonal().sum().item() == 3.0
        # New tensor should have masked values
        assert x_new[0].diagonal().sum().item() == 0.0

    def test_mask_ij_with_features(self, device):
        """Test masking tensor with extra feature dimensions."""
        data = {
            "numbers": torch.tensor([[6, 1, 1]], device=device),
            "_nb_mode": torch.tensor(0),
        }
        data = nbops.calc_masks(data)

        # Tensor with extra feature dimension
        x = torch.ones((1, 3, 3, 5), device=device)
        nbops.mask_ij_(x, data, mask_value=-1.0, inplace=True)

        # Diagonal should be masked for all features
        for i in range(3):
            assert (x[0, i, i, :] == -1.0).all()


class TestMaskI:
    """Tests for mask_i_ function."""

    def test_mask_i_mode_0_padded(self, padded_batch):
        """Test atomic masking for mode 0 with padding."""
        data = padded_batch.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        x = torch.ones((2, 3), device=padded_batch["coord"].device)
        nbops.mask_i_(x, data, mask_value=0.0, inplace=True)

        # First molecule (H2O) has no padding
        assert x[0].sum().item() == 3.0
        # Second molecule (H2) has one padding atom
        assert x[1, 2].item() == 0.0
        assert x[1, :2].sum().item() == 2.0

    def test_mask_i_mode_1(self, device):
        """Test atomic masking for mode 1."""
        N = 4
        numbers = torch.tensor([6, 1, 1, 0], device=device)
        mol_idx = torch.tensor([0, 0, 0, 1], device=device)
        nbmat = torch.randint(0, N, (N, 2), device=device)

        data = {"numbers": numbers, "mol_idx": mol_idx, "nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        x = torch.ones(N, device=device)
        nbops.mask_i_(x, data, mask_value=0.0, inplace=True)

        # Last atom (padding) should be masked
        assert x[-1].item() == 0.0
        assert x[:-1].sum().item() == 3.0

    def test_mask_i_mode_2(self, device):
        """Test atomic masking for mode 2.

        In mode 2, mask_i_ masks every atom where numbers == 0.
        """
        B, N = 2, 3
        numbers = torch.tensor([[6, 0, 1], [6, 1, 0]], device=device)
        nbmat = torch.randint(0, N, (B, N, 2), device=device)

        data = {"numbers": numbers, "nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        x = torch.ones((B, N), device=device)
        nbops.mask_i_(x, data, mask_value=0.0, inplace=True)

        assert x[0, 1].item() == 0.0
        assert x[1, 2].item() == 0.0
        # Non-padding positions should remain unmasked
        assert x[0, 0].item() == 1.0
        assert x[0, 2].item() == 1.0


class TestGetIj:
    """Tests for get_ij function."""

    def test_get_ij_mode_0(self, simple_molecule):
        """Test pairwise expansion for mode 0."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)

        x = torch.tensor([[[1.0], [2.0], [3.0]]], device=simple_molecule["coord"].device)
        x_i, x_j = nbops.get_ij(x, data)

        # x_i should be (B, N, 1, features) - expanded along dim 2
        assert x_i.shape == (1, 3, 1, 1)
        # x_j should be (B, 1, N, features) - expanded along dim 1
        assert x_j.shape == (1, 1, 3, 1)

        # Check values
        assert x_i[0, 0, 0, 0].item() == 1.0
        assert x_j[0, 0, 0, 0].item() == 1.0
        assert x_j[0, 0, 2, 0].item() == 3.0

    def test_get_ij_mode_1(self, device):
        """Test pairwise extraction for mode 1."""
        N = 4
        numbers = torch.tensor([6, 1, 1, 0], device=device)
        mol_idx = torch.tensor([0, 0, 0, 1], device=device)
        nbmat = torch.tensor([[1, 2], [0, 2], [0, 1], [3, 3]], device=device)

        data = {"numbers": numbers, "mol_idx": mol_idx, "nbmat": nbmat, "_nb_mode": torch.tensor(1)}

        x = torch.tensor([[1.0], [2.0], [3.0], [0.0]], device=device)
        x_i, x_j = nbops.get_ij(x, data)

        # x_i: (N, 1, features)
        assert x_i.shape == (N, 1, 1)
        # x_j: (N, max_nb, features)
        assert x_j.shape == (N, 2, 1)

        # Check that x_j indexes correctly
        assert x_j[0, 0, 0].item() == 2.0  # neighbor 1 of atom 0
        assert x_j[0, 1, 0].item() == 3.0  # neighbor 2 of atom 0

    def test_get_ij_mode_2(self, device):
        """Test pairwise extraction for mode 2."""
        B, N = 2, 3
        numbers = torch.tensor([[6, 1, 0], [6, 1, 0]], device=device)
        nbmat = torch.tensor(
            [[[1, 2], [0, 2], [6, 6]], [[4, 5], [3, 5], [6, 6]]],
            device=device,
        )

        data = nbops.calc_masks({"numbers": numbers, "nbmat": nbmat, "_nb_mode": torch.tensor(2)})

        x = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], device=device)
        x_i, x_j = nbops.get_ij(x, data)

        # x_i: (B, N, 1, features)
        assert x_i.shape == (B, N, 1, 1)
        # x_j: (B, N, max_nb, features)
        assert x_j.shape == (B, N, 2, 1)


class TestGetI:
    """Tests for get_i function."""

    def test_get_i_mode_0(self, simple_molecule):
        """Test get_i for mode 0 matches get_ij x_i component."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)

        x = torch.tensor([[[1.0], [2.0], [3.0]]], device=simple_molecule["coord"].device)
        x_i_only = nbops.get_i(x, data)
        x_i, _x_j = nbops.get_ij(x, data)

        # Should match get_ij x_i component
        assert x_i_only.shape == x_i.shape
        assert torch.allclose(x_i_only, x_i)

        # x_i should be (B, N, 1, features) - expanded along dim 2
        assert x_i_only.shape == (1, 3, 1, 1)

    def test_get_i_mode_1(self, device):
        """Test get_i for mode 1 matches get_ij x_i component."""
        N = 4
        numbers = torch.tensor([6, 1, 1, 0], device=device)
        mol_idx = torch.tensor([0, 0, 0, 1], device=device)
        nbmat = torch.tensor([[1, 2], [0, 2], [0, 1], [3, 3]], device=device)

        data = {"numbers": numbers, "mol_idx": mol_idx, "nbmat": nbmat, "_nb_mode": torch.tensor(1)}

        x = torch.tensor([[1.0], [2.0], [3.0], [0.0]], device=device)
        x_i_only = nbops.get_i(x, data)
        x_i, _x_j = nbops.get_ij(x, data)

        # Should match get_ij x_i component
        assert x_i_only.shape == x_i.shape
        assert torch.allclose(x_i_only, x_i)

        # x_i: (N, 1, features)
        assert x_i_only.shape == (N, 1, 1)

    def test_get_i_mode_2(self, device):
        """Test get_i for mode 2 matches get_ij x_i component."""
        B, N = 2, 3
        numbers = torch.tensor([[6, 1, 0], [6, 1, 0]], device=device)
        nbmat = torch.tensor(
            [[[1, 2], [0, 2], [6, 6]], [[4, 5], [3, 5], [6, 6]]],
            device=device,
        )

        data = nbops.calc_masks({"numbers": numbers, "nbmat": nbmat, "_nb_mode": torch.tensor(2)})

        x = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], device=device)
        x_i_only = nbops.get_i(x, data)
        x_i, _x_j = nbops.get_ij(x, data)

        # Should match get_ij x_i component
        assert x_i_only.shape == x_i.shape
        assert torch.allclose(x_i_only, x_i)

        # x_i: (B, N, 1, features)
        assert x_i_only.shape == (B, N, 1, 1)


class TestMolSum:
    """Tests for mol_sum function."""

    def test_mol_sum_mode_0(self, simple_molecule):
        """Test molecular summation for mode 0."""
        data = simple_molecule.copy()
        data = nbops.set_nb_mode(data)

        x = torch.tensor([[1.0, 2.0, 3.0]], device=simple_molecule["coord"].device)
        result = nbops.mol_sum(x, data)

        # Should sum over atoms (dim 1)
        assert result.shape == (1,)
        assert result.item() == 6.0

    def test_mol_sum_mode_0_batch(self, padded_batch):
        """Test molecular summation for batched mode 0."""
        data = padded_batch.copy()
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        x = torch.ones((2, 3), device=padded_batch["coord"].device)
        result = nbops.mol_sum(x, data)

        # Each batch sums over atoms
        assert result.shape == (2,)
        assert result[0].item() == 3.0
        assert result[1].item() == 3.0  # includes padding

    def test_mol_sum_mode_1(self, device):
        """Test molecular summation for mode 1."""
        numbers = torch.tensor([6, 1, 1, 6, 1, 0], device=device)
        mol_idx = torch.tensor([0, 0, 0, 1, 1, 2], device=device)

        data = {"numbers": numbers, "mol_idx": mol_idx, "_nb_mode": torch.tensor(1)}

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0], device=device)
        result = nbops.mol_sum(x, data)

        # Should produce per-molecule sums
        assert result.shape == (3,)  # 2 real molecules + 1 padding
        assert result[0].item() == 6.0  # mol 0: 1+2+3
        assert result[1].item() == 9.0  # mol 1: 4+5
        assert result[2].item() == 0.0  # padding

    def test_mol_sum_mode_1_with_features(self, device):
        """Test molecular summation for mode 1 with feature dimension."""
        N = 4
        numbers = torch.tensor([6, 1, 1, 0], device=device)
        mol_idx = torch.tensor([0, 0, 0, 1], device=device)

        data = {"numbers": numbers, "mol_idx": mol_idx, "_nb_mode": torch.tensor(1)}

        x = torch.ones((N, 5), device=device)  # (N, features)
        result = nbops.mol_sum(x, data)

        assert result.shape == (2, 5)  # (num_mols, features)
        assert result[0, 0].item() == 3.0  # sum of 3 atoms

    def test_mol_sum_mode_1_matches_mode_0(self, device):
        """Test that packed (mode 1) mol_sum matches dense (mode 0) result."""
        torch.manual_seed(0)
        sizes = [3, 2, 4]
        B, N = len(sizes), max(sizes)
        n_total = sum(sizes) + 1  # trailing padding atom

        # dense batch (B, N) with zero-padded numbers and values
        numbers_dense = torch.zeros((B, N), dtype=torch.long, device=device)
        x_dense = torch.zeros((B, N), device=device)
        for b, s in enumerate(sizes):
            numbers_dense[b, :s] = torch.randint(1, 10, (s,), device=device)
            x_dense[b, :s] = torch.randn(s, device=device)
        data_dense = {"numbers": numbers_dense}
        data_dense = nbops.set_nb_mode(data_dense)
        data_dense = nbops.calc_masks(data_dense)

        # same batch in packed layout; padding atom carries the last mol index
        numbers_packed = torch.cat([numbers_dense[b, :s] for b, s in enumerate(sizes)])
        numbers_packed = torch.cat([numbers_packed, torch.zeros(1, dtype=torch.long, device=device)])
        x_packed = torch.cat([x_dense[b, :s] for b, s in enumerate(sizes)])
        x_packed = torch.cat([x_packed, torch.zeros(1, device=device)])
        mol_idx = torch.repeat_interleave(torch.arange(B, device=device), torch.tensor(sizes, device=device))
        mol_idx = torch.cat([mol_idx, mol_idx[-1:]])
        nbmat = torch.full((n_total, 2), n_total - 1, dtype=torch.long, device=device)
        data_packed = {"numbers": numbers_packed, "mol_idx": mol_idx, "nbmat": nbmat}
        data_packed = nbops.set_nb_mode(data_packed)
        data_packed = nbops.calc_masks(data_packed)

        res_dense = nbops.mol_sum(x_dense, data_dense)
        res_packed = nbops.mol_sum(x_packed, data_packed)

        assert res_packed.shape == res_dense.shape == (B,)
        assert torch.allclose(res_packed, res_dense, atol=1e-6)

    def test_mol_sum_mode_1_uses_cached_num_mol(self, device):
        """Test that calc_masks caches _num_mol on CPU and mol_sum uses it."""
        numbers = torch.tensor([6, 1, 1, 6, 1, 0], device=device)
        mol_idx = torch.tensor([0, 0, 0, 1, 1, 1], device=device)  # padding atom in last mol
        nbmat = torch.full((6, 2), 5, dtype=torch.long, device=device)

        data = {"numbers": numbers, "mol_idx": mol_idx, "nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        # cached as a CPU tensor (same pattern as _nb_mode), so reading it
        # in mol_sum does not sync the GPU
        assert data["_num_mol"].device.type == "cpu"
        assert data["_num_mol"].item() == 2

        x = torch.ones(6, device=device)
        result = nbops.mol_sum(x, data)
        assert result.shape == (2,)

        # dicts not built through calc_masks fall back to mol_idx and cache
        bare = {"mol_idx": mol_idx, "_nb_mode": torch.tensor(1)}
        result_bare = nbops.mol_sum(x, bare)
        assert result_bare.shape == (2,)
        assert bare["_num_mol"].item() == 2
        assert torch.allclose(result_bare, result)

    def test_mol_sum_mode_2(self, device):
        """Test molecular summation for mode 2."""
        B, N = 2, 3
        numbers = torch.tensor([[6, 1, 1], [6, 1, 0]], device=device)
        nbmat = torch.randint(0, N, (B, N, 2), device=device)

        data = {"numbers": numbers, "nbmat": nbmat, "_nb_mode": torch.tensor(2)}

        x = torch.ones((B, N), device=device)
        result = nbops.mol_sum(x, data)

        # Should sum over dim 1
        assert result.shape == (B,)
        assert result[0].item() == 3.0
        assert result[1].item() == 3.0


class TestGradientFlow:
    """Tests to verify gradients flow correctly through nbops functions."""

    def test_mol_sum_gradient(self, device):
        """Test that gradients flow through mol_sum."""
        x = torch.tensor([[1.0, 2.0, 3.0]], device=device, requires_grad=True)
        data = {
            "numbers": torch.tensor([[6, 1, 1]], device=device),
            "_nb_mode": torch.tensor(0),
        }

        result = nbops.mol_sum(x, data)
        result.sum().backward()

        # Gradient should be 1 for all inputs
        assert x.grad is not None
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_mask_ij_gradient(self, device):
        """Test that gradients flow through mask_ij_ (not inplace)."""
        data = {
            "numbers": torch.tensor([[6, 1, 1]], device=device),
            "_nb_mode": torch.tensor(0),
        }
        data = nbops.calc_masks(data)

        x = torch.ones((1, 3, 3), device=device, requires_grad=True)
        x_masked = nbops.mask_ij_(x, data, mask_value=0.0, inplace=False)

        loss = x_masked.sum()
        loss.backward()

        # Gradient should be 1 for non-masked, 0 for masked (diagonal)
        assert x.grad is not None
        assert x.grad[0].diagonal().sum().item() == 0.0
        # Off-diagonal: 6 elements with grad=1
        assert x.grad[0].sum().item() == 6.0

    def test_get_ij_gradient_mode_0(self, device):
        """Test gradients through get_ij for mode 0."""
        x = torch.tensor([[[1.0], [2.0], [3.0]]], device=device, requires_grad=True)
        data = {"_nb_mode": torch.tensor(0)}

        x_i, x_j = nbops.get_ij(x, data)
        loss = (x_i * x_j).sum()
        loss.backward()

        assert x.grad is not None


def _global_mode2_data(
    device: torch.device,
    *,
    pad_neighbor: bool = False,
    suffixes: tuple[str, ...] = (),
    include_shifts: bool = False,
) -> dict[str, torch.Tensor]:
    """Build a small global-index mode-2 case with one trailing dummy per system."""
    B, N, M = 2, 4, 3
    sentinel = B * N
    coord = torch.arange(B * N * 3, device=device, dtype=torch.float32).reshape(B, N, 3)
    numbers = torch.tensor([[6, 1, 1, 0], [8, 1, 1, 0]], device=device)
    nbmat = torch.full((B, N, M), sentinel, device=device, dtype=torch.int64)
    for b in range(B):
        base = b * N
        nbmat[b, 0, :2] = torch.tensor([base + 1, base + 2], device=device)
        nbmat[b, 1, :2] = torch.tensor([base, base + 2], device=device)
        nbmat[b, 2, :2] = torch.tensor([base, base + 1], device=device)
    if pad_neighbor:
        nbmat[0, 0, 1] = N - 1
    data: dict[str, torch.Tensor] = {"coord": coord, "numbers": numbers, "nbmat": nbmat}
    if include_shifts:
        shifts = torch.zeros((*nbmat.shape, 3), device=device, dtype=torch.float32)
        data["shifts"] = shifts
    for suffix in suffixes:
        data[f"nbmat{suffix}"] = nbmat.clone()
        if include_shifts:
            data[f"shifts{suffix}"] = shifts.clone()
    return data


def test_global_mode2_gathers_distinct_batch_values(device):
    data = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device)))
    values = torch.tensor([[[10.0], [11.0], [12.0], [0.0]], [[20.0], [21.0], [22.0], [0.0]]], device=device)
    _x_i, x_j = nbops.get_ij(values, data)
    assert x_j[0, 0, 0, 0] == 11.0
    assert x_j[1, 0, 0, 0] == 21.0


def test_global_mode2_excludes_sentinel(device):
    data = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device)))
    values = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4, 1)
    _x_i, x_j = nbops.get_ij(values, data)
    assert x_j[0, 0, 2, 0] == 0.0
    assert x_j[1, 0, 2, 0] == 0.0


def test_global_mode2_masks_padded_neighbor(device):
    data = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device, pad_neighbor=True)))
    assert data["mask_ij"][0, 0, 1]
    assert data["_nbmat_gather"][0, 0, 1] == 0


def test_global_mode2_kernel_indices_exclude_padded_neighbor(device):
    data = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device, pad_neighbor=True)))
    assert data["_nbmat_kernel"][0, 0, 1] == data["nbmat"].shape[0] * data["nbmat"].shape[1]


def test_global_mode2_masks_padded_center(device):
    data = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device)))
    assert data["mask_ij"][0, -1].all()
    assert data["_nbmat_gather"][0, -1].eq(0).all()


def test_global_mode2_rejects_non_sentinel_padded_center_cpu():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"][0, -1, 0] = 0
    with pytest.raises(ValueError, match="padded center"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_local_batch1_index_cpu():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"][1, 0, 0] = 0
    with pytest.raises(ValueError, match="batch interval"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_interleaved_sentinel_cpu():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"][0, 0] = torch.tensor([1, 8, 2])
    with pytest.raises(ValueError, match=r"packed.*tail"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_interleaved_padded_neighbor_cpu():
    data = _global_mode2_data(torch.device("cpu"), pad_neighbor=True)
    data["nbmat"][0, 0, 2] = 2
    with pytest.raises(ValueError, match=r"packed.*tail"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_builds_independent_suffix_tensors(device):
    data = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device, suffixes=("_lr",))))
    assert data["mask_ij"] is not data["mask_ij_lr"]
    assert data["_nbmat_gather"] is not data["_nbmat_gather_lr"]


def test_global_mode2_reuses_exact_alias_int32(device):
    data = _global_mode2_data(device, suffixes=("_lr",))
    data["nbmat"] = data["nbmat"].to(torch.int32)
    data["nbmat_lr"] = data["nbmat"]
    data = nbops.calc_masks(nbops.set_nb_mode(data))
    assert data["_nbmat_gather"] is data["_nbmat_gather_lr"]


def test_global_mode2_reuses_exact_alias_int64(device):
    data = _global_mode2_data(device, suffixes=("_lr",))
    data["nbmat_lr"] = data["nbmat"]
    data = nbops.calc_masks(nbops.set_nb_mode(data))
    data["mask_i"] = data["numbers"] == 0
    nbops._prepare_mode2_neighbor_tensors(data)
    assert data["nbmat"] is data["nbmat_lr"]


def test_global_mode2_compile_alias_reuse_int32(device):
    data = _global_mode2_data(device, suffixes=("_lr",))
    data["nbmat"] = data["nbmat"].to(torch.int32)
    data["nbmat_lr"] = data["nbmat"]
    data = nbops.set_nb_mode(data)
    data["mask_i"] = data["numbers"] == 0
    nbops._prepare_mode2_neighbor_tensors(data)
    assert data["_nbmat_gather"] is data["_nbmat_gather_lr"]


def test_global_mode2_compile_alias_reuse_int64(device):
    data = _global_mode2_data(device, suffixes=("_lr",))
    data["nbmat_lr"] = data["nbmat"]
    data = nbops.set_nb_mode(data)
    data["mask_i"] = data["numbers"] == 0
    nbops._prepare_mode2_neighbor_tensors(data)
    assert data["_nbmat_kernel"] is data["_nbmat_kernel_lr"]


def test_global_mode2_rejects_bool_neighbors():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"] = data["nbmat"].to(torch.bool)
    with pytest.raises(ValueError, match="integer dtype"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_unsigned_neighbors():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"] = data["nbmat"].to(torch.uint8)
    with pytest.raises(ValueError, match="signed"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_float_neighbors():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"] = data["nbmat"].to(torch.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_complex_neighbors():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"] = data["nbmat"].to(torch.complex64)
    with pytest.raises(ValueError, match="integer dtype"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_int32_capacity_overflow():
    n = torch.iinfo(torch.int32).max + 1
    data = {
        "coord": torch.empty((1, n, 3), device="meta"),
        "numbers": torch.empty((1, n), dtype=torch.int64, device="meta"),
        "nbmat": torch.empty((1, n, 1), dtype=torch.int64, device="meta"),
    }
    with pytest.raises(ValueError, match="int32"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_mismatched_shift_shape():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["shifts"] = torch.zeros(2, 4, 2, 3)
    with pytest.raises(ValueError, match="shifts"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_orphan_shift():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    del data["nbmat"]
    with pytest.raises(ValueError, match="matching"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_missing_final_dummy():
    data = _global_mode2_data(torch.device("cpu"))
    data["numbers"][:, -1] = 6
    with pytest.raises(ValueError, match="final dummy"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_noncontiguous_atom_padding():
    data = _global_mode2_data(torch.device("cpu"))
    data["numbers"][0, 1] = 0
    with pytest.raises(ValueError, match="contiguous"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_noncontiguous_coord():
    data = _global_mode2_data(torch.device("cpu"))
    data["coord"] = torch.zeros((4, 2, 3)).transpose(0, 1)
    with pytest.raises(ValueError, match="flatten"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_noncontiguous_numbers():
    data = _global_mode2_data(torch.device("cpu"))
    data["numbers"] = torch.tensor([[6, 8], [1, 1], [1, 1], [0, 0]]).transpose(0, 1)
    with pytest.raises(ValueError, match="flatten"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_noncontiguous_nbmat():
    data = _global_mode2_data(torch.device("cpu"))
    data["nbmat"] = torch.zeros((4, 2, 3), dtype=torch.int64).transpose(0, 1)
    with pytest.raises(ValueError, match="flatten"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_noncontiguous_shifts():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    data["shifts"] = torch.zeros((4, 2, 3, 3)).transpose(0, 1)
    with pytest.raises(ValueError, match="flatten"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_normalizes_single_periodic_geometry():
    data = {"cell": torch.eye(3), "pbc": torch.ones(3, dtype=torch.bool)}
    nbops.normalize_mode2_periodic_geometry(data, B=1)
    assert data["cell"].shape == (1, 3, 3)
    assert data["pbc"].shape == (1, 3)


def test_global_mode2_preserves_batched_periodic_geometry():
    data = _global_mode2_data(torch.device("cpu"))
    cell = torch.stack([torch.eye(3), torch.eye(3) * 2])
    pbc = torch.ones((2, 3), dtype=torch.bool)
    data["cell"] = cell
    data["pbc"] = pbc
    nbops.normalize_mode2_periodic_geometry(data, B=2)
    assert data["cell"] is cell
    assert data["pbc"] is pbc


def test_global_mode2_rejects_pbc_without_cell():
    data = _global_mode2_data(torch.device("cpu"))
    data["pbc"] = torch.ones(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="cell"):
        nbops.normalize_mode2_periodic_geometry(data, B=2)


def test_global_mode2_rejects_shifts_without_cell():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    with pytest.raises(ValueError, match="cell"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_partial_pbc():
    data = _global_mode2_data(torch.device("cpu"))
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="full-3D"):
        nbops.normalize_mode2_periodic_geometry(data, B=2)


def test_global_mode2_rejects_missing_periodic_shifts():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    del data["shifts"]
    with pytest.raises(ValueError, match="shifts"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_fractional_periodic_shifts():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    data["shifts"][0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="integral"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_shift_int32_overflow():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    data["shifts"][0, 0, 0, 0] = float(torch.iinfo(torch.int32).max) + 1
    with pytest.raises(ValueError, match="int32"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_nonzero_sentinel_shift():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    data["shifts"][0, 0, 2, 0] = 1
    with pytest.raises(ValueError, match="sentinel"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_nonzero_padded_neighbor_shift():
    data = _global_mode2_data(torch.device("cpu"), pad_neighbor=True, include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    data["shifts"][0, 0, 1, 0] = 1
    with pytest.raises(ValueError, match="padded-neighbor"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_rejects_nonzero_padded_center_shift():
    data = _global_mode2_data(torch.device("cpu"), include_shifts=True)
    data["cell"] = torch.eye(3).repeat(2, 1, 1)
    data["pbc"] = torch.ones((2, 3), dtype=torch.bool)
    data["shifts"][0, -1, 0, 0] = 1
    with pytest.raises(ValueError, match="padded center"):
        nbops.validate_mode2_nbmat_raw(data, suffix="")


def test_global_mode2_preserves_mode0_and_mode1(device):
    mode0 = nbops.set_nb_mode({"numbers": torch.ones((1, 2), device=device)})
    mode1 = nbops.set_nb_mode({"numbers": torch.ones(3, device=device), "nbmat": torch.zeros((3, 1), device=device)})
    assert nbops.get_nb_mode(mode0) == 0
    assert nbops.get_nb_mode(mode1) == 1


def test_global_mode2_convert_local_success_is_immutable():
    local = torch.tensor([[[1, 2, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]]] * 2, dtype=torch.int64)
    padding = torch.tensor([[[False, False, True], [False, False, True], [True, True, True], [True, True, True]]] * 2)
    local_before = local.clone()
    padding_before = padding.clone()
    global_nbmat = nbops.convert_mode2_local_to_global(local, padding_mask=padding)
    assert torch.equal(global_nbmat[0, 0], torch.tensor([1, 2, 8]))
    assert torch.equal(global_nbmat[1, 0], torch.tensor([5, 6, 8]))
    assert torch.equal(local, local_before)
    assert torch.equal(padding, padding_before)


def test_global_mode2_convert_local_rejects_non_tail_mask():
    local = torch.zeros((1, 2, 3), dtype=torch.int64)
    padding = torch.tensor([[[False, True, False], [False, False, True]]])
    with pytest.raises(ValueError, match="tail"):
        nbops.convert_mode2_local_to_global(local, padding_mask=padding)


def test_global_mode2_convert_local_rejects_out_of_range():
    local = torch.tensor([[[1, 2, 4], [0, 0, 0]]], dtype=torch.int64)
    padding = torch.tensor([[[False, False, False], [True, True, True]]])
    with pytest.raises(ValueError, match="range"):
        nbops.convert_mode2_local_to_global(local, padding_mask=padding)


def test_global_mode2_convert_local_requires_final_dummy_center():
    local = torch.zeros((1, 3, 1), dtype=torch.int32)
    padding = torch.zeros((1, 3, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="final dummy"):
        nbops.convert_mode2_local_to_global(local, padding_mask=padding)


def test_global_mode2_convert_local_upcasts_before_global_arithmetic():
    local = torch.zeros((2, 20_000, 1), dtype=torch.int16)
    padding = torch.ones_like(local, dtype=torch.bool)
    padding[:, :-1] = False
    local[:, :-1] = 0
    result = nbops.convert_mode2_local_to_global(local, padding_mask=padding)
    assert result.dtype == torch.int32
    assert result[1, 0, 0] == 20_000


def test_global_mode2_private_preparation_matches_calc_masks(device):
    expected = nbops.calc_masks(nbops.set_nb_mode(_global_mode2_data(device)))
    actual = nbops.set_nb_mode(_global_mode2_data(device))
    actual["mask_i"] = actual["numbers"] == 0
    nbops._prepare_mode2_neighbor_tensors(actual)
    for key in ("mask_ij", "_nbmat_gather", "_nbmat_kernel"):
        assert torch.equal(actual[key], expected[key])


def test_global_mode2_rejects_suffix_device_mismatch():
    data = _global_mode2_data(torch.device("cpu"), suffixes=("_lr",))
    data["nbmat_lr"] = data["nbmat_lr"].to("meta")
    with pytest.raises(ValueError, match="same device"):
        nbops.validate_mode2_nbmat_raw(data, suffix="_lr")


def test_global_mode2_rejects_mixed_primary_and_suffix_rank():
    data = _global_mode2_data(torch.device("cpu"), suffixes=("_lr",))
    data["nbmat"] = data["nbmat"][0]
    with pytest.raises(ValueError, match="rank"):
        nbops.validate_neighbor_suffix_layout(data)


def test_global_mode2_rejects_suffix_only_3d_matrix():
    data = _global_mode2_data(torch.device("cpu"))
    suffix_only = {"coord": data["coord"], "numbers": data["numbers"], "nbmat_lr": data["nbmat"]}
    with pytest.raises(ValueError, match="primary nbmat"):
        nbops.validate_neighbor_suffix_layout(suffix_only)
