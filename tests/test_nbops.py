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
        numbers = torch.tensor(
            [
                [6, 1, 1, 0],  # molecule with padding
                [6, 1, 0, 0],  # molecule with 2 padding atoms
            ],
            device=device,
        )
        # 3D nbmat: (B, N, max_neighbors)
        nbmat = torch.tensor(
            [
                [[1, 2, 3], [0, 2, 3], [0, 1, 3], [3, 3, 3]],  # batch 0
                [[1, 2, 3], [0, 2, 3], [2, 2, 2], [3, 3, 3]],  # batch 1
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

        In mode 2, mask_i_ only masks the last atom position in each batch,
        assuming padding atoms are always placed at the end.
        """
        B, N = 2, 3
        # Padding (number=0) should be at the last position
        numbers = torch.tensor([[6, 1, 0], [6, 1, 0]], device=device)
        nbmat = torch.randint(0, N, (B, N, 2), device=device)

        data = {"numbers": numbers, "nbmat": nbmat}
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)

        x = torch.ones((B, N), device=device)
        nbops.mask_i_(x, data, mask_value=0.0, inplace=True)

        # In mode 2, only the last position is masked (assumes padding at end)
        assert x[0, 2].item() == 0.0
        assert x[1, 2].item() == 0.0
        # First positions should remain unmasked
        assert x[0, 0].item() == 1.0
        assert x[0, 1].item() == 1.0


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
        numbers = torch.tensor([[6, 1, 1], [6, 1, 0]], device=device)
        nbmat = torch.tensor(
            [[[1, 2], [0, 2], [0, 1]], [[1, 2], [0, 2], [0, 1]]],  # (B, N, max_nb)
            device=device,
        )

        data = {"numbers": numbers, "nbmat": nbmat, "_nb_mode": torch.tensor(2)}

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
        numbers = torch.tensor([[6, 1, 1], [6, 1, 0]], device=device)
        nbmat = torch.tensor(
            [[[1, 2], [0, 2], [0, 1]], [[1, 2], [0, 2], [0, 1]]],  # (B, N, max_nb)
            device=device,
        )

        data = {"numbers": numbers, "nbmat": nbmat, "_nb_mode": torch.tensor(2)}

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
        assert (x.grad == 1.0).all()

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
