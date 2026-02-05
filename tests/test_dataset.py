import os

import numpy as np

from aimnet.data import SizeGroupedDataset, SizeGroupedSampler

datafile = os.path.join(os.path.dirname(__file__), "data", "dataset.h5")


def dataset(**kwargs):
    ds = SizeGroupedDataset(datafile, **kwargs)
    return ds


def test_from_h5():
    ds = dataset()
    assert isinstance(ds, SizeGroupedDataset)
    assert len(ds) == 195
    assert len(ds.groups) == 13
    assert len(ds.datakeys()) == 11


def test_from_h5_shards():
    ds = dataset(shard=(1, 3))
    assert isinstance(ds, SizeGroupedDataset)
    assert len(ds) == 65
    assert len(ds.groups) == 13
    assert len(ds.datakeys()) == 11


def test_from_h5_withkeys():
    ds = dataset(keys=["energy", "forces"])
    assert isinstance(ds, SizeGroupedDataset)
    assert len(ds) == 195
    assert len(ds.groups) == 13
    assert len(ds.datakeys()) == 2


def test_split():
    ds = dataset()
    ds1, ds2 = ds.random_split(0.8, 0.1)
    assert len(ds1) == 157
    assert len(ds2) == 19


def test_sae():
    ds = dataset()
    sae = ds.apply_peratom_shift("energy", "_energy")
    assert len(sae) == 14
    assert ds.concatenate("_energy").mean() < 0.39


def test_iter():
    ds = dataset()
    n = 0
    batchsize = 32
    for batch in ds.numpy_batches(batchsize):
        assert isinstance(batch, dict)
        for k, v in batch.items():
            assert isinstance(v, np.ndarray)
            assert len(v) <= batchsize
            assert isinstance(k, str)
        n += len(v)
    assert n == len(ds)


def test_merge_groups():
    ds = dataset()
    ds.merge_groups(32)
    assert len(ds.groups) == 4
    assert len(ds) == 195
    for g in ds.groups:
        assert len(g) >= 32


def test_loader():
    ds = dataset()
    ds.merge_groups(32)
    sampler = SizeGroupedSampler(ds, batch_size=12, batch_mode="molecules", shuffle=True, batches_per_epoch=10)
    assert len(sampler) == 10
    loader = ds.get_loader(sampler, x=["coord", "numbers"], y=["energy"], num_workers=0)
    assert len(loader) == 10
    assert sum(1 for _ in loader) == 10
    for batch in loader:
        assert len(batch) == 2
        assert "coord" in batch[0]
        assert "numbers" in batch[0]
        assert "energy" in batch[1]
        assert len(batch[0]["coord"]) <= 12
        assert len(batch[0]["numbers"]) <= 12
        assert len(batch[1]["energy"]) <= 12
    sampler = SizeGroupedSampler(ds, batch_size=12, batch_mode="molecules", shuffle=True, batches_per_epoch=-1)
    loader = ds.get_loader(sampler, x=["coord", "numbers"], y=["energy"], num_workers=0)
    n = 0
    for batch in loader:
        assert len(batch) == 2
        assert "coord" in batch[0]
        assert "numbers" in batch[0]
        assert "energy" in batch[1]
        assert len(batch[0]["coord"]) <= 12
        assert len(batch[0]["numbers"]) <= 12
        assert len(batch[1]["energy"]) <= 12
        n += len(batch[0]["coord"])
    assert n == len(ds)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_invalid_file_path_raises():
    """Test that invalid file path raises appropriate error."""
    import pytest

    with pytest.raises((FileNotFoundError, OSError)):
        SizeGroupedDataset("/nonexistent/path/to/file.h5")


def test_shard_boundary():
    """Test that shard boundary parameters work correctly."""
    # Valid shard should work and return subset of data
    ds = dataset(shard=(0, 3))
    assert len(ds) > 0
    assert len(ds) < 195  # Less than total

    # Last shard should also work
    ds_last = dataset(shard=(2, 3))
    assert len(ds_last) > 0


def test_valid_keys_subset():
    """Test that valid key subset works correctly."""
    ds = dataset(keys=["energy", "forces"])
    assert len(ds.datakeys()) == 2
    assert "energy" in ds.datakeys()
    assert "forces" in ds.datakeys()


def test_large_batch_size():
    """Test handling of large batch sizes."""
    ds = dataset()
    ds.merge_groups(32)

    # Very large batch should still work (returns all data)
    sampler = SizeGroupedSampler(ds, batch_size=1000, batch_mode="molecules", shuffle=False, batches_per_epoch=-1)
    loader = ds.get_loader(sampler, x=["coord", "numbers"], y=["energy"], num_workers=0)
    n = 0
    for batch in loader:
        n += len(batch[0]["coord"])
    assert n == len(ds)


def test_split_fractions():
    """Test split with various fractions."""
    ds = dataset()

    # Standard 80/20 split
    ds1, ds2 = ds.random_split(0.8, 0.2)
    assert len(ds1) + len(ds2) == len(ds)
    assert len(ds1) > len(ds2)
