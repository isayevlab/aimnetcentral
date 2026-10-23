"""CPU tests of allocation safety required before speculative CUDA replay."""

from types import SimpleNamespace

import pytest
import torch

from aimnet.calculators.cuda_graph.cache import CUDAGraphCache, _Entry


def test_detached_keepalive_does_not_pin_resized_allocation():
    value = torch.nn.Parameter(torch.arange(2048, dtype=torch.float32).reshape(64, 32), requires_grad=False)
    expected = value.clone()
    entry = _Entry(None, 0, (value.detach(),))
    assert entry.storage_valid()
    original_ptr = value.data_ptr()
    value.resize_(2048 * 1024)
    value.resize_(64, 32)
    value.copy_(expected)
    assert torch.equal(value, expected)
    assert entry.keepalive[0].data_ptr() == value.data_ptr()
    assert original_ptr != value.data_ptr()
    assert not entry.storage_valid(), "must reject before launching kernels with old captured pointers"


def test_replacement_and_set_keep_old_allocation_safe():
    value = torch.arange(64, dtype=torch.float32)
    original = value.data_ptr()
    entry = _Entry(None, 0, (value.detach(),))
    value.set_(torch.zeros(64, dtype=torch.float32))
    assert value.data_ptr() != original
    assert entry.keepalive[0].data_ptr() == original
    assert entry.storage_valid(), "old allocation remains valid until full postlaunch model rejection"


def test_inplace_numeric_update_does_not_need_prelaunch_rejection():
    value = torch.arange(64, dtype=torch.float32)
    entry = _Entry(None, 0, (value.detach(),))
    value.add_(1)
    assert entry.storage_valid(), "full contract checks values/version changes after a memory-safe replay"


def test_shared_storage_is_checked_once():
    value = torch.arange(64, dtype=torch.float32)
    entry = _Entry(None, 0, (value.detach(), value[:32].detach(), value[32:].detach()))
    assert len(entry.storages) == 1
    assert entry.storage_valid()


def test_raw_storage_shrink_rejected_even_without_tensor_version_change():
    value = torch.arange(64, dtype=torch.float32)
    entry = _Entry(None, 0, (value.detach(),))
    version = value._version
    value.untyped_storage().resize_(4)
    assert value._version == version
    assert not entry.storage_valid()


def test_same_address_but_insufficient_size_rejected():
    # Some allocators can shrink a storage without changing the allocation
    # address. Check both conditions, not only address equality.
    class Storage:
        def data_ptr(self):
            return 123456

        def nbytes(self):
            return 8

    entry = _Entry(None, 0)
    entry.storages = ((Storage(), 123456, 16),)
    assert not entry.storage_valid()


def test_guard_does_not_read_tensor_values(monkeypatch):
    value = torch.arange(64, dtype=torch.float32)
    entry = _Entry(None, 0, (value.detach(),))

    def forbidden(*args, **kwargs):
        pytest.fail("storage safety must use metadata only")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__float__", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    assert entry.storage_valid()


@pytest.mark.parametrize("attribute", ["external_coulomb", "external_dftd3"])
def test_unsupported_external_module_falls_back_before_stock_attribute_reads(attribute):
    class Custom(torch.nn.Module):
        # This can be valid on the regular path without exposing the stock
        # implementation's method/subtract_sr/smoothing attributes.
        def forward(self, data):
            return data

    model = object()
    calc = SimpleNamespace(model=model, external_coulomb=None, external_dftd3=None)
    setattr(calc, attribute, Custom())
    cache = CUDAGraphCache.__new__(CUDAGraphCache)
    cache._calc, cache._model = calc, model
    assert cache._settings() is None
