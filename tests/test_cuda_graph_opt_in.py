"""The optional graph backend is selected explicitly and imported lazily."""

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import aimnet.calculators.calculator as calculator_module
from aimnet.calculators import AIMNet2Calculator


def test_constructor_defaults_to_regular_execution():
    assert inspect.signature(AIMNet2Calculator).parameters["cuda_graph"].default is False


@pytest.mark.parametrize(
    ("options", "environment", "selected"),
    [
        ({}, None, False),
        ({}, "1", False),
        ({"cuda_graph": False}, "1", False),
        ({"cuda_graph": True}, None, True),
        ({"cuda_graph": True}, "0", True),
        ({"cuda_graph": None}, None, False),
        ({"cuda_graph": None}, "0", False),
        ({"cuda_graph": None}, "false", False),
        ({"cuda_graph": None}, "no", False),
        ({"cuda_graph": None}, "off", False),
        ({"cuda_graph": None}, "", False),
        ({"cuda_graph": None}, "invalid", False),
        ({"cuda_graph": None}, "1", True),
        ({"cuda_graph": None}, "TRUE", True),
        ({"cuda_graph": None}, " yes ", True),
        ({"cuda_graph": None}, "on", True),
    ],
)
def test_public_constructor_graph_selection(monkeypatch, options, environment, selected):
    """Exercise dispatch on a CUDA-labelled calculator without allocating CUDA tensors."""
    if environment is None:
        monkeypatch.delenv("AIMNET_CUDA_GRAPH", raising=False)
    else:
        monkeypatch.setenv("AIMNET_CUDA_GRAPH", environment)
    model = torch.nn.Identity()
    monkeypatch.setattr(calculator_module, "resolve_model", lambda *args, **kwargs: (model, None, 5.0))
    calls, backend = [], object()

    class Cache:
        @classmethod
        def create(cls, calc, **kwargs):
            calls.append((calc, kwargs))
            return backend

    stub = ModuleType("aimnet.calculators.cuda_graph")
    stub.CUDAGraphCache = Cache
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    calc = AIMNet2Calculator(model, device="cuda", **options)
    assert calc._cuda_graph is (backend if selected else None)
    assert calls == ([(calc, {"cuda_graph": True})] if selected else [])


@pytest.mark.slow  # a fresh interpreter importing torch takes several seconds
def test_import_cpu_and_disabled_cuda_do_not_load_graph_backend():
    """Fresh interpreter catches eager imports even when other tests loaded the backend."""
    script = """
import importlib.abc
import os
import sys

class RejectGraphImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'aimnet.calculators.cuda_graph' or fullname.startswith('aimnet.calculators.cuda_graph.'):
            raise AssertionError(f'graph backend imported without CUDA opt-in: {fullname}')

sys.meta_path.insert(0, RejectGraphImport())
import torch
import aimnet.calculators.calculator as module
from aimnet.calculators import AIMNet2Calculator

model = torch.nn.Identity()
module.resolve_model = lambda *args, **kwargs: (model, None, 5.0)
os.environ['AIMNET_CUDA_GRAPH'] = '1'
for options in ({}, {'cuda_graph': False}, {'cuda_graph': True}, {'cuda_graph': None}):
    assert AIMNet2Calculator(model, device='cpu', **options)._cuda_graph is None
for options in ({}, {'cuda_graph': False}):
    assert AIMNet2Calculator(model, device='cuda', **options)._cuda_graph is None
os.environ['AIMNET_CUDA_GRAPH'] = '0'
assert AIMNet2Calculator(model, device='cuda', cuda_graph=None)._cuda_graph is None
assert not any(name.startswith('aimnet.calculators.cuda_graph') for name in sys.modules)
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(  # noqa: S603 -- fixed test source, current Python interpreter, no shell
        [sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
