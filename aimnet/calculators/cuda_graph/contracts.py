"""CPU-only contracts for the specialized CUDA-graph inference engines.

Graph replay bypasses Python module dispatch and embeds some scalar constants. A
successful comparison at one geometry cannot validate an arbitrary ``forward``
or hook. Accept only the stock module implementation, and fingerprint its
parameters, buffers, topology and scalar options on *every* call.

No device tensor is read here: tensor identity and PyTorch's mutation version
detect ordinary in-place updates, ``load_state_dict`` and parameter replacement.
As with PyTorch autograd, mutations through ``.data`` / raw pointers that evade
version counters are unsupported; clear the graph cache after such changes.
Concurrent model mutation while a calculator call is running is unsupported.
"""

from __future__ import annotations

import inspect
from operator import attrgetter, itemgetter

import torch
from torch import nn
from torch.nn.modules import module as module_hooks

from aimnet.models.aimnet2 import AIMNet2
from aimnet.models.base import AIMNet2Base
from aimnet.modules.aev import AEVSV, ConvSV
from aimnet.modules.core import AtomicShift, AtomicSum, Embedding, Output
from aimnet.modules.lr import DFTD3, LRCoulomb, SRCoulomb

_TYPES = frozenset((
    AIMNet2,
    AEVSV,
    ConvSV,
    Embedding,
    Output,
    AtomicShift,
    AtomicSum,
    SRCoulomb,
    LRCoulomb,
    DFTD3,
    nn.ModuleDict,
    nn.ModuleList,
    nn.Sequential,
    nn.Linear,
    nn.Embedding,
    nn.GELU,
))
_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GLOBAL_HOOKS = tuple("_global" + name for name in _HOOKS)
_BOOKKEEPING = frozenset((
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
    "_is_full_backward_hook",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_metadata",
    *_HOOKS,
))
_DISPATCH = ("__call__", "_call_impl", "_wrapped_call_impl", "children", "named_children")
# Capture trusted implementations once, not from a possibly patched instance.
_METHODS = {}
for _cls in (*_TYPES, AIMNet2Base):
    _names = {"forward", *_DISPATCH}
    for _base in _cls.__mro__:
        if _base in (nn.Module, object):
            continue
        _names.update(name for name, value in vars(_base).items() if inspect.isfunction(value))
    _METHODS[_cls] = tuple((name, getattr(_cls, name)) for name in _names if hasattr(_cls, name))


def _value(value):
    """Hashable non-device state; never call float/item on a tensor."""
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, (torch.dtype, torch.device)):
        return value
    if isinstance(value, torch.Tensor):
        return (id(value), value._version, value.dtype, value.device)
    if type(value) in (tuple, list):
        return tuple(_value(item) for item in value)
    raise TypeError("unsupported mutable or executable module option")


class ModelContract:
    """Contract and mutation token for a stock closed-shell AIMNet2 model.

    ``snapshot()`` returns a hashable token, or ``None`` when a specialized
    engine cannot preserve the regular-path semantics. The owner must drop
    cached engines whenever the token changes, including a transition to None.
    Instantiate a new contract when a root model / external module is replaced.

    The hot path uses a cached flat module list rather than recursively calling
    ``modules()`` or ``state_dict()``. A topology or instance-attribute change
    rebuilds this small CPU plan before accepting the new state.
    """

    def __init__(self, model, external_coulomb=None, external_dftd3=None):
        self.model = model
        self.coulomb = external_coulomb
        self.dftd3 = external_dftd3
        self._nodes = ()
        self._classes = ()
        self._last = None
        self._generation = 0
        self._rebuild()

    def _supported(self):
        m, c, d = self.model, self.coulomb, self.dftd3
        if type(m) is not AIMNet2 or m.training:
            return False
        if type(m.num_charge_channels) is not int or m.num_charge_channels != 1 or m.d2features is not True:
            return False
        if type(m.aev) is not AEVSV or type(m.conv_a) is not ConvSV or type(m.conv_q) is not ConvSV:
            return False
        if m.aev._dual_basis is not False:
            return False
        if m.conv_a.d2features is not True or m.conv_q.d2features is not False or type(m.outputs) is not nn.ModuleDict:
            return False
        expected = {"energy_mlp": Output, "atomic_shift": AtomicShift, "atomic_sum": AtomicSum, "srcoulomb": SRCoulomb}
        outputs = m.outputs._modules
        if not {"energy_mlp", "atomic_sum"} <= outputs.keys() <= expected.keys():
            return False
        if any(type(mod) is not expected[name] for name, mod in outputs.items()):
            return False
        order = tuple(outputs)
        canonical = tuple(name for name in expected if name in outputs)
        if order != canonical:
            return False
        for name, mod in outputs.items():
            expected_input = "aim" if name == "energy_mlp" else "charges" if name == "srcoulomb" else "energy"
            if type(mod.key_in) is not str or type(mod.key_out) is not str:
                return False
            if mod.key_in != expected_input or mod.key_out != "energy":
                return False
        if "atomic_shift" in outputs and outputs["atomic_shift"].reduce_sum is not False:
            return False
        # Sparse engines use the canonical electrostatic conversion factor.
        from aimnet import constants

        factor = constants.half_Hartree * constants.Bohr
        sr = outputs.get("srcoulomb")
        if sr is not None and (
            type(sr.envelope) is not str
            or sr.envelope != "exp"
            or type(sr._factor) is not float
            or sr._factor != factor
        ):
            return False
        if c is not None and (
            type(c) is not LRCoulomb
            or type(c.method) is not str
            or c.method != "simple"
            or c.subtract_sr is not False
            or type(c._factor) is not float
            or c._factor != factor
            or type(c.key_in) is not str
            or c.key_in != "charges"
            or type(c.key_out) is not str
            or c.key_out != "energy"
        ):
            return False
        return d is None or (type(d) is DFTD3 and type(d.key_out) is str and d.key_out == "energy")

    def _rebuild(self):
        self._generation += 1
        self._last = None
        nodes, classes, seen, values, tensors = [], set(), set(), [], []
        stack = [root for root in (self.model, self.coulomb, self.dftd3) if root is not None]
        while stack:
            mod = stack.pop()
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            cls = type(mod)
            if cls not in _TYPES:
                self._nodes = ()
                return
            classes.add(cls)
            state = vars(mod)
            # Callable overrides and compiled dispatch are not graph-compatible,
            # even if their effect happens to vanish at the capture geometry.
            if any(name in state for name, _ in _METHODS[cls]):
                self._nodes = ()
                return
            if state.get("_compiled_call_impl") is not None:
                self._nodes = ()
                return
            config = tuple(name for name in state if name not in _BOOKKEEPING)
            try:
                tuple(_value(state[name]) for name in config)
            except (TypeError, RuntimeError):
                self._nodes = ()
                return
            children = tuple(state["_modules"].items())
            params, buffers = state["_parameters"], state["_buffers"]
            nodes.append((mod, cls, state, tuple(state), children, params, len(params), buffers, len(buffers)))
            if config:
                value_types = tuple(type(state[name]) for name in config)
                plain = all(cls in (bool, int, float, str, type(None)) for cls in value_types)
                values.append((state, itemgetter(*config), value_types, len(config) == 1, plain))
            for mapping in (params, buffers):
                tensors.extend((mapping, name) for name in mapping)
            stack.extend(child for _, child in children if child is not None)
        self._nodes, self._classes = tuple(nodes), tuple(classes)
        self._values, self._tensors = tuple(values), tuple(tensors)
        self._methods = tuple(
            (
                cls,
                attrgetter(*(name for name, _ in _METHODS[cls])),
                tuple(id(original) for _, original in _METHODS[cls]),
            )
            for cls in classes
        )

    def snapshot(self):
        """Return current mutation token or None; performs no CUDA operations."""
        if any(getattr(module_hooks, name, None) for name in _GLOBAL_HOOKS):
            return None
        try:
            if not self._nodes:
                self._rebuild()
                if not self._nodes:
                    return None
            if any(tuple(map(id, getter(cls))) != expected for cls, getter, expected in self._methods):
                return None
            token = [self._generation]
            for mod, cls, state, keys, children, params, nparam, buffers, nbuffer in self._nodes:
                # Identity of children avoids Tensor/module equality operators.
                if (
                    type(mod) is not cls
                    or tuple(state) != keys
                    or tuple(state["_modules"].items()) != children
                    or state["_parameters"] is not params
                    or len(params) != nparam
                    or state["_buffers"] is not buffers
                    or len(buffers) != nbuffer
                ):
                    self._rebuild()
                    return self.snapshot() if self._nodes else None
                if (
                    state["_forward_hooks"]
                    or state["_forward_pre_hooks"]
                    or state["_backward_hooks"]
                    or state["_backward_pre_hooks"]
                ):
                    return None
            for state, getter, types, single, plain in self._values:
                values = getter(state)
                if single:
                    values = (values,)
                if tuple(map(type, values)) != types:
                    self._rebuild()
                    return self.snapshot() if self._nodes else None
                token.append(values if plain else tuple(map(_value, values)))
            for mapping, name in self._tensors:
                value = mapping[name]
                if value is None:
                    token.append(None)
                else:
                    if value.requires_grad:
                        return None
                    token.append((id(value), value._version, value.dtype, value.device))
            token = tuple(token)
            if token == self._last:
                return self._last
            if not self._supported():
                return None
            self._last = token
            return token  # noqa: TRY300 - fail-closed checks intentionally share one exception guard
        except (AttributeError, KeyError, TypeError, RuntimeError):
            # Changed types / inference tensors / malformed custom modules stay
            # on the regular path rather than bypassing its validation.
            return None
