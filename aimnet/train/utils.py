import inspect
import logging
import re
from collections.abc import Callable

import numpy as np
import omegaconf
import torch
from ignite import distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, ProgressBar, TerminateOnNan, global_step_from_engine
from omegaconf import OmegaConf
from torch import Tensor, nn

from aimnet import nbops
from aimnet.config import build_module, get_init_module, get_module, load_yaml
from aimnet.data import SizeGroupedDataset
from aimnet.modules import Forces


def enable_tf32(enable=True):
    """Toggle TF32 reduced-precision float32 matmul (training-throughput knob).

    NOTE: this sets *process-global* float32 matmul precision. Never call it
    from inference/calculator code paths used for thermochemistry — those must
    run at full precision ("highest"). The deliberate float64 energy
    accumulation in aimnet.modules.lr is unaffected (TF32 governs float32 GEMM
    only).
    """
    # set_float32_matmul_precision is the forward-compatible control; the
    # allow_tf32 booleans are its deprecated alias from torch 2.9 onward.
    torch.set_float32_matmul_precision("high" if enable else "highest")
    # cudnn keeps a dedicated flag not covered by set_float32_matmul_precision.
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enable


def _to_config_dict(cfg: omegaconf.DictConfig, name: str) -> dict:
    d = OmegaConf.to_container(cfg)
    if not isinstance(d, dict):
        raise TypeError(f"{name} configuration must be a dictionary.")
    return d


def load_dataset(cfg: omegaconf.DictConfig, kind="train"):
    # only load required subset of keys
    keys = list(cfg.x) + list(cfg.y)
    # in DDP setting, will only load 1/WORLD_SIZE of the data
    if idist.get_world_size() > 1 and not cfg.ddp_load_full_dataset:
        shard = (idist.get_rank(), idist.get_world_size())
    else:
        shard = None

    d = _to_config_dict(cfg.datasets[kind], "Dataset")
    kwargs = dict(d.get("kwargs", {}))
    kwargs.update({"keys": keys, "shard": shard})
    d["kwargs"] = kwargs
    d["args"] = [cfg[kind]]
    ds = build_module(d)  # type: ignore[arg-type]
    ds = apply_sae(ds, cfg)  # type: ignore[arg-type]
    return ds


def apply_sae(ds: SizeGroupedDataset, cfg: omegaconf.DictConfig):
    for k, c in cfg.sae.items():
        if c is not None and k in cfg.y:
            sae = load_yaml(c.file)
            if not isinstance(sae, dict):
                raise TypeError(f"SAE file {c.file} must contain a dictionary.")
            unique_numbers = set(np.unique(ds.concatenate("numbers").tolist()))
            if not unique_numbers.issubset(sae.keys()):
                raise ValueError(f"Keys in SAE file {c.file} do not cover all the dataset atoms")
            if c.mode == "linreg":
                ds.apply_peratom_shift(k, k, sap_dict=sae)
            elif c.mode == "logratio":
                ds.apply_pertype_logratio(k, k, sap_dict=sae)
            else:
                raise ValueError(f"Unknown SAE mode {c.mode}")
            for g in ds.groups:
                g[k] = g[k].astype("float32")
    return ds


def get_sampler(ds: SizeGroupedDataset, cfg: omegaconf.DictConfig, kind="train"):
    d = _to_config_dict(cfg.samplers[kind], "Sampler")
    if "kwargs" not in d:
        d["kwargs"] = {}
    d["kwargs"]["ds"] = ds
    sampler = build_module(d)
    return sampler


def log_ds_group_sizes(ds):
    logging.info("Group sizes")
    for _n, g in ds.items():
        logging.info(f"{_n:03d}: {len(g)}")


def get_loaders(cfg: omegaconf.DictConfig):
    ds_train: SizeGroupedDataset
    # load datasets
    ds_train = load_dataset(cfg, kind="train")
    logging.info(f"Loaded train dataset from {cfg.train} with {len(ds_train)} samples.")
    log_ds_group_sizes(ds_train)
    if cfg.val is not None:
        ds_val = load_dataset(cfg, kind="val")
        logging.info(f"Loaded validation dataset from {cfg.val} with {len(ds_val)} samples.")
    else:
        if cfg.separate_val:
            ds_train, ds_val = ds_train.random_split(1 - cfg.val_fraction, cfg.val_fraction)
            logging.info(
                f"Randomly train dataset into train and val datasets, sizes {len(ds_train)} and {len(ds_val)} {cfg.val_fraction * 100:.1f}%."
            )
        else:
            ds_val = ds_train.random_split(cfg.val_fraction)[0]
            logging.info(
                f"Using a random fraction ({cfg.val_fraction * 100:.1f}%, {len(ds_val)} samples) of train dataset for validation."
            )

    # merge small groups
    ds_train.merge_groups(
        min_size=8 * cfg.samplers.train.kwargs.batch_size, mode_atoms=cfg.samplers.train.kwargs.batch_mode == "atoms"
    )
    logging.info("After merging small groups in train dataset")
    log_ds_group_sizes(ds_train)

    loader_train = ds_train.get_loader(get_sampler(ds_train, cfg, kind="train"), cfg.x, cfg.y, **cfg.loaders.train)
    loader_val = ds_val.get_loader(get_sampler(ds_val, cfg, kind="val"), cfg.x, cfg.y, **cfg.loaders.val)
    return loader_train, loader_val


def get_optimizer(model: nn.Module, cfg: omegaconf.DictConfig):
    logging.info("Building optimizer")
    param_groups = {}
    for k, c in cfg.param_groups.items():
        c = _to_config_dict(c, "Param group")
        c.pop("re")
        param_groups[k] = {"params": [], **c}
    param_groups["default"] = {"params": []}
    logging.info(f"Default parameters: {cfg.kwargs}")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        _matched = False
        for k, c in cfg.param_groups.items():
            if re.search(c.re, n):
                param_groups[k]["params"].append(p)
                logging.info(f"{n}: {c}")
                _matched = True
                break
        if not _matched:
            param_groups["default"]["params"].append(p)
    d = _to_config_dict(cfg, "Optimizer")
    d["args"] = [[v for v in param_groups.values() if len(v["params"])]]
    optimizer = get_init_module(d["class"], d["args"], d["kwargs"])
    logging.info(f"Optimizer: {optimizer}")
    logging.info("Trainable parameters:")
    N = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(f"{n}: {p.shape}")
            N += p.numel()
    logging.info(f"Total number of trainable parameters: {N}")
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: omegaconf.DictConfig):
    d = _to_config_dict(cfg, "Scheduler")
    d["args"] = [optimizer]
    scheduler = build_module(d)
    return scheduler


def get_loss(cfg: omegaconf.DictConfig):
    d = _to_config_dict(cfg, "Loss")
    loss = build_module(d)
    return loss


def set_trainable_parameters(model: nn.Module, force_train: list[str], force_no_train: list[str]) -> nn.Module:
    for n, p in model.named_parameters():
        if any(re.search(x, n) for x in force_no_train):
            p.requires_grad_(False)
            logging.info(f"requires_grad {n} {p.requires_grad}")
        if any(re.search(x, n) for x in force_train):
            p.requires_grad_(True)
            logging.info(f"requires_grad {n} {p.requires_grad}")
    return model


def unwrap_module(net):
    while isinstance(net, (Forces, _CompiledTrainingRunner, torch.nn.parallel.DistributedDataParallel)):
        net = net.core if isinstance(net, _CompiledTrainingRunner) else net.module
    return net


def build_model(cfg, forces=False):
    d = _to_config_dict(cfg, "Model")
    model = build_module(d)
    if forces:
        model = Forces(model)  # type: ignore[attr-defined]
    return model


def get_metrics(cfg: omegaconf.DictConfig):
    d = _to_config_dict(cfg, "Metrics")
    metrics = build_module(d)
    return metrics


def prepare_batch(batch: dict[str, Tensor], device="cuda", non_blocking=True) -> dict[str, Tensor]:
    for k, v in batch.items():
        if v.is_floating_point() and v.dtype != torch.float32:
            v = v.float()
        batch[k] = v.to(device, non_blocking=non_blocking)
    return batch


class _CompiledTrainingRunner(nn.Module):
    """Run one fixed derivative contract against the original AIMNet2 module."""

    _DYNAMIC_INPUTS = frozenset({
        "coord",
        "numbers",
        "charge",
        "mult",
        "mol_idx",
        "cell",
        "pbc",
        "nbmat",
        "nbmat_lr",
        "nbmat_coulomb",
        "nbmat_dftd3",
        "shifts",
        "shifts_lr",
        "shifts_coulomb",
        "shifts_dftd3",
    })

    def __init__(self, core: nn.Module, target_keys: tuple[str, ...] = (), *, compile_training: bool = True):
        super().__init__()
        self.core = core
        self.target_keys = target_keys
        self.compile_training = compile_training
        self._input_schema: tuple[tuple[str, int, torch.dtype, str, tuple[int, ...] | None], ...] | None = None
        self._input_keys: tuple[str, ...] = ()
        self._state_names: tuple[str, ...] = ()
        self._output_keys: tuple[str, ...] = ()
        self._neighbor_mode: int | None = None
        self._compiled_forward: Callable[..., tuple[Tensor, ...]] | None = None

    @property
    def need_forces(self) -> bool:
        return "forces" in self.target_keys

    @property
    def need_stress(self) -> bool:
        return "stress" in self.target_keys

    def set_target_keys(self, target_keys: tuple[str, ...]) -> None:
        if self.target_keys and self.target_keys != target_keys:
            raise ValueError("Training target properties changed after the derivative runner's first batch.")
        self.target_keys = target_keys

    def _validate_stress_input(self, data: dict[str, Tensor]) -> None:
        if not self.need_stress:
            return
        mode = nbops.infer_nb_mode(data)
        if mode == 0:
            raise ValueError(
                "Stress training requires explicit neighbor topology (mode 1 or 2); dense mode 0 is not supported."
            )
        if "cell" not in data:
            raise ValueError("Stress training requires a cell input.")
        for suffix in nbops.NBMAT_SUFFIXES:
            neighbor_key, shifts_key = f"nbmat{suffix}", f"shifts{suffix}"
            if neighbor_key in data and shifts_key not in data:
                raise ValueError(f"Stress training requires a matching {shifts_key!r} tensor for {neighbor_key!r}.")
            if neighbor_key in data and data[shifts_key].shape != (*data[neighbor_key].shape, 3):
                raise ValueError(f"Stress training requires {shifts_key!r} to align with {neighbor_key!r}.")

    @staticmethod
    def _normalize_cell(cell: Tensor, n_systems: int) -> Tensor:
        if cell.ndim == 2 and cell.shape == (3, 3):
            cell = cell.unsqueeze(0)
        if cell.ndim != 3 or cell.shape[-2:] != (3, 3) or cell.shape[0] not in (1, n_systems):
            raise ValueError(f"cell must have shape (3, 3), (1, 3, 3), or (B, 3, 3) with B={n_systems}.")
        return cell.expand(n_systems, -1, -1).contiguous() if cell.shape[0] == 1 and n_systems != 1 else cell

    def validate_mode1_indices(self, data: dict[str, Tensor]) -> None:
        if nbops.infer_nb_mode(data) != 1:
            return
        numbers, mol_idx, charge = data["numbers"], data["mol_idx"], data["charge"]
        if numbers.ndim != 1 or mol_idx.ndim != 1 or numbers.shape != mol_idx.shape:
            raise ValueError("Mode-1 training requires aligned one-dimensional numbers and mol_idx tensors.")
        n_systems = charge.shape[0]
        real = numbers != 0
        if bool(((mol_idx[real] < 0) | (mol_idx[real] >= n_systems)).any()):
            raise ValueError("Mode-1 training requires every real mol_idx entry to index a declared system.")

    def _normalize_mode1_dummy(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if nbops.infer_nb_mode(data) != 1:
            return data
        numbers, mol_idx, charge = data["numbers"], data["mol_idx"], data["charge"]
        n_systems = charge.shape[0]
        real = numbers != 0
        normalized = torch.where(real, mol_idx, torch.full_like(mol_idx, n_systems - 1)).to(torch.long)
        return {**data, "mol_idx": normalized}

    def _schema(self, data: dict[str, Tensor]) -> tuple[tuple[str, int, torch.dtype, str, tuple[int, ...] | None], ...]:
        return tuple(
            (
                key,
                value.ndim,
                value.dtype,
                value.device.type,
                None if key in self._DYNAMIC_INPUTS else tuple(value.shape),
            )
            for key, value in data.items()
        )

    def _validate_input(self, data: dict[str, Tensor]) -> None:
        schema = self._schema(data)
        if self._input_schema is None:
            if data["numbers"].device.type == "cpu":
                self.validate_mode1_indices(data)
            self._input_schema = schema
            self._input_keys = tuple(data)
            self._neighbor_mode = nbops.infer_nb_mode(data)
            return
        if tuple(key for key, *_ in schema) != self._input_keys:
            expected, actual = set(self._input_keys), set(data)
            if expected == actual:
                raise ValueError("Compiled training input key order changed after the first batch.")
            raise ValueError(
                "Compiled training input keys changed after the first batch; "
                f"missing {tuple(sorted(expected - actual))}, unexpected {tuple(sorted(actual - expected))}."
            )
        if self._neighbor_mode != nbops.infer_nb_mode(data):
            raise ValueError("Compiled training neighbor mode changed after the first batch.")
        for expected, actual in zip(self._input_schema, schema, strict=True):
            key, rank, dtype, device_type, fixed_shape = expected
            if actual[1:4] != (rank, dtype, device_type):
                raise ValueError(f"Compiled training input schema changed for {key!r} (rank, dtype, or device type).")
            if fixed_shape is not None and actual[4] != fixed_shape:
                raise ValueError(f"Compiled training fixed-shape input changed for {key!r}.")

    def _live_state_tensors(self) -> tuple[Tensor, ...]:
        state = dict(self.core.named_parameters())
        state.update(self.core.named_buffers())
        return tuple(state[name] for name in self._state_names)

    def _derivative_predictions(
        self,
        data: dict[str, Tensor],
        apply_core: Callable[[dict[str, Tensor]], dict[str, Tensor]],
        *,
        create_graph: bool,
    ) -> dict[str, Tensor]:
        """Run the common energy/forces/stress derivative topology."""
        data = self._normalize_mode1_dummy(data)
        coord = data["coord"].detach().requires_grad_(True)
        data["coord"] = coord
        strain = None
        if self.need_stress:
            n_systems = data["charge"].shape[0]
            cell = self._normalize_cell(data["cell"], n_systems)
            strain = torch.eye(3, dtype=coord.dtype, device=coord.device).unsqueeze(0).repeat(n_systems, 1, 1)
            strain.requires_grad_(True)
            if coord.ndim == 2:
                data["coord"] = torch.einsum("ni,nij->nj", coord, strain[data["mol_idx"]])
            else:
                data["coord"] = torch.einsum("bni,bij->bnj", coord, strain)
            data["cell"] = cell @ strain
        data = apply_core(data)
        if self.need_forces or self.need_stress:
            inputs = (coord,) if strain is None else (coord, strain)
            derivatives = torch.autograd.grad(
                data["energy"].sum(), inputs, create_graph=create_graph, retain_graph=create_graph
            )
            if self.need_forces:
                data["forces"] = -derivatives[0]
            if strain is not None:
                volume = torch.linalg.det(data["cell"].detach()).abs().unsqueeze(-1).unsqueeze(-1)
                data["stress"] = derivatives[-1] / volume
        return data

    def _capture(self, example: dict[str, Tensor]) -> None:
        # Keep private Torch compiler imports local: their APIs change faster
        # than AIMNet2's public training surface.
        try:
            from torch._dynamo.source import ConstantSource
            from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler
            from torch._inductor.compile_fx import compile_fx
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.func import functional_call
            from torch.fx.experimental.proxy_tensor import make_fx
            from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
        except (AttributeError, ImportError) as error:
            raise RuntimeError(
                "Compiled training requires PyTorch 2.10 or newer with its AOTAutograd and Inductor compiler APIs."
            ) from error

        self._state_names = tuple(name for name, _ in (*self.core.named_parameters(), *self.core.named_buffers()))
        self._output_keys = tuple(dict.fromkeys((*self.target_keys, "_natom", "_input_padded", "numbers")))

        def derivative_forward(*tensors: Tensor) -> tuple[Tensor, ...]:
            data = dict(zip(self._input_keys, tensors[: len(self._input_keys)], strict=True))
            state = dict(zip(self._state_names, tensors[len(self._input_keys) :], strict=True))
            data = self._derivative_predictions(
                data,
                lambda values: functional_call(self.core, state, (values,)),
                create_graph=True,
            )
            return tuple(data[key] for key in self._output_keys)

        shape_env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)
        mode = nbops.infer_nb_mode(example)
        symbols: dict[str, torch.SymInt] = {}

        def symbolic_size(label: str, hint: int) -> torch.SymInt:
            if label not in symbols:
                source = ConstantSource(f"aimnet2_{label}")
                expression = shape_env.create_symbol(
                    max(hint, 2),
                    source,
                    dynamic_dim=DimDynamic.DYNAMIC,
                    do_not_specialize_zero_one=True,
                )
                symbols[label] = shape_env.create_symintnode(expression, hint=max(hint, 2), source=source)
            return symbols[label]

        def symbolic_shape(key: str, value: Tensor) -> tuple[int | torch.SymInt, ...] | None:
            B = symbolic_size("B", example["charge"].shape[0])

            def neighbor_width(prefix: str, axis: int) -> torch.SymInt:
                suffix = key.removeprefix(prefix)
                return symbolic_size(f"M{suffix}", value.shape[axis])

            if mode == 0:
                N = symbolic_size("N", example["numbers"].shape[1])
                if key in {"coord", "numbers"}:
                    return (B, N, *value.shape[2:])
                if key in {"charge", "mult"}:
                    return (B, *value.shape[1:])
                if key == "cell" and value.ndim == 3:
                    return (B, 3, 3)
                if key == "pbc" and value.ndim == 2:
                    return (B, 3)
            elif mode == 1:
                L = symbolic_size("L", example["numbers"].shape[0])
                if key in {"coord", "numbers", "mol_idx"}:
                    return (L, *value.shape[1:])
                if key in {"charge", "mult"}:
                    return (B, *value.shape[1:])
                if key.startswith("nbmat") and value.ndim == 2:
                    M = neighbor_width("nbmat", 1)
                    return (L, M)
                if key.startswith("shifts") and value.ndim == 3:
                    M = neighbor_width("shifts", 1)
                    return (L, M, 3)
                if key == "cell" and value.ndim == 3:
                    return (B, 3, 3)
                if key == "pbc" and value.ndim == 2:
                    return (B, 3)
            else:
                N = symbolic_size("N", example["numbers"].shape[1])
                if key in {"coord", "numbers"}:
                    return (B, N, *value.shape[2:])
                if key in {"charge", "mult"}:
                    return (B, *value.shape[1:])
                if key.startswith("nbmat") and value.ndim == 3:
                    M = neighbor_width("nbmat", 2)
                    return (B, N, M)
                if key.startswith("shifts") and value.ndim == 4:
                    M = neighbor_width("shifts", 2)
                    return (B, N, M, 3)
                if key == "cell" and value.ndim == 3:
                    return (B, 3, 3)
                if key == "pbc" and value.ndim == 2:
                    return (B, 3)
            return None

        def contiguous_strides(shape: tuple[int | torch.SymInt, ...]) -> tuple[int | torch.SymInt, ...]:
            strides: list[int | torch.SymInt] = []
            stride: int | torch.SymInt = 1
            for size in reversed(shape):
                strides.append(stride)
                stride = stride * size
            return tuple(reversed(strides))

        with fake_mode, nbops._symbolic_trace_context():
            fake_inputs = []
            for key, value in example.items():
                shape = symbolic_shape(key, value)
                if shape is None:
                    fake_inputs.append(fake_mode.from_tensor(value, static_shapes=True))
                else:
                    fake_inputs.append(
                        torch.empty_strided(shape, contiguous_strides(shape), dtype=value.dtype, device=value.device)
                    )
            fake_state = tuple(fake_mode.from_tensor(value, static_shapes=True) for value in self._live_state_tensors())
            fake_args = (*fake_inputs, *fake_state)
            traced = make_fx(
                derivative_forward,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
                _error_on_data_dependent_ops=True,
            )(*fake_args)
        # Compile from the captured fake inputs so AOTAutograd preserves this
        # ShapeEnv. Rebuilding fake inputs from the first real batch would
        # specialize singleton dimensions and lose shared size relationships.
        object.__setattr__(
            self,
            "_compiled_forward",
            aot_module_simplified(
                traced,
                fake_args,
                fw_compiler=make_boxed_compiler(compile_fx),
                bw_compiler=make_boxed_compiler(compile_fx),
            ),
        )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        self._validate_stress_input(data)
        if self.need_stress:
            data = {**data, "cell": self._normalize_cell(data["cell"], data["charge"].shape[0])}
        self._validate_input(data)
        if not self.compile_training:
            with torch.enable_grad():
                return self._derivative_predictions(data, self.core, create_graph=self.training)
        if self._compiled_forward is None:
            self._capture(data)
        assert self._compiled_forward is not None
        args = (*(data[key] for key in self._input_keys), *self._live_state_tensors())
        values = self._compiled_forward(*args)
        result = dict(data)
        result.update(zip(self._output_keys, values, strict=True))
        return result


def build_compiled_training_runner(
    model: nn.Module, target_keys: tuple[str, ...], *, compile_training: bool = True
) -> _CompiledTrainingRunner:
    return _CompiledTrainingRunner(unwrap_module(model), target_keys, compile_training=compile_training)


def _eager_derivative_predictions(
    model: nn.Module, data: dict[str, Tensor], target_keys: tuple[str, ...]
) -> dict[str, Tensor]:
    """Evaluate energy/derivative targets through the same stress topology."""
    contract = _CompiledTrainingRunner(unwrap_module(model), target_keys)
    contract._validate_stress_input(data)
    return contract._derivative_predictions(data, unwrap_module(model), create_graph=False)


def default_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable | torch.nn.Module,
    device: str | torch.device | None = None,
    non_blocking: bool = True,
    compile_training: bool = False,
) -> Engine:
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    if compile_training and (model_device.type != "cuda" or target_device.type != "cuda"):
        raise RuntimeError("Compiled training requires the model and batches to use CUDA.")
    if compile_training and not isinstance(model, (torch.nn.parallel.DistributedDataParallel, _CompiledTrainingRunner)):
        # Direct callers may still pass a regular core/Forces model. The CLI
        # builds this runner earlier so DDP sees the runner itself.
        model = build_compiled_training_runner(model, ())
    if (
        compile_training
        and isinstance(model, torch.nn.parallel.DistributedDataParallel)
        and not isinstance(model.module, _CompiledTrainingRunner)
    ):
        raise RuntimeError("Compiled DDP training requires a compiled training runner before DDP wrapping.")
    forward = model

    def _check_compiled_batch(x: dict[str, Tensor]) -> None:
        if "numbers" not in x:
            raise ValueError("Compiled training requires a numbers input.")

    def _any_rank_failed(flag: Tensor, message: str) -> None:
        flag = flag.to(device=model_device, dtype=torch.int32)
        if idist.get_world_size() > 1:
            flag = idist.all_reduce(flag, op="MAX")
        if bool(flag.item()):
            raise RuntimeError(message)

    def _update(engine: Engine, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> float:
        model.train()
        optimizer.zero_grad()
        source_x = dict(batch[0])
        runner = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if isinstance(runner, _CompiledTrainingRunner) and source_x["numbers"].device.type == "cpu":
            runner.validate_mode1_indices(source_x)
        x = prepare_batch(source_x, device=device, non_blocking=non_blocking)  # type: ignore
        y = prepare_batch(dict(batch[1]), device=device, non_blocking=non_blocking)  # type: ignore
        if compile_training:
            _check_compiled_batch(x)
            assert isinstance(runner, _CompiledTrainingRunner)
            runner.set_target_keys(tuple(y))
        y_pred = forward(x)
        loss = loss_fn(y_pred, y)["loss"]
        _any_rank_failed(~torch.isfinite(loss).all(), "Non-finite training loss; optimizer step skipped.")
        loss.backward()
        gradient_checks = [
            torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None
        ]
        gradients_finite = (
            torch.stack(gradient_checks).all()
            if gradient_checks
            else torch.ones((), device=model_device, dtype=torch.bool)
        )
        _any_rank_failed(~gradients_finite, "Non-finite training gradients; optimizer step skipped.")
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
        optimizer.step()

        return loss.item()

    return Engine(_update)


def default_evaluator(
    model: torch.nn.Module,
    device: str | torch.device | None = None,
    non_blocking: bool = True,
    stress: bool = False,
) -> Engine:
    def _inference(
        engine: Engine, batch: tuple[dict[str, Tensor], dict[str, Tensor]]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        model.eval()
        x = prepare_batch(dict(batch[0]) if stress else batch[0], device=device, non_blocking=non_blocking)  # type: ignore
        y = prepare_batch(batch[1], device=device, non_blocking=non_blocking)  # type: ignore
        if not stress:
            with torch.no_grad():
                y_pred = model(x)
            return y_pred, y

        target_keys = tuple(dict.fromkeys((*y, "stress")))
        if (
            isinstance(model, torch.nn.parallel.DistributedDataParallel)
            and isinstance(model.module, _CompiledTrainingRunner)
        ) or isinstance(model, _CompiledTrainingRunner):
            runner = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            if not runner.need_stress:
                raise ValueError("Stress evaluation requires a derivative runner configured with stress.")
            y_pred = model(x)
        else:
            y_pred = _eager_derivative_predictions(model, x, target_keys)
        # Validation does not retain the derivative graph.
        return {key: value.detach() if isinstance(value, Tensor) else value for key, value in y_pred.items()}, y

    return Engine(_inference)


class TerminateOnLowLR:
    def __init__(self, optimizer, low_lr=1e-5):
        self.low_lr = low_lr
        self.optimizer = optimizer

    def __call__(self, engine):
        if self.optimizer.param_groups[0]["lr"] < self.low_lr:
            engine.terminate()


def build_engine(model, optimizer, scheduler, loss_fn, metrics, cfg, loader_val):
    device = next(model.parameters()).device
    evaluator_name = cfg.trainer.evaluator
    need_stress = "stress" in cfg.data.y

    train_fn = get_module(cfg.trainer.trainer)
    train_kwargs = {"device": device, "non_blocking": True}
    if bool(cfg.trainer.get("compile", False)) and train_fn is default_trainer:
        train_kwargs["compile_training"] = True
    trainer = train_fn(model, optimizer, loss_fn, **train_kwargs)
    # check for NaNs after each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # log LR
    def log_lr(engine):
        lr = optimizer.param_groups[0]["lr"]
        logging.info(f"LR: {lr}")

    trainer.add_event_handler(Events.EPOCH_STARTED, log_lr)

    # log loss weights
    def log_loss_weights(engine):
        s = []
        for k, v in loss_fn.components.items():
            s.append(f"{k}: {v[1]:.4f}")
        s = " ".join(s)
        logging.info(s)

    trainer.add_event_handler(Events.EPOCH_STARTED, log_loss_weights)

    # write TQDM progress
    if idist.get_local_rank() == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, event_name=Events.ITERATION_COMPLETED(every=100))

    # attach validator
    validate_fn = get_module(evaluator_name)
    evaluator_kwargs = {"device": device, "non_blocking": True}
    if need_stress and (
        "stress" in inspect.signature(validate_fn).parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(validate_fn).parameters.values()
        )
    ):
        evaluator_kwargs["stress"] = True
    validator = validate_fn(model, **evaluator_kwargs)
    metrics.attach(validator, "multi")
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), validator.run, data=loader_val)

    # attach optimizer and loss to engines
    trainer.state.optimizer = optimizer
    trainer.state.loss_fn = loss_fn
    validator.state.optimizer = optimizer
    validator.state.loss_fn = loss_fn

    # scheduler
    if scheduler is not None:
        validator.state.scheduler = scheduler
        validator.add_event_handler(Events.COMPLETED, scheduler)
        terminator = TerminateOnLowLR(optimizer, cfg.scheduler.terminate_on_low_lr)
        trainer.add_event_handler(Events.EPOCH_STARTED, terminator)

    # checkpoint after each epoch
    if cfg.checkpoint and idist.get_local_rank() == 0:
        kwargs = OmegaConf.to_container(cfg.checkpoint.kwargs) if "kwargs" in cfg.checkpoint else {}
        if not isinstance(kwargs, dict):
            raise TypeError("Checkpoint kwargs must be a dictionary.")
        kwargs["global_step_transform"] = global_step_from_engine(trainer)
        kwargs["dirname"] = cfg.checkpoint.dirname
        kwargs["filename_prefix"] = cfg.checkpoint.filename_prefix
        checkpointer = ModelCheckpoint(**kwargs)  # type: ignore
        validator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {"model": unwrap_module(model)})

    return trainer, validator


def setup_wandb(cfg, model_cfg, model, trainer, validator, optimizer):
    import wandb
    from ignite.handlers import WandBLogger, global_step_from_engine
    from ignite.handlers.wandb_logger import OptimizerParamsHandler

    init_kwargs = OmegaConf.to_container(cfg.wandb.init, resolve=True)
    wandb.init(**init_kwargs)  # type: ignore
    wandb_logger = WandBLogger(init=False)

    OmegaConf.save(model_cfg, wandb.run.dir + "/model.yaml")  # type: ignore
    OmegaConf.save(cfg, wandb.run.dir + "/train.yaml")  # type: ignore

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=200),
        output_transform=lambda loss: {"loss": loss},
        tag="train",
    )
    wandb_logger.attach_output_handler(
        validator,
        event_name=Events.EPOCH_COMPLETED,
        global_step_transform=lambda *_: trainer.state.iteration,
        metric_names="all",
        tag="val",
    )

    class EpochLRLogger(OptimizerParamsHandler):
        def __call__(self, engine, logger, event_name):
            global_step = engine.state.iteration
            params = {
                f"{self.param_name}_{i}": float(g[self.param_name]) for i, g in enumerate(self.optimizer.param_groups)
            }
            if hasattr(engine.state, "loss_fn") and hasattr(engine.state.loss_fn, "components"):  # type: ignore
                for name, (_, w) in engine.state.loss_fn.components.items():  # type: ignore
                    params[f"weight/{name}"] = w
            logger.log(params, step=global_step, sync=self.sync)

    wandb_logger.attach(trainer, log_handler=EpochLRLogger(optimizer), event_name=Events.EPOCH_STARTED)

    score_function = lambda engine: 1.0 / engine.state.metrics["loss"]
    model_checkpoint = ModelCheckpoint(
        wandb.run.dir,  # type: ignore
        n_saved=1,
        filename_prefix="best",  # type: ignore
        require_empty=False,
        score_function=score_function,
        global_step_transform=global_step_from_engine(trainer),
    )
    validator.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {"model": unwrap_module(model)})

    if cfg.wandb.watch_model:
        wandb.watch(unwrap_module(model), **OmegaConf.to_container(cfg.wandb.watch_model, resolve=True))  # type: ignore
