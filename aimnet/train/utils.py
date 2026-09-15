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
from torch._decomp import core_aten_decompositions
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

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
    if isinstance(net, (Forces, torch.nn.parallel.DistributedDataParallel)):
        net = net.module
        return unwrap_module(net)
    else:
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


class _SymbolicTrainingForward:
    """Compile one fixed training layout from the first batch.

    The loader must keep input keys, tensor ranks and dtypes, and neighbor mode
    unchanged for that trainer.
    """

    def __init__(self, model: nn.Module, example: dict[str, Tensor], target_keys: tuple[str, ...]):
        self.module = model.module if isinstance(model, Forces) else model
        self.input_keys = tuple(example)
        self.state_names = tuple(name for name, _ in (*self.module.named_parameters(), *self.module.named_buffers()))
        self.force_key = model.key_out if isinstance(model, Forces) else None
        self.coord_key = model.x if isinstance(model, Forces) else "coord"
        self.energy_key = model.y if isinstance(model, Forces) else "energy"
        self.need_stress = "stress" in target_keys
        if self.need_stress and "cell" not in example:
            raise ValueError("Compiled training stress targets require a cell input.")
        if self.force_key is not None and self.coord_key not in example:
            raise ValueError(f"Compiled training force targets require {self.coord_key!r} input.")

        output_keys = tuple(dict.fromkeys((*target_keys, "_natom", "_input_padded")))

        def derivative_forward(*tensors: Tensor) -> tuple[Tensor, ...]:
            input_tensors = tensors[: len(self.input_keys)]
            state_tensors = tensors[len(self.input_keys) :]
            data = dict(zip(self.input_keys, input_tensors, strict=True))
            coord = data[self.coord_key]
            need_derivatives = self.force_key is not None or self.need_stress
            if need_derivatives:
                coord = coord.detach().requires_grad_(True)
                data[self.coord_key] = coord
            strain = None
            if self.need_stress:
                n_systems = data["cell"].shape[0] if coord.ndim == 2 else coord.shape[0]
                strain = (
                    torch.eye(3, dtype=coord.dtype, device=coord.device)
                    .unsqueeze(0)
                    .repeat(n_systems, 1, 1)
                    .requires_grad_(True)
                )
                if coord.ndim == 2:
                    data[self.coord_key] = torch.einsum("ni,nij->nj", coord, strain[data["mol_idx"]])
                else:
                    data[self.coord_key] = torch.einsum("bni,bij->bnj", coord, strain)
                data["cell"] = data["cell"] @ strain
            state = dict(zip(self.state_names, state_tensors, strict=True))
            data = functional_call(self.module, state, (data,))
            if need_derivatives:
                grad_inputs = [coord]
                if strain is not None:
                    grad_inputs.append(strain)
                derivatives = torch.autograd.grad(
                    data[self.energy_key].sum(), grad_inputs, create_graph=True, retain_graph=True
                )
                if self.force_key is not None:
                    data[self.force_key] = -derivatives[0]
                if strain is not None:
                    volume = torch.linalg.det(data["cell"].detach()).abs().unsqueeze(-1).unsqueeze(-1)
                    data["stress"] = derivatives[-1] / volume
            return tuple(data[key] for key in output_keys)

        example_args = (*tuple(example.values()), *self._live_state_tensors())
        # ``make_fx`` is not reported as compiling by Torch 2.13. AIMNet2's
        # local trace context selects its compile-safe tensor paths without
        # changing Torch's process-global compiler state.
        with nbops._symbolic_trace_context():
            traced = make_fx(
                derivative_forward,
                tracing_mode="symbolic",
                decomposition_table=core_aten_decompositions(),
                _allow_non_fake_inputs=True,
                _error_on_data_dependent_ops=True,
            )(*example_args)
        self.forward = torch.compile(traced, dynamic=True, fullgraph=False)
        self.output_keys = output_keys

    def _live_state_tensors(self) -> tuple[Tensor, ...]:
        state = dict(self.module.named_parameters())
        state.update(self.module.named_buffers())
        return tuple(state[name] for name in self.state_names)

    def __call__(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        input_key_set = set(x)
        expected_key_set = set(self.input_keys)
        if input_key_set != expected_key_set:
            missing = tuple(sorted(expected_key_set - input_key_set))
            unexpected = tuple(sorted(input_key_set - expected_key_set))
            raise ValueError(
                "Compiled training input keys changed after the first batch; "
                f"missing {missing}, unexpected {unexpected}."
            )
        values = self.forward(*(x[key] for key in self.input_keys), *self._live_state_tensors())
        y_pred = dict(x)
        y_pred.update(zip(self.output_keys, values, strict=True))
        return y_pred


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
    if compile_training and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise RuntimeError("Compiled training is not supported with DistributedDataParallel.")

    # Keep the original module as the optimizer/checkpoint owner. The compiled
    # callable covers the model and derivative computation, including its
    # autograd backward. Loss evaluation, clipping, and the optimizer step stay
    # eager; the trainer invokes loss.backward() to enter the compiled backward.
    if compile_training:
        # The first batch supplies the tensor schema for the symbolic
        # derivative graph. Energy-only training can continue to use the
        # ordinary module compile path.
        forward = None if isinstance(model, Forces) else torch.compile(model, dynamic=True)
    else:
        forward = model
    symbolic_forward: _SymbolicTrainingForward | None = None

    def _check_compiled_batch(x: dict[str, Tensor]) -> None:
        if "numbers" not in x:
            raise ValueError("Compiled training requires a numbers input.")

    def _update(engine: Engine, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> float:
        nonlocal symbolic_forward
        model.train()
        optimizer.zero_grad()
        x = prepare_batch(dict(batch[0]), device=device, non_blocking=non_blocking)  # type: ignore
        y = prepare_batch(dict(batch[1]), device=device, non_blocking=non_blocking)  # type: ignore
        if compile_training:
            _check_compiled_batch(x)
            if isinstance(model, Forces):
                if symbolic_forward is None:
                    symbolic_forward = _SymbolicTrainingForward(model, x, tuple(y))
                y_pred = symbolic_forward(x)
            else:
                assert forward is not None
                y_pred = forward(x)
        else:
            y_pred = forward(x)
        loss = loss_fn(y_pred, y)["loss"]
        loss.backward()
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

        if "cell" not in x:
            raise ValueError("Compiled training stress targets require a cell input.")
        module = model.module if isinstance(model, Forces) else model
        coord = x["coord"].detach().requires_grad_(True)
        n_systems = x["cell"].shape[0] if coord.ndim == 2 else coord.shape[0]
        strain = (
            torch.eye(3, dtype=coord.dtype, device=coord.device)
            .unsqueeze(0)
            .repeat(n_systems, 1, 1)
            .requires_grad_(True)
        )
        if coord.ndim == 2:
            x["coord"] = torch.einsum("ni,nij->nj", coord, strain[x["mol_idx"]])
        else:
            x["coord"] = torch.einsum("bni,bij->bnj", coord, strain)
        x["cell"] = x["cell"] @ strain
        y_pred = module(x)
        derivatives = torch.autograd.grad(y_pred["energy"].sum(), (coord, strain))
        if isinstance(model, Forces):
            y_pred[model.key_out] = -derivatives[0]
        volume = torch.linalg.det(x["cell"].detach()).abs().unsqueeze(-1).unsqueeze(-1)
        y_pred["stress"] = derivatives[1] / volume
        return y_pred, y

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
    need_stress = bool(cfg.trainer.get("compile", False)) and "stress" in cfg.data.y
    if need_stress and evaluator_name != "aimnet.train.utils.default_evaluator":
        raise RuntimeError("trainer.compile=True with stress targets requires aimnet.train.utils.default_evaluator.")

    train_fn = get_module(cfg.trainer.trainer)
    train_kwargs = {"device": device, "non_blocking": True}
    if bool(cfg.trainer.get("compile", False)):
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
    if need_stress:
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
