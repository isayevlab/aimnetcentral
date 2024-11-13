import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import omegaconf
import torch
from ignite import distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, ProgressBar, TerminateOnNan, global_step_from_engine
from omegaconf import OmegaConf
from torch import Tensor, nn

from aimnet.config import build_module, get_init_module, get_module, load_yaml
from aimnet.data import SizeGroupedDataset
from aimnet.modules import Forces


def enable_tf32(enable=True):
    if enable:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def make_seed(all_reduce=True):
    # create seed
    seed = int.from_bytes(os.urandom(2), "big")
    if all_reduce and idist.get_world_size() > 1:
        seed = idist.all_reduce(seed)


def load_dataset(cfg: omegaconf.DictConfig, kind="train"):
    # only load required subset of keys
    keys = list(cfg.x) + list(cfg.y)
    # in DDP setting, will only load 1/WORLD_SIZE of the data
    if idist.get_world_size() > 1 and not cfg.ddp_load_full_dataset:
        shard = (idist.get_local_rank(), idist.get_world_size())
    else:
        shard = None

    extra_kwargs = {
        "keys": keys,
        "shard": shard,
    }
    cfg.datasets[kind].kwargs.update(extra_kwargs)
    cfg.datasets[kind].args = [cfg[kind]]
    ds = build_module(OmegaConf.to_container(cfg.datasets[kind]))  # type: ignore[arg-type]
    ds = apply_sae(ds, cfg)  # type: ignore[arg-type]
    return ds


def apply_sae(ds: SizeGroupedDataset, cfg: omegaconf.DictConfig):
    for k, c in cfg.sae.items():
        if c is not None and k in cfg.y:
            sae = load_yaml(c.file)
            unique_numbers = set(np.unique(ds.concatenate("numbers").tolist()))
            if not set(sae.keys()).issubset(unique_numbers):  # type: ignore[attr-defined]
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
    d = OmegaConf.to_container(cfg.samplers[kind])
    if not isinstance(d, dict):
        raise TypeError("Sampler configuration must be a dictionary.")
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
        c = OmegaConf.to_container(c)
        if not isinstance(c, dict):
            raise TypeError("Param groups must be a dictionary.")
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
    d = OmegaConf.to_container(cfg)
    if not isinstance(d, dict):
        raise TypeError("Optimizer configuration must be a dictionary.")
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
    d = OmegaConf.to_container(cfg)
    if not isinstance(d, dict):
        raise TypeError("Scheduler configuration must be a dictionary.")
    d["args"] = [optimizer]
    scheduler = build_module(d)
    return scheduler


def get_loss(cfg: omegaconf.DictConfig):
    d = OmegaConf.to_container(cfg)
    if not isinstance(d, dict):
        raise TypeError("Loss configuration must be a dictionary.")
    loss = build_module(d)
    return loss


def set_trainable_parameters(model: nn.Module, force_train: List[str], force_no_train: List[str]) -> nn.Module:
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
    d = OmegaConf.to_container(cfg)
    if not isinstance(d, dict):
        raise TypeError("Model configuration must be a dictionary.")
    model = build_module(d)
    if forces is not None:
        model = Forces(model)  # type: ignore[attr-defined]
    return model


def get_metrics(cfg: omegaconf.DictConfig):
    d = OmegaConf.to_container(cfg)
    if not isinstance(d, dict):
        raise TypeError("Metrics configuration must be a dictionary.")
    metrics = build_module(d)
    return metrics


def train_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
    global model
    global optimizer
    global prepare_batch
    global loss_fn
    global device

    model.train()  # type: ignore
    optimizer.zero_grad()  # type: ignore
    x, y = prepare_batch(batch, device=device, non_blocking=True)  # type: ignore
    y_pred = model(x)  # type: ignore
    loss = loss_fn(y_pred, y)["loss"]  # type: ignore
    loss.backward()
    optimizer.step()  # type: ignore

    return loss.item()


def val_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
    global model
    global optimizer
    global prepare_batch
    global loss_fn
    global device

    model.eval()  # type: ignore
    if not next(iter(batch[0].values())).numel():
        return None
    x, y = prepare_batch(batch, device=device, non_blocking=True)  # type: ignore
    with torch.no_grad():
        y_pred = model(x)  # type: ignore
    return y_pred, y


def prepare_batch(batch: Dict[str, Tensor], device="cuda", non_blocking=True) -> Dict[str, Tensor]:  # noqa: F811
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=non_blocking)
    return batch


def default_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = True,
) -> Engine:
    def _update(engine: Engine, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> float:
        model.train()
        optimizer.zero_grad()
        x = prepare_batch(batch[0], device=device, non_blocking=non_blocking)  # type: ignore
        y = prepare_batch(batch[1], device=device, non_blocking=non_blocking)  # type: ignore
        y_pred = model(x)
        loss = loss_fn(y_pred, y)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
        optimizer.step()
        return loss.item()

    return Engine(_update)


def default_evaluator(
    model: torch.nn.Module, device: Optional[Union[str, torch.device]] = None, non_blocking: bool = True
) -> Engine:
    def _inference(
        engine: Engine, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        model.eval()
        x = prepare_batch(batch[0], device=device, non_blocking=non_blocking)  # type: ignore
        y = prepare_batch(batch[1], device=device, non_blocking=non_blocking)  # type: ignore
        with torch.no_grad():
            y_pred = model(x)
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

    train_fn = get_module(cfg.trainer.trainer)
    trainer = train_fn(model, optimizer, loss_fn, device=device, non_blocking=True)
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
    validate_fn = get_module(cfg.trainer.evaluator)
    validator = validate_fn(model, device=device, non_blocking=True)
    metrics.attach(validator, "multi")
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), validator.run, data=loader_val)

    # scheduler
    if scheduler is not None:
        validator.state.scheduler = scheduler
        validator.state.optimizer = optimizer
        validator.state.loss_fn = loss_fn
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
            if hasattr(engine, "loss_fn") and hasattr(engine.state.loss_fn, "components"):  # type: ignore
                for name, (_, w) in engine.state.loss_fn.components.items():  # type: ignore
                    params[name] = w
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
