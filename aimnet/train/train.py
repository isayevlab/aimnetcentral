import logging
import os

import click
import omegaconf
import torch
from omegaconf import OmegaConf

from aimnet.train import utils

_default_model = os.path.join(os.path.dirname(__file__), "..", "models", "aimnet2.yaml")
_default_config = os.path.join(os.path.dirname(__file__), "default_train.yaml")


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    multiple=True,
    help="Path to the extra configuration file (overrides values, could be preficied multiple times).",
)
@click.option(
    "--model", type=click.Path(exists=True), default=_default_model, help="Path to the model definition file."
)
@click.option("--load", type=click.Path(exists=True), default=None, help="Path to the model weights to load.")
@click.option("--save", type=click.Path(), default=None, help="Path to save the model weights.")
@click.option(
    "--no-default-config",
    is_flag=True,
    default=False,
)
@click.argument("args", type=str, nargs=-1)
def train(config, model, load=None, save=None, args=None, no_default_config=False):
    """Train AIMNet2 model.
    By default, will load AIMNet2 model and default train config.
    ARGS are one or more parameters wo overwrite in config in a dot-separated form.
    For example: `train.data=mydataset.h5`.
    """
    logging.basicConfig(level=logging.INFO)

    # model config
    logging.info("Start training")
    logging.info(f"Using model definition: {model}")
    model_cfg = OmegaConf.load(model)
    logging.info("--- START model.yaml ---")
    model_yaml = OmegaConf.to_yaml(model_cfg)
    logging.info(model_yaml)
    logging.info("--- END model.yaml ---")

    # train config
    if not no_default_config:
        logging.info(f"Using default training configuration: {_default_config}")
        train_cfg = OmegaConf.load(_default_config)
    else:
        train_cfg = OmegaConf.create()

    for cfg in config:
        logging.info(f"Using configuration: {cfg}")
        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.load(cfg))

    if args:
        logging.info("Overriding configuration:")
        for arg in args:
            logging.info(arg)
        args_cfg = OmegaConf.from_dotlist(args)
        train_cfg = OmegaConf.merge(train_cfg, args_cfg)
    logging.info("--- START train.yaml ---")
    train_cfg = OmegaConf.to_yaml(train_cfg)
    logging.info(train_cfg)
    logging.info("--- END train.yaml ---")

    # try load model and pring its configuration
    logging.info("Building model")
    model = utils.build_model(model_cfg)
    logging.info(model)

    # launch
    num_gpus = torch.cuda.device_count()
    logging.info(f"Start training using {num_gpus} GPU(s):")
    for i in range(num_gpus):
        logging.info(torch.cuda.get_device_name(i))
    if num_gpus == 0:
        logging.warning("No GPU available. Training will run on CPU. Use for testing only.")
    if num_gpus > 1:
        logging.info("Using DDP training.")
        from ignite import distributed as idist

        with idist.Parallel(backend="nccl", nproc_per_node=num_gpus) as parallel:  # type: ignore[attr-defined]
            parallel.run(run, num_gpus, model_cfg, train_cfg, load, save)
    else:
        run(0, 1, model_cfg, train_cfg, load, save)


def run(local_rank, world_size, model_cfg, train_cfg, load, save):
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # load configs
    model_cfg = OmegaConf.create(model_cfg)
    if not isinstance(model_cfg, omegaconf.DictConfig):
        raise TypeError("Model configuration must be a dictionary.")
    train_cfg = OmegaConf.create(train_cfg)
    if not isinstance(train_cfg, omegaconf.DictConfig):
        raise TypeError("Train configuration must be a dictionary.")

    # build model
    _force_training = "forces" in train_cfg.data.y
    model = utils.build_model(model_cfg, forces=_force_training)
    if world_size > 1:
        from ignite import distributed as idist

        model = idist.auto_model(model)  # type: ignore[attr-defined]
    elif torch.cuda.is_available():
        model = model.cuda()  # type: ignore

    # load weights
    if load is not None:
        device = next(model.parameters()).device  # type: ignore[attr-defined]
        logging.info(f"Loading weights from file {load}")
        sd = torch.load(load, map_location=device)
        logging.info(utils.unwrap_module(model).load_state_dict(sd, strict=False))

    # data loaders
    train_loader, val_loader = utils.get_loaders(train_cfg.data)

    # optimizer, scheduler, etc
    model = utils.set_trainable_parameters(
        model,  # type: ignore[attr-defined]
        train_cfg.optimizer.force_train,
        train_cfg.optimizer.force_no_train,
    )
    optimizer = utils.get_optimizer(model, train_cfg.optimizer)
    if world_size > 1:
        optimizer = idist.auto_optim(optimizer)  # type: ignore[attr-defined]
    scheduler = utils.get_scheduler(optimizer, train_cfg.scheduler) if train_cfg.scheduler is not None else None  # type: ignore[attr-defined]
    loss = utils.get_loss(train_cfg.loss)

    metrics = utils.get_metrics(train_cfg.metrics)
    metrics.attach_loss(loss)  # type: ignore[attr-defined]

    # ignite engine
    trainer, validator = utils.build_engine(model, optimizer, scheduler, loss, metrics, train_cfg, val_loader)

    if local_rank == 0 and train_cfg.wandb is not None:
        utils.setup_wandb(train_cfg, model_cfg, model, trainer, validator, optimizer)

    trainer.run(train_loader, max_epochs=train_cfg.trainer.epochs)

    if local_rank == 0 and save is not None:
        logging.info(f"Saving model weights to file {save}")
        torch.save(utils.unwrap_module(model).state_dict(), save)


if __name__ == "__main__":
    train()
