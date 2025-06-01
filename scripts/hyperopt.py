import os, sys, json, datetime, multiprocessing as mp
import click, optuna

import pytorch_lightning as pl
from pytorch_lightning import callbacks, loggers, strategies

import torch
from omegaconf import OmegaConf

from synformer.data.projection_dataset_new import ProjectionDataModule
from synformer.models.wrapper import SynformerWrapper
from synformer.utils.misc import (
    get_config_name,
    get_experiment_name,
    get_experiment_version,
)
from synformer.utils.vc import get_vc_info
from synformer.utils.misc import n_params

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class OptunaPruningCallback(pl.callbacks.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val/loss"):
        self.trial = trial
        self.monitor = monitor

    def state_dict(self):  
        return {}
    def load_state_dict(self, state):
        pass

    def on_validation_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        self.trial.report(float(metric), step=trainer.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# 1 Optuna-trial = 1 complete training run
def pl_one_run(
    config_path: str,
    overrides: list[str],
    trial: optuna.Trial | None = None,
    **cli_kwargs
) -> float:

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    seed = cli_kwargs.get("seed", 42) + (trial.number if trial else 0)
    pl.seed_everything(seed, workers=True)

    if cli_kwargs["batch_size"] % cli_kwargs["devices"] != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size_per_process = cli_kwargs["batch_size"] // cli_kwargs["devices"]

    dm = ProjectionDataModule(
        cfg,
        batch_size=batch_size_per_process,
        num_workers=cli_kwargs["num_workers"],
        **cfg.data,
    )

    cfg_name = get_config_name(config_path)
    vc_info = get_vc_info()
    exp_name = get_experiment_name(cfg_name, vc_info.display_version, vc_info.committed_at)
    exp_ver  = get_experiment_version()

    cfg.train.exp_name = exp_name
    cfg.train.exp_ver = exp_ver
    cfg.train.seed = seed
    cfg.train.batch_size = cli_kwargs["batch_size"]
    cfg.train.num_workers = cli_kwargs["num_workers"]
    cfg.train.devices = cli_kwargs["devices"]
    cfg.train.num_nodes = cli_kwargs["num_nodes"]

    model = SynformerWrapper(
        config=cfg,
        args={
            "config_path": config_path,
            "seed": seed,
            "batch_size": cli_kwargs["batch_size"],
            "num_workers": cli_kwargs["num_workers"],
            "devices": cli_kwargs["devices"],
            "resume": cli_kwargs.get("resume"),
        },
    )

    if cli_kwargs.get("resume"):
        ckpt_path = cli_kwargs["resume"]
        print("Resuming decoder-/head-weights from checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu" if cfg.system.device == "cpu" else None)
        state_dict = ckpt["state_dict"]
        filtered = {
            k: v for k, v in state_dict.items()
            if any([
                k.startswith("model.decoder."),
                k.startswith("model.fingerprint_head."),
                k.startswith("model.token_head."),
                k.startswith("model.reaction_head."),
            ])
        }
        model.load_state_dict(filtered, strict=False)

    if cfg.model.decoder.lora:
        for name, param in model.named_parameters():
            if "lora" not in name and any([
                name.startswith("model.decoder"),
                name.startswith("model.fingerprint_head"),
                name.startswith("model.token_head"),
                name.startswith("model.reaction_head"),
            ]):
                param.requires_grad = False
        print("Parameters: model:\t\t", n_params(model))
        print("Trainable parameters: model:\t", n_params(model, only_trainable=True))
        print("Trainable parameters: lora_dec:\t",
              n_params(model.model.decoder.lora_dec, only_trainable=True))

    checkpoint_cb = callbacks.ModelCheckpoint(
        save_last=True, monitor="val/loss", mode="min", save_top_k=5
    )
    lr_monitor_cb = callbacks.LearningRateMonitor(logging_interval="step")
    prune_cb = OptunaPruningCallback(trial, "val/loss") if trial else None

    cb_list = [checkpoint_cb, lr_monitor_cb]
    if prune_cb:
        cb_list.append(prune_cb)

    trainer = pl.Trainer(
        accelerator=cfg.system.device,
        devices=cli_kwargs["devices"],
        num_nodes=cli_kwargs["num_nodes"],
        strategy=(strategies.DDPStrategy(static_graph=True, process_group_backend="gloo")
                  if cli_kwargs["devices"] > 1 else "auto"),
        num_sanity_val_steps=cli_kwargs["num_sanity_val_steps"],
        gradient_clip_val=cfg.train.max_grad_norm,
        log_every_n_steps=1,
        max_steps=cfg.train.max_iters,
        callbacks=cb_list,
        logger=[
            loggers.TensorBoardLogger(cli_kwargs["log_dir"], name=exp_name, version=exp_ver),
        ],
        val_check_interval=cfg.train.val_freq,
        limit_val_batches=4,
    )

    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics["val/loss"].item()


def objective(trial: optuna.Trial, config_path: str, **cli_kwargs) -> float:
    lr         = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    wd         = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    lora_rank  = trial.suggest_categorical("lora_rank", [8, 16, 32, 64, 128])

    overrides = [
        f"train.optimizer.lr={lr}",
        f"train.optimizer.weight_decay={wd}",
        f"model.decoder.lora_rank={lora_rank}",
    ]

    return pl_one_run(config_path, overrides, trial=trial, **cli_kwargs)


# CLI 
@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n-trials", type=int, default=20)
@click.option("--storage", default="sqlite:///optuna.db")
@click.option("--seed", type=int, default=42)
@click.option("--batch-size", "-b", type=int, default=196)
@click.option("--num-workers", type=int, default=8)
@click.option("--devices", type=int, default=1)
@click.option("--num-nodes", type=int, default=1)
@click.option("--num-sanity-val-steps", type=int, default=0)
@click.option("--log-dir", default="./logs")
@click.option("--resume", type=str, default=None)
@click.option("--n-jobs", type=int, default=1,
              help="Parallel Optuna trials.")
def main(**cli_kwargs):
    mp.set_start_method("spawn", force=True)

    study = optuna.create_study(
        direction="minimize",
        storage=cli_kwargs.pop("storage"),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    study.optimize(
        lambda t: objective(t, **cli_kwargs),
        n_trials=cli_kwargs.pop("n_trials"),
        n_jobs=cli_kwargs.pop("n_jobs"),
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\nBest val/loss =", best.value)
    print(json.dumps(best.params, indent=2))
    print(f"\nLaunch dashboard:\n    optuna-dashboard {study._storage.url}\n")


if __name__ == "__main__":
    main()
