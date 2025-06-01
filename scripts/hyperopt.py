import os, sys, json, datetime, multiprocessing as mp
import click, optuna, pytorch_lightning as pl, torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies
from optuna.integration import PyTorchLightningPruningCallback

from synformer.data.projection_dataset_new import ProjectionDataModule
from synformer.chem.mol import FingerprintOption
from synformer.models.wrapper import SynformerWrapper
from synformer.utils.misc import (
    get_config_name, get_experiment_name, get_experiment_version,
)
from synformer.utils.vc import get_vc_info

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


#  Objective: one complete Lightning run
def pl_one_run(cfg_path: str,
               overrides: list[str],
               trial: optuna.Trial | None = None,
               **cli_kwargs) -> float:
    """Train once and return best validation loss."""

    # 1) load + override YAML
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # 2) Lightning-style seeding (each trial gets unique seed)
    seed = cli_kwargs.get("seed", 42) + (trial.number if trial else 0)
    pl.seed_everything(seed, workers=True)

    # 3) Data & model
    dm = ProjectionDataModule(
        cfg,
        batch_size=cli_kwargs["batch_size"] // cli_kwargs["devices"],
        num_workers=cli_kwargs["num_workers"],
        fp_option=FingerprintOption(**cfg.chem.fp_option),
        **cfg.data,
    )
    model = SynformerWrapper(config=cfg, args=cli_kwargs)

    # 4) Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        save_last=True, monitor="val/loss", mode="min", save_top_k=3)
    lr_monitor_cb = callbacks.LearningRateMonitor(logging_interval="step")
    pl_prune_cb = (PyTorchLightningPruningCallback(trial, "val/loss")
                   if trial else None)
    cbs = [checkpoint_cb, lr_monitor_cb]
    if pl_prune_cb:
        cbs.append(pl_prune_cb)

    # 5) Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cli_kwargs["devices"],
        num_nodes=cli_kwargs["num_nodes"],
        strategy=strategies.DDPStrategy(static_graph=True),
        gradient_clip_val=cfg.train.max_grad_norm,
        log_every_n_steps=1,
        max_steps=cfg.train.max_iters,
        num_sanity_val_steps=cli_kwargs["num_sanity_val_steps"],
        val_check_interval=cfg.train.val_freq,
        limit_val_batches=4,
        callbacks=cbs,
        logger=loggers.TensorBoardLogger(
            cli_kwargs["log_dir"],
            name=get_experiment_name(
                get_config_name(cfg_path),
                get_vc_info().display_version,
                get_vc_info().committed_at),
            version=get_experiment_version()),
    )
    trainer.fit(model, dm, ckpt_path=cli_kwargs.get("resume", None))

    # 6) Return metric Optuna will minimise
    return trainer.callback_metrics["val/loss"].item()


#  Optuna objective
def objective(trial: optuna.Trial, cfg_path: str, **cli_kwargs) -> float:
    # Suggest hyper-parameters
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)            # log-scale:contentReference[oaicite:3]{index=3}
    wd = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32, 64, 128])

    overrides = [
        f"train.optimizer.lr={lr}",
        f"train.optimizer.weight_decay={wd}",
        f"model.decoder.lora_rank={lora_rank}",
    ]

    return pl_one_run(cfg_path, overrides, trial=trial, **cli_kwargs)


#  CLI wrapper
@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n-trials", type=int, default=20)
@click.option("--storage", default="sqlite:///optuna.db",
              help="Optuna storage URL.")
@click.option("--seed", type=int, default=42)
@click.option("--batch-size", "-b", type=int, default=196)
@click.option("--num-workers", type=int, default=8)
@click.option("--devices", type=int, default=1)
@click.option("--num-nodes", type=int, default=1)
@click.option("--num-sanity-val-steps", type=int, default=0)
@click.option("--log-dir", default="./logs")
@click.option("--resume", type=str, default=None)
@click.option("--n-jobs", type=int, default=1,
              help="Parallel Optuna trials. "
                   "Set CUDA_VISIBLE_DEVICES accordingly.")
def main(**cli_kwargs):
    mp.set_start_method("spawn", force=True)

    study = optuna.create_study(direction="minimize",
                                storage=cli_kwargs.pop("storage"),
                                pruner=optuna.pruners.MedianPruner(
                                    n_warmup_steps=2))  # quick pruning
    study.optimize(
        lambda t: objective(t, **cli_kwargs),
        n_trials=cli_kwargs.pop("n_trials"),
        n_jobs=cli_kwargs.pop("n_jobs"),
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\nBest val/loss =", best.value)
    print(json.dumps(best.params, indent=2))

    # Hint for dashboard users
    dash_cmd = f"optuna-dashboard {study._storage.url}"
    print(f"\nLaunch dashboard:\n    {dash_cmd}\n")


if __name__ == "__main__":
    main()
