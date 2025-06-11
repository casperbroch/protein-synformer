import os, sys, json, warnings, multiprocessing as mp
import click, optuna, pytorch_lightning as pl, torch, wandb
from pytorch_lightning import callbacks, loggers, strategies
from omegaconf import OmegaConf

# Project-specific imports
from synformer.data.projection_dataset_new import ProjectionDataModule
from synformer.models.wrapper import SynformerWrapper
from synformer.utils.misc import (
    get_config_name, get_experiment_name, get_experiment_version, n_params,
)
from synformer.utils.vc import get_vc_info

# Enable performance optimizations for matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True
torch.set_float32_matmul_precision("medium")


# Optuna pruning callback
class OptunaPruningCallback(pl.callbacks.Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val/loss"):
        self.trial, self.monitor = trial, monitor

    def state_dict(self):            return {}
    def load_state_dict(self, _):    pass

    # Called at the end of each validation loop
    def on_validation_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        self.trial.report(float(metric), step=trainer.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# Runs a single PyTorch Lightning training trial
def pl_one_run(
    config_path: str,
    overrides: list[str],
    trial: optuna.Trial | None = None,
    **cli,
) -> float:
    # Load and merge configuration
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    print(cfg)

    # Set seed for reproducibility
    seed = cli.get("seed", 42) + (trial.number if trial else 0)
    pl.seed_everything(seed, workers=True)

    # Ensure batch size is divisible by number of devices
    if cli["batch_size"] % cli["devices"] != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    bs_per_proc = cli["batch_size"] // cli["devices"]

    # Initialize dataloader module
    dm = ProjectionDataModule(
        cfg,
        batch_size=bs_per_proc,
        num_workers=cli["num_workers"],
        **cfg.data,
    )

    # Generate experiment metadata
    cfg_name = get_config_name(config_path)
    vc_info  = get_vc_info()
    exp_name = get_experiment_name(cfg_name, vc_info.display_version, vc_info.committed_at)
    exp_ver  = get_experiment_version()

    # Update training config
    cfg.train.exp_name = exp_name
    cfg.train.exp_ver  = exp_ver
    cfg.train.seed     = seed
    cfg.train.batch_size   = cli["batch_size"]
    cfg.train.num_workers  = cli["num_workers"]
    cfg.train.devices      = cli["devices"]
    cfg.train.num_nodes    = cli["num_nodes"]

    # Initialize wandb logging if config exists
    if not os.path.exists("configs/wandb.yml"):
        warnings.warn("No wandb config found! Not using wandb.")
        wandb_run = None
    else:
        wandb_cfg = OmegaConf.load("configs/wandb.yml")
        if wandb_cfg.enabled:
            os.environ["WANDB_API_KEY"] = wandb_cfg["api_key"]
            run_name = f"{exp_name}-trial{trial.number if trial else '0'}"
            wandb_run = wandb.init(
                project=cfg.project.name,
                entity=wandb_cfg.entity,
                sync_tensorboard=True,
                config=OmegaConf.to_container(cfg),
                name=run_name,
                reinit=True,
                save_code=False,
            )
            wandb_run.config.update(trial.params if trial else {}, allow_val_change=True)
        else:
            wandb_run = None

    # Create model wrapper
    model = SynformerWrapper(
        config=cfg,
        args={
            "config_path": config_path,
            "seed": seed,
            "batch_size": cli["batch_size"],
            "num_workers": cli["num_workers"],
            "devices": cli["devices"],
            "resume": cli.get("resume"),
        },
    )

    # Load model weights if resuming from checkpoint
    if cli.get("resume"):
        ckpt_path = cli["resume"]
        print("Resuming decoder-/head-weights from checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu" if cfg.system.device == "cpu" else None)
        sd = ckpt["state_dict"]
        model.load_state_dict(
            {k: v for k, v in sd.items() if any([
                k.startswith("model.decoder."),
                k.startswith("model.fingerprint_head."),
                k.startswith("model.token_head."),
                k.startswith("model.reaction_head."),
            ])},
            strict=False,
        )

    # If using LoRA: freeze base model parameters
    if cfg.model.decoder.lora:
        for n, p in model.named_parameters():
            if "lora" not in n and any([
                n.startswith("model.decoder"),
                n.startswith("model.fingerprint_head"),
                n.startswith("model.token_head"),
                n.startswith("model.reaction_head"),
            ]):
                p.requires_grad = False
        print("Parameters: model:\t\t", n_params(model))
        print("Trainable parameters: model:\t", n_params(model, only_trainable=True))
        print("Trainable parameters: lora_dec:\t",
              n_params(model.model.decoder.lora_dec, only_trainable=True))

    # Define callbacks
    ckpt_cb  = callbacks.ModelCheckpoint(save_last=True, monitor="val/loss",
                                         mode="min", save_top_k=5)
    lr_cb    = callbacks.LearningRateMonitor(logging_interval="step")
    prune_cb = OptunaPruningCallback(trial, "val/loss") if trial else None
    cb_list  = [ckpt_cb, lr_cb] + ([prune_cb] if prune_cb else [])

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator=cfg.system.device,
        devices=cli["devices"],
        num_nodes=cli["num_nodes"],
        strategy=(strategies.DDPStrategy(static_graph=True, process_group_backend="gloo")
                  if cli["devices"] > 1 else "auto"),
        num_sanity_val_steps=cli["num_sanity_val_steps"],
        gradient_clip_val=cfg.train.max_grad_norm,
        log_every_n_steps=1,
        #max_steps=cfg.train.max_iters,
        callbacks=cb_list,
        logger=[loggers.TensorBoardLogger(cli["log_dir"], name=exp_name, version=exp_ver)],        
        limit_val_batches=4,
        
        max_epochs=cfg.train.max_epochs,
        val_check_interval=cfg.train.val_check_interval,
    )

    # Start training
    trainer.fit(model, datamodule=dm)

    # Finalize wandb if used
    if wandb_run is not None:
        wandb_run.finish()

    # Return final validation loss
    return trainer.callback_metrics["val/loss"].item()

# Optuna objective function for one trial
def objective(trial: optuna.Trial, config_path: str, **cli) -> float:
    # Suggest hyperparameters to try
    lr         = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    wd         = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.001)
    lora_rank  = trial.suggest_categorical("lora_rank", [8, 16, 32, 64, 128])

    factor     = trial.suggest_float("scheduler_factor", 0.1, 0.9)
    patience   = trial.suggest_int("scheduler_patience", 1, 20)
    min_lr     = trial.suggest_float("scheduler_min_lr", 1e-7, 1e-4, log=True)

    # Apply overrides based on suggested values
    overrides = [
        f"train.optimizer.lr={lr}",
        f"train.optimizer.weight_decay={wd}",
        f"model.decoder.lora_rank={lora_rank}",
        f"train.scheduler.factor={factor}",
        f"train.scheduler.patience={patience}",
        f"train.scheduler.min_lr={min_lr}",
    ]

    # Run training with these overrides
    return pl_one_run(config_path, overrides, trial=trial, **cli)


# CLI entry point using click
@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--n-trials", type=int, default=20)
@click.option("--seed", type=int, default=42)
@click.option("--batch-size", "-b", type=int, default=196)
@click.option("--num-workers", type=int, default=8)
@click.option("--devices", type=int, default=1)
@click.option("--num-nodes", type=int, default=1)
@click.option("--num-sanity-val-steps", type=int, default=0)
@click.option("--log-dir", default="./logs")
@click.option("--resume", type=str, default=None)
@click.option("--n-jobs", type=int, default=1,
              help="Parallel Optuna trials – set CUDA_VISIBLE_DEVICES accordingly.")
def main(**cli):
    # Ensure multiprocessing works properly
    mp.set_start_method("spawn", force=True)

    # Create a new Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    # Run the optimization process
    study.optimize(
        lambda t: objective(t, **cli),
        n_trials=cli.pop("n_trials"),
        n_jobs=cli.pop("n_jobs"),
        show_progress_bar=True,
    )

    # Print best result
    best = study.best_trial
    print("\nBest val/loss =", best.value)
    print(json.dumps(best.params, indent=2))

# Entry point
if __name__ == "__main__":
    main()
