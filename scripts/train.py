import os
import warnings

import click
import pytorch_lightning as pl
import torch
from torch import nn 
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies
import wandb

# Project-specific modules
from synformer.data.projection_dataset_new import ProjectionDataModule  
from synformer.models.wrapper import SynformerWrapper
from synformer.utils.misc import (
    get_config_name,
    get_experiment_name,
    get_experiment_version,
)
from synformer.utils.vc import get_vc_info
from synformer.utils.misc import n_params


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--seed", type=int, default=42)
@click.option("--debug", is_flag=True)
@click.option("--batch-size", "-b", type=int, default=196)
@click.option("--num-workers", type=int, default=8)
@click.option("--devices", type=int, default=1)
@click.option("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", 1)))
@click.option("--num-sanity-val-steps", type=int, default=1)
@click.option("--log-dir", type=click.Path(dir_okay=True, file_okay=False), default="./logs")
@click.option("--resume", type=click.Path(exists=True, dir_okay=False), default=None)
def main(
    config_path: str,
    seed: int,
    debug: bool,
    batch_size: int,
    num_workers: int,
    devices: int,
    num_nodes: int,
    num_sanity_val_steps: int,
    log_dir: str,
    resume: str | None,
):
    # Ensure batch size is divisible across devices
    if batch_size % devices != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size_per_process = batch_size // devices

    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    # Load config and experiment metadata
    config = OmegaConf.load(config_path)
    config_name = get_config_name(config_path)
    vc_info = get_vc_info()
    vc_info.disallow_changes(debug)
    exp_name = get_experiment_name(config_name, vc_info.display_version, vc_info.committed_at)
    exp_ver = get_experiment_version()

    # Enable performance options if using GPU
    if config.system.device in ("gpu", "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    # Inject CLI values into config
    config.train.exp_name = exp_name
    config.train.exp_ver = exp_ver
    config.train.seed = seed
    config.train.debug = debug
    config.train.batch_size = batch_size
    config.train.num_workers = num_workers
    config.train.devices = devices
    config.train.num_nodes = num_nodes

    # Initialize WandB logging if enabled
    if not os.path.exists("configs/wandb.yml"):
        warnings.warn("No wandb config found! Not using wandb.")
    else:
        wandb_config = OmegaConf.load("configs/wandb.yml")
        if wandb_config.enabled:
            os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
            wandb.init(
                project=config.project.name,
                entity=wandb_config.entity,
                sync_tensorboard=True,
                config=OmegaConf.to_container(config), 
                name=config.train.exp_name, 
                save_code=False
            )

    # Initialize dataloader
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size_per_process,
        num_workers=num_workers,
        **config.data,
    )

    # Initialize model
    model = SynformerWrapper(
        config=config,
        args={
            "config_path": config_path,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
            "resume": resume,
        },
    )

    # Resume checkpoint if specified
    if resume:
        print("Resuming from checkpoint:", resume)
        ckpt = torch.load(resume, map_location="cpu" if config.system.device == "cpu" else None)
        state_dict = ckpt["state_dict"]
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if any([
                k.startswith("model.decoder."), 
                k.startswith("model.fingerprint_head."),
                k.startswith("model.token_head."),
                k.startswith("model.reaction_head."),
            ])
        }
        model.load_state_dict(filtered_state_dict, strict=False)

        # Reinitialize cross-attention layers if needed
        if OmegaConf.select(config, "model.decoder.reinit"):
            for name, module in model.named_modules():
                if name.endswith("multihead_attn.module") and hasattr(module, "reset_parameters"):
                    print("Re-initializing", name)
                    module.reset_parameters()

    # Handle LoRA fine-tuning if enabled
    if OmegaConf.select(config, "model.decoder.lora"):
        assert resume, "LoRA requires resuming from a checkpoint."
        assert not OmegaConf.select(config, "model.decoder.last_n_layers"), \
            "LoRA cannot be combined with last_n_layers."
        assert OmegaConf.select(config, "model.decoder.lora_rank"), \
            "Missing LoRA rank in config."

        # Freeze all non-LoRA parameters in decoder and heads
        for name, param in model.named_parameters():
            if "lora" not in name and "multihead_attn" not in name and any([
                name.startswith("model.decoder"),
                name.startswith("model.fingerprint_head"),
                name.startswith("model.token_head"), 
                name.startswith("model.reaction_head"),
            ]):
                param.requires_grad = False

        print(n_params(model.model.decoder.lora_dec, only_trainable=True),
              "\t" + "Trainable parameters: lora_dec")

    # Handle training of only the last N decoder layers
    if OmegaConf.select(config, "model.decoder.last_n_layers"):
        assert resume, "last_n_layers requires resuming from a checkpoint."
        assert not OmegaConf.select(config, "model.decoder.lora"), \
            "Cannot combine last_n_layers with LoRA."
        assert OmegaConf.select(config, "model.decoder.num_trainable_layers"), \
            "Missing num_trainable_layers in config."

        num_trainable_layers = OmegaConf.select(config, "model.decoder.num_trainable_layers")
        print(f"Only training last {num_trainable_layers} layers (cross-attn stays trainable)")

        num_decoder_layers = len(list(set(
            int(name.split(".")[4]) 
            for name, _ in model.named_parameters()
            if name.startswith("model.decoder.dec.layers")
        )))
        print("Found", num_decoder_layers, "decoder layers")

        for name, param in model.named_parameters():
            if name.startswith("model.decoder.dec.layers"):
                layer_idx = int(name.split(".")[4])
                if layer_idx < num_decoder_layers - num_trainable_layers and "multihead_attn" not in name:
                    print("Freeze", (layer_idx, name))
                    param.requires_grad = False
                else:
                    print("Train ", (layer_idx, name))

    # Print parameter counts
    print(n_params(model), "\t" + "Parameters: model")  
    print(n_params(model, only_trainable=True), "\t" + "Trainable parameters: model") 

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator=config.system.device, 
        devices=devices,
        num_nodes=num_nodes,
        strategy=(
            strategies.DDPStrategy(static_graph=True, process_group_backend="gloo") 
            if devices > 1 
            else "auto"
        ),
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=config.train.max_grad_norm,
        log_every_n_steps=1,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True, monitor="val/loss", mode="min", save_top_k=5),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=[
            loggers.TensorBoardLogger(log_dir, name=exp_name, version=exp_ver),
        ],
        max_epochs=config.train.max_epochs,
        val_check_interval=config.train.val_check_interval,
        limit_val_batches=4,
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)
    print("Finished training")


if __name__ == "__main__":
    main()
