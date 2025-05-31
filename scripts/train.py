import os
import warnings

import click
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import callbacks, loggers, strategies
import wandb

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
@click.option("--devices", type=int, default=1)  # default=4
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
    if batch_size % devices != 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    batch_size_per_process = batch_size // devices

    os.makedirs(log_dir, exist_ok=True)
    pl.seed_everything(seed)

    config = OmegaConf.load(config_path)
    config_name = get_config_name(config_path)
    vc_info = get_vc_info()
    vc_info.disallow_changes(debug)
    exp_name = get_experiment_name(config_name, vc_info.display_version, vc_info.committed_at)
    exp_ver = get_experiment_version()

    if config.system.device == "gpu" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    # Add flags (click options) to config for convenience
    config.train.exp_name = exp_name
    config.train.exp_ver = exp_ver
    config.train.seed = seed
    config.train.debug = debug
    config.train.batch_size = batch_size
    config.train.num_workers = num_workers
    config.train.devices = devices
    config.train.num_nodes = num_nodes

    # Monitoring
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
                save_code=False,
                # resume=True
            )

    # Dataloaders
    datamodule = ProjectionDataModule(
        config,
        batch_size=batch_size_per_process,
        num_workers=num_workers,
        **config.data,
    )
    # print(datamodule)

    # Model
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
    # print(model)

    if resume:
        # Assuming: only resuming the decoder and decoder heads
        print("Resuming from checkpoint:", resume)
        if config.system.device == "cpu":
            ckpt = torch.load(resume, map_location="cpu")
        else:
            ckpt = torch.load(resume)
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
        model.load_state_dict(
            filtered_state_dict, 
            strict=False
        )
    
    if config.model.decoder.lora:
        # It might already be doing this automatically, but to be sure:
        # Freezing: non-LoRA decoder, decoder heads 
        # Not freezing: LoRA, encoder 
        for name, param in model.named_parameters():
            if "lora" not in name and any([
                name.startswith("model.decoder"),
                name.startswith("model.fingerprint_head"),
                name.startswith("model.token_head"), 
                name.startswith("model.reaction_head"),
            ]):
                param.requires_grad = False
        print("Parameters: model:\t\t", 
              n_params(model))  # entire model 
        print("Trainable parameters: model:\t", 
              n_params(model, only_trainable=True))  # lora + encoder
        print("Trainable parameters: lora_dec:\t", 
              n_params(model.model.decoder.lora_dec, only_trainable=True))  # only lora 
    import sys; sys.exit(0)

    # Train
    trainer = pl.Trainer(
        accelerator=config.system.device, 
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategies.DDPStrategy(static_graph=True, process_group_backend="gloo") if devices > 1 else "auto",
        num_sanity_val_steps=num_sanity_val_steps,
        gradient_clip_val=config.train.max_grad_norm,
        log_every_n_steps=1,
        max_steps=config.train.max_iters,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True, monitor="val/loss", mode="min", save_top_k=5),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=[
            loggers.TensorBoardLogger(log_dir, name=exp_name, version=exp_ver),
        ],
        val_check_interval=config.train.val_freq,
        limit_val_batches=4,
    )
    trainer.fit(
        model, 
        datamodule=datamodule
    )
    print("Finished training")

    '''
    if config.model.decoder.lora:
        # Save LoRA weights 
        #
        #
        #
        # Merge LoRA weights into the original model weights
        print("Merging...")
        merged_decoder = model.model.decoder.lora_dec.merge_lora() 
        # How to save inside of model? and delete lora_dec?
        ...
    '''


if __name__ == "__main__":
    main()
