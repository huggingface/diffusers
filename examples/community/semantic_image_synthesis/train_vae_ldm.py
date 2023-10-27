# This implementation is based on this one
# https://github.com/Pie31415/diffusers/blob/vae-training/examples/vae/train_vae.py
import argparse
import math
import os


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader,Subset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils.import_utils import is_bitsandbytes_available

from tqdm.auto import tqdm
from typing import List
import shutil

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from sis_dataset import CELEBAHQ_DICT, SISDataset
from src.models import AutoencoderKL,AutoencoderSIS

from datetime import datetime
import lpips

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")



parser = argparse.ArgumentParser(description="VAE training script.")
# Data Management
parser.add_argument("--dataset_img_dir", type=str, default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/img/")
parser.add_argument("--dataset_ann_dir", type=str, default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/mask/")
parser.add_argument("--dataset_img_size", type=int, default=128)
parser.add_argument("--output_dir",type=str,default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/output/vae",
    help="The output directory where the model predictions and checkpoints will be written.",
)
# Training Management
parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--max_train_steps", type=int, default=10)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--scale_lr",
    action="store_true",
    default=False,
    help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
parser.add_argument(
    "--lr_warmup_steps",
    type=int,
    default=500,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
parser.add_argument("--optim_weight_decay", type=float, default=1e-2, help="optimizer weight decay like in paper")
parser.add_argument("--optim_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="fp16",
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
)
parser.add_argument("--lambda_kl",type=float,default=1e-6,
    help="Scaling factor for the Kullback-Leibler divergence penalty term.",
)
parser.add_argument(
    "--lambda_lpips",
    type=float,
    default=1e-1,
    help="Scaling factor for the LPIPS metric",
)
# Validation Management
parser.add_argument("--val_every_nepochs", type=int, default=1, help="Number of training epochs before validation...")
parser.add_argument("--val_num_samples", type=int, default=10)
parser.add_argument(
    "--tracker_name",
    type=str,
    default=["mlflow", "wandb"],
    help=(
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
        ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    ),
)
parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Indicate the checkpoint frequency")
parser.add_argument(
    "--checkpointing_total_limit", type=int, default=10, help="Indicate the max number of checkpoints to keep"
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default="latest",
    help="Indicate either the checkpoint directory or the latest",
)
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
parser.add_argument("--debug", action="store_true", default=False)

def main(
        dataset_img_dir:str,
        dataset_ann_dir:str,
        dataset_img_size:str,
        output_dir:str,
        train_batch_size:int,
        max_train_steps:int,
        gradient_accumulation_steps:int,
        gradient_checkpointing:int,
        learning_rate:float,
        scale_lr:bool,
        lr_scheduler:str,
        lr_warmup_steps:int,
        use_8bit_adam:bool,
        optim_weight_decay:float,
        optim_max_grad_norm:float,
        mixed_precision:str,
        lambda_kl:float,
        lambda_lpips:float,
        val_every_nepochs:int,
        val_num_samples:int,
        tracker_name:List[str],
        checkpointing_steps:int,
        checkpointing_total_limit:int,
        resume_from_checkpoint:str,
        seed:int,
        debug: bool = False,
        all_args:dict=None
    ):
    if debug:
        NMAX = 100
    else:
        NMAX = None
    # We create the logging directory...
    logdir_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-SISModel"
    logdir_path = os.path.join(output_dir, logdir_name)
    project_dir = output_dir
    if resume_from_checkpoint.upper() == "NONE":
        # We won't from checkpoint then we'll create the logdir to save the model...
        project_dir = logdir_path
    print(f"Project Directory : {project_dir}")
    # Configure Accelerate
    accelerator_project_configuration = ProjectConfiguration(project_dir=project_dir, logging_dir=logdir_path)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=tracker_name,
        project_config=accelerator_project_configuration,
    )
    if accelerator.scaler is not None:
        accelerator.scaler.set_growth_interval(100)
    if accelerator.is_main_process:
        os.makedirs(logdir_path, exist_ok=True)

    # Manage Seed
    set_seed(seed)
    # We create a Dataset and Dataloaders
    train_dataset = SISDataset(
        dataset_img_dir, dataset_ann_dir, img_size=dataset_img_size, cls_dict=CELEBAHQ_DICT, nmax=NMAX
    )
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    train_indices = indices[val_num_samples:]
    val_indices = indices[:val_num_samples]

    train_dataloader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), train_batch_size),
    )
    val_dataloader = DataLoader(
        Subset(train_dataset, val_indices),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    # Load Model
    input_channels = 3
    config = AutoencoderSIS.get_config(
        sample_size = dataset_img_size,
        latent_size = 64,
        in_channels=input_channels,
        out_channels=input_channels)
    vae = AutoencoderSIS(**config)
    vae:AutoencoderSIS
    vae.requires_grad_(True)

    if gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )
    if is_bitsandbytes_available() and use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(vae.parameters(), lr=learning_rate,weight_decay=optim_weight_decay)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.


    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Prepare everything with our `accelerator`.
    (
        vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vae, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("train-vae-CelebA", all_args)

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
        train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_indices)}")
    logger.info(f"  Num test samples = {len(val_indices)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)

    for epoch in range(first_epoch, num_train_epochs):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                x, _, _ = batch
                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                # We use a "trick" in order to use DDP that needs a forward method.
                posterior = vae.forward(x,encode=True).latent_dist
                z = posterior.mode()
                pred = vae.forward(z,decode=True).sample

                kl_loss = posterior.kl().mean()
                mse_loss = F.mse_loss(pred, x, reduction="mean")
                lpips_loss = lpips_loss_fn(pred, x).mean()

                loss = (
                    mse_loss + lambda_lpips * lpips_loss + lambda_kl * kl_loss
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), optim_max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if checkpointing_total_limit is not None:
                            checkpoints = os.listdir(project_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= checkpointing_total_limit:
                                num_to_remove = len(checkpoints) - checkpointing_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(project_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(project_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= max_train_steps:
                break
        
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse": mse_loss.detach().item(),
                "lpips": lpips_loss.detach().item(),
                "kl": kl_loss.detach().item(),
            }
            accelerator.log(logs)
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            if epoch % val_every_nepochs == 0:
                with torch.no_grad():
                    logger.info("Running validation... ")
                    vae_model = accelerator.unwrap_model(vae)
                    images = []
                    image_ids = []
                    for _, batch in enumerate(val_dataloader):
                        x, _, ids = batch
                        reconstructions = vae_model(x).sample
                        images.append(
                            torch.cat([x.cpu(), reconstructions.cpu()], axis=-1).squeeze(0) # Last dim = Width
                        )
                        image_ids.append(ids)

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images(
                                "Original (left) / Reconstruction (right)", np_images, epoch
                            )
                        elif tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {image_ids[i]}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )
                        else:
                            logger.warn(f"image logging not implemented for {tracker.name}")
                    del vae_model
                    torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['all_args']=args_dict.copy()
    main(**args_dict)