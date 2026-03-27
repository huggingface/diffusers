#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from diffusers.models.autoencoders import AutoencoderULEncoder
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    ul_alpha_sigma_from_logsnr,
    ul_decoder_loss_weight,
    ul_dlogsnr_dt,
    ul_elbo_prefactor,
    ul_logsnr_schedule,
    ul_sample_t,
    ul_terminal_gaussian_kl,
)


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Latents Stage-1 (UL) with Diffusers.")

    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name on the Hub.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Dataset config name.")
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="Local dataset directory (imagefolder style)."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face datasets cache directory.")
    parser.add_argument("--image_column", type=str, default="image", help="Column containing images.")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset from the Hub/local files.")
    parser.add_argument("--output_dir", type=str, default="ul-stage1", help="Where to save checkpoints and samples.")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="unified-latents-stage1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--lambda_z_min", type=float, default=-10.0)
    parser.add_argument("--lambda_z_max", type=float, default=5.0)
    parser.add_argument("--lambda_x_min", type=float, default=-10.0)
    parser.add_argument("--lambda_x_max", type=float, default=10.0)
    parser.add_argument("--prior_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--decoder_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--decoder_loss_factor", type=float, default=1.6)
    parser.add_argument("--decoder_sigmoid_bias", type=float, default=0.0)
    parser.add_argument(
        "--decoder_prediction_type",
        type=str,
        default="x0",
        choices=["epsilon", "x0"],
        help="Decoder target parameterization.",
    )
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument(
        "--latent_downsample_factor",
        type=int,
        default=8,
        help="Downsample factor of encoder output. Current encoder config is 8x; paper target is 16x with 2x2 patching.",
    )

    # Prior (paper Sec. 5.1: single-level ViT, 8 blocks, 1024 channels)
    parser.add_argument("--prior_num_layers", type=int, default=8)
    parser.add_argument("--prior_num_heads", type=int, default=16)
    parser.add_argument("--prior_head_dim", type=int, default=64)  # 16 * 64 = 1024
    parser.add_argument("--prior_patch_size", type=int, default=1)

    # Decoder approximation (paper Sec. 5.1: UViT conv channels [128, 256, 512], dropout 0.1)
    parser.add_argument("--decoder_block_out_channels", type=str, default="128,256,512")
    parser.add_argument("--decoder_layers_per_block", type=int, default=2)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--sample_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    return parser.parse_args()


def build_transforms(args):
    transform_list = [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if args.center_crop:
        transform_list.append(transforms.CenterCrop(args.resolution))
    else:
        transform_list.append(transforms.RandomCrop(args.resolution))

    if args.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_list)


def parse_channels(channels_str: str) -> tuple[int, ...]:
    values = [int(x.strip()) for x in channels_str.split(",") if x.strip()]
    if not values:
        raise ValueError("`decoder_block_out_channels` must contain at least one integer.")
    return tuple(values)


class HFStreamingImageDataset(IterableDataset):
    def __init__(self, hf_iterable, image_column: str, image_transform):
        super().__init__()
        self.hf_iterable = hf_iterable
        self.image_column = image_column
        self.image_transform = image_transform

    def __iter__(self):
        for example in self.hf_iterable:
            image = example[self.image_column].convert("RGB")
            yield {"pixel_values": self.image_transform(image)}


def get_train_dataset_and_collate(args, image_transform):
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
            streaming=args.streaming,
        )
    else:
        if args.train_data_dir is None:
            raise ValueError("Provide either `--dataset_name` or `--train_data_dir`.")
        data_files = {"train": os.path.join(args.train_data_dir, "**")}
        dataset = load_dataset(
            "imagefolder", data_files=data_files, cache_dir=args.cache_dir, streaming=args.streaming
        )

    train_dataset = dataset["train"]

    if args.streaming:
        image_column = args.image_column
        train_dataset = HFStreamingImageDataset(
            train_dataset, image_column=image_column, image_transform=image_transform
        )
    else:
        column_names = train_dataset.column_names
        image_column = args.image_column if args.image_column in column_names else column_names[0]

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [image_transform(image) for image in images]
            return examples

        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        return {"pixel_values": pixel_values.contiguous().float()}

    return train_dataset, collate_fn


def _save_pretrained_components(
    accelerator: Accelerator, encoder: nn.Module, prior: nn.Module, decoder: nn.Module, output_dir
):
    output_dir = Path(output_dir)
    accelerator.unwrap_model(encoder).save_pretrained(output_dir / "encoder")
    accelerator.unwrap_model(prior).save_pretrained(output_dir / "prior")
    accelerator.unwrap_model(decoder).save_pretrained(output_dir / "decoder")


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logging_dir = output_dir / args.logging_dir
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "samples").mkdir(exist_ok=True)
        logger.info("Starting UL Stage-1 training")
        logger.info(f"Output dir: {output_dir}")

    image_transform = build_transforms(args)
    train_dataset, collate_fn = get_train_dataset_and_collate(args, image_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=not args.streaming,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    latent_size = args.resolution // args.latent_downsample_factor
    if latent_size < 1:
        raise ValueError("`resolution // latent_downsample_factor` must be >= 1.")

    encoder = AutoencoderULEncoder(in_channels=3, latent_channels=args.latent_channels)

    prior = DiTTransformer2DModel(
        num_attention_heads=args.prior_num_heads,
        attention_head_dim=args.prior_head_dim,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        num_layers=args.prior_num_layers,
        sample_size=latent_size,
        patch_size=args.prior_patch_size,
        num_embeds_ada_norm=args.num_train_timesteps,
    )

    decoder_channels = parse_channels(args.decoder_block_out_channels)
    decoder = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3 + args.latent_channels,
        out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=decoder_channels,
        layers_per_block=args.decoder_layers_per_block,
        dropout=args.decoder_dropout,
        attention_head_dim=8,
        add_attention=True,
        num_train_timesteps=args.num_train_timesteps,
    )

    params = list(encoder.parameters()) + list(prior.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    has_dataloader_length = True
    try:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    except TypeError:
        has_dataloader_length = False
        num_update_steps_per_epoch = None
    if args.max_train_steps is None or args.max_train_steps <= 0:
        if not has_dataloader_length:
            raise ValueError("For streaming datasets, set `--max_train_steps` to a positive value.")
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    encoder, prior, decoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        encoder, prior, decoder, optimizer, train_dataloader, lr_scheduler
    )
    trainable_params = list(encoder.parameters()) + list(prior.parameters()) + list(decoder.parameters())

    # Recompute schedule values after `accelerator.prepare`.
    if has_dataloader_length:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    else:
        args.num_train_epochs = max(args.num_train_epochs, 1)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    logger.info("***** Running training *****")
    if has_dataloader_length:
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    else:
        logger.info("  Num examples = streaming (unknown)")
        logger.info("  Num batches each epoch = streaming (unknown)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = "
        f"{args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir) if os.path.exists(args.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch if has_dataloader_length else 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    train_loss = 0.0
    fixed_eval_batch_cpu = None
    for epoch in range(first_epoch, args.num_train_epochs):
        encoder.train()
        prior.train()
        decoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(encoder, prior, decoder):
                x = batch["pixel_values"].to(accelerator.device)
                bsz = x.shape[0]
                if fixed_eval_batch_cpu is None:
                    # Keep a fixed mini-batch for deterministic reconstruction snapshots.
                    fixed_eval_batch_cpu = x[: min(4, bsz)].detach().float().cpu()

                # Encode deterministic latent z_clean
                z_clean = encoder.encode(x).latent

                # Prior branch: z_t -> z_clean with unweighted ELBO-style denoising loss
                t_z = ul_sample_t(bsz, x.device)
                lambda_z = ul_logsnr_schedule(
                    t_z,
                    schedule_type=args.prior_schedule,
                    lambda_min=args.lambda_z_min,
                    lambda_max=args.lambda_z_max,
                )
                dlogsnr_dt_z = ul_dlogsnr_dt(
                    t_z,
                    schedule_type=args.prior_schedule,
                    lambda_min=args.lambda_z_min,
                    lambda_max=args.lambda_z_max,
                )
                prefactor_z = ul_elbo_prefactor(lambda_z, dlogsnr_dt_z)
                alpha_z, sigma_z = ul_alpha_sigma_from_logsnr(lambda_z)
                eps_z = torch.randn_like(z_clean)
                z_t = alpha_z[:, None, None, None] * z_clean + sigma_z[:, None, None, None] * eps_z
                t_z_idx = (t_z * (args.num_train_timesteps - 1)).long().clamp(0, args.num_train_timesteps - 1)
                dummy_labels = torch.zeros((bsz,), device=x.device, dtype=torch.long)
                z_pred = prior(z_t, timestep=t_z_idx, class_labels=dummy_labels).sample
                prior_per_sample = F.mse_loss(z_pred.float(), z_clean.float(), reduction="none").sum(dim=(1, 2, 3))
                prior_loss = (prefactor_z * prior_per_sample).mean()
                prior_loss = prior_loss + ul_terminal_gaussian_kl(z_clean, logsnr_terminal=args.lambda_z_min).mean()

                # Decoder branch: x_t conditioned on fixed-noise z0
                lambda0 = torch.full((bsz,), args.lambda_z_max, device=x.device, dtype=x.dtype)
                alpha0, sigma0 = ul_alpha_sigma_from_logsnr(lambda0)
                eps_0 = torch.randn_like(z_clean)
                z0 = alpha0[:, None, None, None] * z_clean + sigma0[:, None, None, None] * eps_0

                t_x = ul_sample_t(bsz, x.device)
                lambda_x = ul_logsnr_schedule(
                    t_x,
                    schedule_type=args.decoder_schedule,
                    lambda_min=args.lambda_x_min,
                    lambda_max=args.lambda_x_max,
                )
                dlogsnr_dt_x = ul_dlogsnr_dt(
                    t_x,
                    schedule_type=args.decoder_schedule,
                    lambda_min=args.lambda_x_min,
                    lambda_max=args.lambda_x_max,
                )
                prefactor_x = ul_elbo_prefactor(lambda_x, dlogsnr_dt_x)
                alpha_x, sigma_x = ul_alpha_sigma_from_logsnr(lambda_x)
                eps_x = torch.randn_like(x)
                x_t = alpha_x[:, None, None, None] * x + sigma_x[:, None, None, None] * eps_x
                t_x_idx = (t_x * (args.num_train_timesteps - 1)).long().clamp(0, args.num_train_timesteps - 1)

                z0_up = F.interpolate(z0, size=x.shape[-2:], mode="bilinear", align_corners=False)
                decoder_input = torch.cat([x_t, z0_up], dim=1)
                decoder_pred = decoder(decoder_input, t_x_idx).sample
                if args.decoder_prediction_type == "epsilon":
                    decoder_target = eps_x
                    decoder_prefactor = 0.5 * (-dlogsnr_dt_x)
                else:
                    decoder_target = x
                    decoder_prefactor = prefactor_x

                decoder_per_sample = F.mse_loss(decoder_pred.float(), decoder_target.float(), reduction="none").sum(
                    dim=(1, 2, 3)
                )
                decoder_w = ul_decoder_loss_weight(
                    lambda_x,
                    bias=args.decoder_sigmoid_bias,
                    loss_factor=args.decoder_loss_factor,
                    invert=False,
                )
                decoder_loss = (decoder_prefactor * decoder_w * decoder_per_sample).mean()

                num_pixels = x.shape[-2] * x.shape[-1]
                bpp_denom = num_pixels * math.log(2.0)
                prior_bpp = prior_loss / bpp_denom
                decoder_bpp = decoder_loss / bpp_denom
                loss = prior_bpp + decoder_bpp
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process and global_step % 50 == 0:
                    logger.info(
                        f"step={global_step} loss_bpp={loss.item():.4f} prior_bpp={prior_bpp.item():.4f} "
                        f"decoder_bpp={decoder_bpp.item():.4f} lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                if accelerator.is_main_process and global_step % args.sample_steps == 0:
                    with torch.no_grad():
                        encoder.eval()
                        decoder.eval()

                        eval_x = fixed_eval_batch_cpu.to(accelerator.device)
                        eval_z_clean = encoder.encode(eval_x).latent
                        eval_z_up = F.interpolate(
                            eval_z_clean, size=eval_x.shape[-2:], mode="bilinear", align_corners=False
                        )

                        # Deterministic decode from encoder latents at t=0 with zero x-noise input.
                        eval_x_t = torch.zeros_like(eval_x)
                        eval_t_idx = torch.zeros((eval_x.shape[0],), device=eval_x.device, dtype=torch.long)
                        eval_decoder_input = torch.cat([eval_x_t, eval_z_up], dim=1)
                        eval_decoder_out = decoder(eval_decoder_input, eval_t_idx).sample
                        if args.decoder_prediction_type == "epsilon":
                            eval_lambda0 = ul_logsnr_schedule(
                                torch.zeros((eval_x.shape[0],), device=eval_x.device),
                                schedule_type=args.decoder_schedule,
                                lambda_min=args.lambda_x_min,
                                lambda_max=args.lambda_x_max,
                            )
                            eval_alpha0, eval_sigma0 = ul_alpha_sigma_from_logsnr(eval_lambda0)
                            eval_recon = (
                                (eval_x_t - eval_sigma0[:, None, None, None] * eval_decoder_out)
                                / eval_alpha0[:, None, None, None].clamp_min(1e-5)
                            ).clamp(-1, 1)
                        else:
                            eval_recon = eval_decoder_out.clamp(-1, 1)

                        # Save side-by-side: left=input, right=reconstruction.
                        side_by_side = torch.cat([eval_x.clamp(-1, 1), eval_recon], dim=3).detach().float().cpu()
                        save_image(
                            (side_by_side + 1.0) / 2.0,
                            output_dir / "samples" / f"step_{global_step:07d}.png",
                            nrow=2,
                        )

                        encoder.train()
                        decoder.train()

                if accelerator.is_main_process and global_step % args.save_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)}"
                            )
                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_dir)
                    _save_pretrained_components(
                        accelerator=accelerator,
                        encoder=encoder,
                        prior=prior,
                        decoder=decoder,
                        output_dir=save_dir,
                    )
                    logger.info(f"Saved state to {save_dir}")

            logs = {
                "step_loss_bpp": loss.detach().item(),
                "prior_loss_bpp": prior_bpp.detach().item(),
                "decoder_loss_bpp": decoder_bpp.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(accelerator.unwrap_model(encoder).state_dict(), final_dir / "encoder.pt")
        torch.save(accelerator.unwrap_model(prior).state_dict(), final_dir / "prior.pt")
        torch.save(accelerator.unwrap_model(decoder).state_dict(), final_dir / "decoder.pt")
        _save_pretrained_components(
            accelerator=accelerator,
            encoder=encoder,
            prior=prior,
            decoder=decoder,
            output_dir=final_dir,
        )
        logger.info(f"Training finished. Saved final checkpoints to {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
