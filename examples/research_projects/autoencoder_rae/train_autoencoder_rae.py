#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from diffusers import AutoencoderRAE
from diffusers.optimization import get_scheduler


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a stage-1 Representation Autoencoder (RAE) decoder.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to an ImageFolder-style dataset root.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="autoencoder-rae", help="Directory to save checkpoints/model."
    )
    parser.add_argument("--logging_dir", type=str, default="logs", help="Accelerate logging directory.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--validation_steps", type=int, default=500)

    parser.add_argument("--encoder_cls", type=str, choices=["dinov2", "siglip2", "mae"], default="dinov2")
    parser.add_argument("--encoder_name_or_path", type=str, default=None)
    parser.add_argument("--encoder_input_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_channels", type=int, default=3)

    parser.add_argument("--decoder_hidden_size", type=int, default=1152)
    parser.add_argument("--decoder_num_hidden_layers", type=int, default=28)
    parser.add_argument("--decoder_num_attention_heads", type=int, default=16)
    parser.add_argument("--decoder_intermediate_size", type=int, default=4096)

    parser.add_argument("--noise_tau", type=float, default=0.0)
    parser.add_argument("--scaling_factor", type=float, default=1.0)
    parser.add_argument("--reshape_to_2d", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--reconstruction_loss_type",
        type=str,
        choices=["l1", "mse"],
        default="l1",
        help="Pixel reconstruction loss.",
    )
    parser.add_argument(
        "--encoder_loss_weight",
        type=float,
        default=0.0,
        help="Weight for encoder feature consistency loss in the training loop.",
    )
    parser.add_argument(
        "--use_encoder_loss",
        action="store_true",
        help="Enable encoder feature consistency loss term in the training loop.",
    )
    parser.add_argument("--report_to", type=str, default="tensorboard")

    return parser.parse_args()


def build_transforms(args):
    image_transforms = [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    ]
    if args.random_flip:
        image_transforms.append(transforms.RandomHorizontalFlip())
    image_transforms.append(transforms.ToTensor())
    return transforms.Compose(image_transforms)


def compute_losses(model, pixel_values, reconstruction_loss_type: str, use_encoder_loss: bool, encoder_loss_weight: float):
    decoded = model(pixel_values).sample

    if decoded.shape[-2:] != pixel_values.shape[-2:]:
        raise ValueError(
            "Training requires matching reconstruction and target sizes, got "
            f"decoded={tuple(decoded.shape[-2:])}, target={tuple(pixel_values.shape[-2:])}."
        )

    if reconstruction_loss_type == "l1":
        reconstruction_loss = F.l1_loss(decoded.float(), pixel_values.float())
    else:
        reconstruction_loss = F.mse_loss(decoded.float(), pixel_values.float())

    encoder_loss = torch.zeros_like(reconstruction_loss)
    if use_encoder_loss and encoder_loss_weight > 0:
        base_model = model.module if hasattr(model, "module") else model
        target_encoder_input = base_model._maybe_resize_and_normalize(pixel_values)
        reconstructed_encoder_input = base_model._maybe_resize_and_normalize(decoded)

        target_tokens = base_model.encoder(target_encoder_input, requires_grad=False).detach()
        reconstructed_tokens = base_model.encoder(reconstructed_encoder_input, requires_grad=True)
        encoder_loss = F.mse_loss(reconstructed_tokens.float(), target_tokens.float())

    loss = reconstruction_loss + float(encoder_loss_weight) * encoder_loss
    return decoded, loss, reconstruction_loss, encoder_loss


def main():
    args = parse_args()
    if args.resolution != args.image_size:
        raise ValueError(
            f"`--resolution` ({args.resolution}) must match `--image_size` ({args.image_size}) "
            "for stage-1 reconstruction loss."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
        log_with=args.report_to,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset = ImageFolder(args.train_data_dir, transform=build_transforms(args))

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples]).float()
        return {"pixel_values": pixel_values}

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = AutoencoderRAE(
        encoder_cls=args.encoder_cls,
        encoder_name_or_path=args.encoder_name_or_path,
        decoder_hidden_size=args.decoder_hidden_size,
        decoder_num_hidden_layers=args.decoder_num_hidden_layers,
        decoder_num_attention_heads=args.decoder_num_attention_heads,
        decoder_intermediate_size=args.decoder_intermediate_size,
        patch_size=args.patch_size,
        encoder_input_size=args.encoder_input_size,
        image_size=args.image_size,
        num_channels=args.num_channels,
        noise_tau=args.noise_tau,
        reshape_to_2d=args.reshape_to_2d,
        use_encoder_loss=args.use_encoder_loss,
        scaling_factor=args.scaling_factor,
    )
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.train()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if overrode_max_train_steps:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("train_autoencoder_rae", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"]

                _, loss, reconstruction_loss, encoder_loss = compute_losses(
                    model,
                    pixel_values,
                    reconstruction_loss_type=args.reconstruction_loss_type,
                    use_encoder_loss=args.use_encoder_loss,
                    encoder_loss_weight=args.encoder_loss_weight,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "reconstruction_loss": reconstruction_loss.detach().item(),
                    "encoder_loss": encoder_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.validation_steps == 0:
                    with torch.no_grad():
                        _, val_loss, val_reconstruction_loss, val_encoder_loss = compute_losses(
                            model,
                            pixel_values,
                            reconstruction_loss_type=args.reconstruction_loss_type,
                            use_encoder_loss=args.use_encoder_loss,
                            encoder_loss_weight=args.encoder_loss_weight,
                        )
                    accelerator.log(
                        {
                            "val/loss": val_loss.detach().item(),
                            "val/reconstruction_loss": val_reconstruction_loss.detach().item(),
                            "val/encoder_loss": val_encoder_loss.detach().item(),
                        },
                        step=global_step,
                    )

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
