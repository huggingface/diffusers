#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import json
import logging
import math
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from diffusers import AutoencoderRAE, FlowMatchEulerDiscreteScheduler, RAEDiTTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version
from diffusers.utils.torch_utils import is_compiled_module


check_min_version("0.38.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal stage-2 trainer for RAEDiTTransformer2DModel.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to an ImageFolder-style dataset root. Class folder names define label ids.",
    )
    parser.add_argument("--output_dir", type=str, default="rae-dit", help="Directory to save checkpoints/model.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Accelerate logging directory.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Tracker to use with Accelerate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Training image resolution. Defaults to the loaded RAE image size.",
    )
    parser.add_argument("--center_crop", action="store_true", help="Use center crop instead of random crop.")
    parser.add_argument("--random_flip", action="store_true", help="Apply random horizontal flips during training.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Training epochs if max steps is not set.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total training steps. Overrides num_train_epochs when provided.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before optimizer step.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the transformer.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by world size, accumulation steps, and batch size.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help='Scheduler type. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Scheduler warmup steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Override Accelerate mixed precision mode.",
    )
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 matmul on Ampere+ GPUs.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help="Save Accelerate checkpoints every N optimizer steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoint folders to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Checkpoint path or "latest" to resume from the latest checkpoint in output_dir.',
    )
    parser.add_argument(
        "--pretrained_rae_model_name_or_path",
        type=str,
        required=True,
        help="Path or Hub id for the pretrained stage-1 AutoencoderRAE.",
    )
    parser.add_argument(
        "--pretrained_transformer_model_name_or_path",
        type=str,
        default=None,
        help="Optional path or Hub id for a pretrained RAEDiT transformer checkpoint.",
    )
    parser.add_argument("--patch_size", type=int, default=1, help="Latent patch size for the Stage-2 transformer.")
    parser.add_argument("--encoder_hidden_size", type=int, default=1152, help="Encoder token width.")
    parser.add_argument("--decoder_hidden_size", type=int, default=2048, help="Decoder token width.")
    parser.add_argument("--encoder_num_layers", type=int, default=28, help="Number of encoder blocks.")
    parser.add_argument("--decoder_num_layers", type=int, default=2, help="Number of decoder blocks.")
    parser.add_argument("--encoder_num_attention_heads", type=int, default=16, help="Encoder attention heads.")
    parser.add_argument("--decoder_num_attention_heads", type=int, default=16, help="Decoder attention heads.")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio.")
    parser.add_argument(
        "--class_dropout_prob",
        type=float,
        default=0.1,
        help="Class dropout probability for classifier-free guidance readiness.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of class labels. Defaults to the number of ImageFolder classes.",
    )
    parser.add_argument("--use_qknorm", action="store_true", help="Enable QK norm in attention.")
    parser.add_argument("--use_swiglu", action=argparse.BooleanOptionalAction, default=True, help="Use SwiGLU MLPs.")
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True, help="Use rotary embeddings.")
    parser.add_argument(
        "--use_rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RMSNorm instead of LayerNorm.",
    )
    parser.add_argument("--wo_shift", action="store_true", help="Disable AdaLN shift modulation.")
    parser.add_argument(
        "--use_pos_embed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed sin-cos positional embeddings on the encoder stream.",
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000,
        help="Number of flow-matching training timesteps.",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Explicit flow-matching shift. If omitted, it is derived from the latent size.",
    )
    parser.add_argument(
        "--time_shift_base",
        type=float,
        default=4096.0,
        help="Base latent dimensionality used to derive the default flow shift.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help='Weighting scheme for flow-matching timestep sampling and loss weighting.',
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean used when the logit-normal weighting scheme is selected.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std used when the logit-normal weighting scheme is selected.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Mode weighting scale used when weighting_scheme=mode.",
    )
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


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples]).float()
    class_labels = torch.tensor([example[1] for example in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "class_labels": class_labels}


def get_latent_spec(autoencoder: AutoencoderRAE) -> tuple[int, int]:
    if not autoencoder.config.reshape_to_2d:
        raise ValueError("Stage-2 RAE DiT training expects `AutoencoderRAE.reshape_to_2d=True`.")

    latent_channels = int(autoencoder.config.encoder_hidden_size)
    latent_size = int(autoencoder.config.encoder_input_size // autoencoder.config.encoder_patch_size)
    return latent_channels, latent_size


def resolve_flow_shift(args, latent_channels: int, latent_size: int) -> float:
    if args.flow_shift is not None:
        return float(args.flow_shift)

    latent_dim = latent_channels * latent_size * latent_size
    return math.sqrt(latent_dim / float(args.time_shift_base))


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device=None):
    if device is None:
        device = timesteps.device

    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device=device)
    timesteps = timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def maybe_prune_checkpoints(output_dir: str, checkpoints_total_limit: int | None):
    if checkpoints_total_limit is None:
        return

    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    if len(checkpoints) < checkpoints_total_limit:
        return

    num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
    for checkpoint in checkpoints[:num_to_remove]:
        shutil.rmtree(os.path.join(output_dir, checkpoint))


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
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

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    autoencoder = AutoencoderRAE.from_pretrained(args.pretrained_rae_model_name_or_path)
    autoencoder.requires_grad_(False)
    autoencoder.eval()

    latent_channels, latent_size = get_latent_spec(autoencoder)
    if args.resolution is None:
        args.resolution = int(autoencoder.config.image_size)

    dataset = ImageFolder(args.train_data_dir, transform=build_transforms(args))
    inferred_num_classes = len(dataset.classes)
    num_classes = inferred_num_classes if args.num_classes is None else int(args.num_classes)
    if num_classes < inferred_num_classes:
        raise ValueError(
            f"`--num_classes` ({num_classes}) must be >= the number of dataset classes ({inferred_num_classes})."
        )

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.pretrained_transformer_model_name_or_path is not None:
        transformer = RAEDiTTransformer2DModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
        if transformer.config.in_channels != latent_channels or transformer.config.sample_size != latent_size:
            raise ValueError(
                "Loaded transformer latent shape does not match the selected AutoencoderRAE. "
                f"Expected channels={latent_channels}, size={latent_size}; got "
                f"channels={transformer.config.in_channels}, size={transformer.config.sample_size}."
            )
        if transformer.config.num_classes < num_classes:
            raise ValueError(
                f"Loaded transformer supports {transformer.config.num_classes} classes but dataset requires {num_classes}."
            )
    else:
        transformer = RAEDiTTransformer2DModel(
            sample_size=latent_size,
            patch_size=args.patch_size,
            in_channels=latent_channels,
            hidden_size=(args.encoder_hidden_size, args.decoder_hidden_size),
            depth=(args.encoder_num_layers, args.decoder_num_layers),
            num_heads=(args.encoder_num_attention_heads, args.decoder_num_attention_heads),
            mlp_ratio=args.mlp_ratio,
            class_dropout_prob=args.class_dropout_prob,
            num_classes=num_classes,
            use_qknorm=args.use_qknorm,
            use_swiglu=args.use_swiglu,
            use_rope=args.use_rope,
            use_rmsnorm=args.use_rmsnorm,
            wo_shift=args.wo_shift,
            use_pos_embed=args.use_pos_embed,
        )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    flow_shift = resolve_flow_shift(args, latent_channels=latent_channels, latent_size=latent_size)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.num_train_timesteps,
        shift=flow_shift,
    )

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
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

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        for model in models:
            if isinstance(unwrap_model(accelerator, model), RAEDiTTransformer2DModel):
                unwrap_model(accelerator, model).save_pretrained(os.path.join(output_dir, "transformer"))
            else:
                raise ValueError(f"Unexpected model type during save: {type(model)}")

            if weights:
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            target_model = unwrap_model(accelerator, model)
            if not isinstance(target_model, RAEDiTTransformer2DModel):
                raise ValueError(f"Unexpected model type during load: {type(model)}")

            load_model = RAEDiTTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
            target_model.register_to_config(**load_model.config)
            target_model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    autoencoder.to(accelerator.device, dtype=weight_dtype)

    if overrode_max_train_steps:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "train_rae_dit",
            config={
                **vars(args),
                "latent_channels": latent_channels,
                "latent_size": latent_size,
                "flow_shift": flow_shift,
                "inferred_num_classes": inferred_num_classes,
            },
        )
        with open(os.path.join(args.output_dir, "id2label.json"), "w", encoding="utf-8") as f:
            json.dump({idx: label for idx, label in enumerate(dataset.classes)}, f, indent=2, sort_keys=True)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running stage-2 RAE DiT training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num classes = {inferred_num_classes}")
    logger.info(f"  RAE latent shape = ({latent_channels}, {latent_size}, {latent_size})")
    logger.info(f"  Flow shift = {flow_shift:.4f}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
            if not os.path.isdir(path):
                path = os.path.join(args.output_dir, os.path.basename(path))
        else:
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, checkpoints[-1]) if checkpoints else None

        if path is None or not os.path.isdir(path):
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(os.path.basename(path).split("-")[1])
            initial_global_step = global_step
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        epoch_dataloader = train_dataloader
        if args.resume_from_checkpoint and epoch == first_epoch and resume_step > 0:
            epoch_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=resume_step)

        for batch in epoch_dataloader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                class_labels = batch["class_labels"].to(device=accelerator.device, non_blocking=True)

                with torch.no_grad():
                    latents = autoencoder.encode(pixel_values).latent

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=latents.device,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps.to(device=latents.device)[indices]

                sigmas = get_sigmas(
                    noise_scheduler,
                    timesteps,
                    n_dim=latents.ndim,
                    dtype=latents.dtype,
                    device=latents.device,
                )
                noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)

                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps / noise_scheduler.config.num_train_timesteps,
                    class_labels=class_labels,
                ).sample

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        maybe_prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                    accelerator.wait_for_everyone()

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        noise_scheduler.save_pretrained(os.path.join(save_path, "scheduler"))
                    logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_transformer = unwrap_model(accelerator, transformer)
        unwrapped_transformer.save_pretrained(os.path.join(args.output_dir, "transformer"))
        noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
