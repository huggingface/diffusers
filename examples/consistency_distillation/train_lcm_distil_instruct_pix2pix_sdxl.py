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
import copy
import functools
import gc
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import datasets
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("file_name", "edited_image", "edit_prompt"),
}
WANDB_TABLE_COL_NAMES = ["file_name", "edited_image", "edit_prompt"]
TORCH_DTYPE_MAPPING = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ============================================================================
# Helper Functions from LCM Training
# ============================================================================

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    """Get boundary scalings for LCM."""
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    """Get predicted x_0 from model output."""
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} is not supported")
    return pred_x_0


def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    """Get predicted noise from model output."""
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(f"Prediction type {prediction_type} is not supported")
    return pred_epsilon


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters using exponential moving average."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """Generate guidance scale embeddings."""
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


class DDIMSolver:
    """DDIM solver for LCM distillation."""
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


# ============================================================================
# Validation and Logging
# ============================================================================

def log_validation(pipeline, args, accelerator, generator, global_step, is_final_validation=False):
    """Run validation and log images."""
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    val_save_dir = os.path.join(args.output_dir, "validation_images")
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else Image.open(image_url_or_path).convert("RGB")
    )(args.val_image_url_or_path)

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        edited_images = []
        for val_img_idx in range(args.num_validation_images):
            a_val_img = pipeline(
                args.validation_prompt,
                image=original_image,
                num_inference_steps=4,  # LCM uses few steps
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator,
            ).images[0]
            edited_images.append(a_val_img)
            a_val_img.save(os.path.join(val_save_dir, f"step_{global_step}_val_img_{val_img_idx}.png"))

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt)
            logger_name = "test" if is_final_validation else "validation"
            tracker.log({logger_name: wandb_table})


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    """Import text encoder class."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def convert_to_np(image, resolution):
    """Convert image to numpy array."""
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LCM Distillation for InstructPix2Pix SDXL")
    
    # Model loading arguments
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained InstructPix2Pix SDXL teacher model.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model for better numerical stability.",
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="VAE precision to avoid NaN issues.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., fp16).",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name for training.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset configuration name.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Folder containing training data.",
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="Column name for original images.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="Column name for edited images.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="Column name for edit instructions.",
    )
    
    # Validation arguments
    parser.add_argument(
        "--val_image_url_or_path",
        type=str,
        default=None,
        help="Path to validation image.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Validation prompt for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation images to generate.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples.",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-xl-lcm-distilled",
        help="Output directory for model and checkpoints.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for datasets and models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Input image resolution.",
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help="Crop coordinate (height) for SDXL.",
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help="Crop coordinate (width) for SDXL.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Use center crop instead of random crop.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Randomly flip images horizontally.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps (overrides num_train_epochs).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale learning rate by number of GPUs and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for learning rate scheduler.",
    )
    
    # LCM-specific arguments
    parser.add_argument(
        "--w_min",
        type=float,
        default=3.0,
        help="Minimum guidance scale for sampling.",
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        help="Maximum guidance scale for sampling.",
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="Number of DDIM timesteps for distillation.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="Loss type for LCM distillation.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="Huber loss parameter.",
    )
    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help="Time condition projection dimension for U-Net.",
    )
    parser.add_argument(
        "--timestep_scaling_factor",
        type=float,
        default=10.0,
        help="Timestep scaling factor for boundary conditions.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        help="EMA decay rate for target U-Net.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability for CFG support.",
    )
    
    # Optimizer arguments
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    
    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    
    # Logging and checkpointing
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard logging directory.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting integration (tensorboard, wandb, etc.).",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Cast teacher U-Net to mixed precision dtype.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("Mixed precision bf16 is not supported on MPS.")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token
            ).repo_id

    # ========================================================================
    # 1. Load Scheduler and Create DDIM Solver
    # ========================================================================
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler", revision=args.revision
    )

    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )

    # ========================================================================
    # 2. Load Tokenizers and Text Encoders
    # ========================================================================
    tokenizer_1 = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder_cls_1 = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.revision
    )
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.revision, subfolder="text_encoder_2"
    )

    text_encoder_1 = text_encoder_cls_1.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant
    )
    text_encoder_2 = text_encoder_cls_2.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant
    )

    # ========================================================================
    # 3. Load VAE
    # ========================================================================
    vae_path = (
        args.pretrained_teacher_model
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )

    # ========================================================================
    # 4. Load Teacher U-Net (InstructPix2Pix with 8 input channels)
    # ========================================================================
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant
    )

    # Verify teacher has 8 input channels
    if teacher_unet.config.in_channels != 8:
        raise ValueError(
            f"Teacher U-Net must have 8 input channels for InstructPix2Pix, "
            f"but has {teacher_unet.config.in_channels}"
        )

    # ========================================================================
    # 5. Create Student U-Net (Online) with Time Conditioning
    # ========================================================================
    time_cond_proj_dim = (
        teacher_unet.config.time_cond_proj_dim
        if teacher_unet.config.time_cond_proj_dim is not None
        else args.unet_time_cond_proj_dim
    )
    
    unet = UNet2DConditionModel.from_config(
        teacher_unet.config,
        time_cond_proj_dim=time_cond_proj_dim
    )
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    unet.train()

    # ========================================================================
    # 6. Create Target Student U-Net (EMA)
    # ========================================================================
    target_unet = UNet2DConditionModel.from_config(unet.config)
    target_unet.load_state_dict(unet.state_dict())
    target_unet.train()
    target_unet.requires_grad_(False)

    # ========================================================================
    # 7. Freeze Models
    # ========================================================================
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # ========================================================================
    # 8. Setup Mixed Precision
    # ========================================================================
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        warnings.warn("weight_dtype fp16 may cause NaN during VAE encoding", UserWarning)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        warnings.warn("weight_dtype bf16 may cause NaN during VAE encoding", UserWarning)

    # Move to device
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=TORCH_DTYPE_MAPPING[args.vae_precision])

    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)

    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # ========================================================================
    # 9. Enable Optimizations
    # ========================================================================
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 may have issues. Consider updating to 0.0.17+."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            target_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ========================================================================
    # 10. Create Optimizer
    # ========================================================================
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps 
            * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes for 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ========================================================================
    # 11. Load Dataset
    # ========================================================================
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names

    # Get column names
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(f"--original_image_column '{args.original_image_column}' not in columns")

    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(f"--edit_prompt_column '{args.edit_prompt_column}' not in columns")

    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(f"--edited_image_column '{args.edited_image_column}' not in columns")

    # ========================================================================
    # 12. Preprocessing Functions
    # ========================================================================
    train_transforms = transforms.Compose([
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    ])

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    def encode_prompt(text_encoders, tokenizers, prompt):
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    f"Input truncated because CLIP can only handle {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def encode_prompts(text_encoders, tokenizers, prompts):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []
        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)
        return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)

    def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(text_encoders, tokenizers, prompts)
            add_text_embeds_all = pooled_prompt_embeds_all
            prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
            add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
        return prompt_embeds_all, add_text_embeds_all

    # Get null conditioning for CFG
    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def compute_null_conditioning():
        null_conditioning_list = []
        for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            null_conditioning_list.append(
                a_text_encoder(
                    tokenize_captions([""], tokenizer=a_tokenizer).to(accelerator.device),
                    output_hidden_states=True,
                ).hidden_states[-2]
            )
        return torch.concat(null_conditioning_list, dim=-1)

    null_conditioning = compute_null_conditioning()

    def compute_time_ids():
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        original_size = target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)

    add_time_ids = compute_time_ids()

    def preprocess_train(examples):
        preprocessed_images = preprocess_images(examples)
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        captions = list(examples[edit_prompt_column])
        prompt_embeds_all, add_text_embeds_all = compute_embeddings_for_prompts(captions, text_encoders, tokenizers)
        examples["prompt_embeds"] = prompt_embeds_all
        examples["add_text_embeds"] = add_text_embeds_all
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        prompt_embeds = torch.concat([example["prompt_embeds"] for example in examples], dim=0)
        add_text_embeds = torch.concat([example["add_text_embeds"] for example in examples], dim=0)
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "prompt_embeds": prompt_embeds,
            "add_text_embeds": add_text_embeds,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # ========================================================================
    # 13. Create Learning Rate Scheduler
    # ========================================================================
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ========================================================================
    # 14. Prepare with Accelerator
    # ========================================================================
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # ========================================================================
    # 15. Setup Model Saving/Loading Hooks
    # ========================================================================
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                target_unet.save_pretrained(os.path.join(output_dir, "unet_target"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet_target"))
            target_unet.load_state_dict(load_model.state_dict())
            target_unet.to(accelerator.device)
            del load_model

            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # ========================================================================
    # 16. Training Setup
    # ========================================================================
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix-xl-lcm", config=vars(args))

    # Create uncond embeddings for CFG
    uncond_prompt_embeds = torch.zeros(args.train_batch_size, 77, 2048).to(accelerator.device)
    uncond_pooled_prompt_embeds = torch.zeros(args.train_batch_size, 1280).to(accelerator.device)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # ========================================================================
    # 17. Resume from Checkpoint
    # ========================================================================
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ========================================================================
    # 18. Training Loop
    # ========================================================================
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Get edited images and encode to latents
                if args.pretrained_vae_model_name_or_path is not None:
                    edited_pixel_values = batch["edited_pixel_values"].to(dtype=weight_dtype)
                else:
                    edited_pixel_values = batch["edited_pixel_values"]
                
                latents = vae.encode(edited_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Get original image embeddings for conditioning
                if args.pretrained_vae_model_name_or_path is not None:
                    original_pixel_values = batch["original_pixel_values"].to(dtype=weight_dtype)
                else:
                    original_pixel_values = batch["original_pixel_values"]
                
                original_image_embeds = vae.encode(original_pixel_values).latent_dist.sample()
                if args.pretrained_vae_model_name_or_path is None:
                    original_image_embeds = original_image_embeds.to(weight_dtype)

                bsz = latents.shape[0]

                # Sample timesteps for LCM distillation
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # Get boundary scalings
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(
                    timesteps, timestep_scaling=args.timestep_scaling_factor
                )
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # Add noise
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, start_timesteps)

                # Sample guidance scale
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w_embedding = guidance_scale_embedding(w, embedding_dim=time_cond_proj_dim)
                w = w.reshape(bsz, 1, 1, 1)
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

                # Get text embeddings
                encoder_hidden_states = batch["prompt_embeds"]
                add_text_embeds = batch["add_text_embeds"]

                # Apply conditioning dropout if specified
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    
                    # Prompt dropout
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Image dropout
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate conditioning
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Online student prediction
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = unet(
                    concatenated_noisy_latents,
                    start_timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=encoder_hidden_states.float(),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )

                model_pred = c_skip_start * noisy_latents + c_out_start * pred_x_0

                # Teacher predictions with CFG
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)

                    with autocast_ctx:
                        # Conditional teacher prediction
                        cond_teacher_output = teacher_unet(
                            concatenated_noisy_latents.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=encoder_hidden_states.to(weight_dtype),
                            added_cond_kwargs={k: v.to(weight_dtype) for k, v in added_cond_kwargs.items()},
                        ).sample
                        
                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_latents,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_latents,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # Unconditional teacher prediction
                        uncond_added_conditions = copy.deepcopy(added_cond_kwargs)
                        uncond_added_conditions["text_embeds"] = uncond_pooled_prompt_embeds
                        
                        # For unconditional, zero out the original image conditioning
                        concatenated_noisy_latents_uncond = torch.cat(
                            [noisy_latents, torch.zeros_like(original_image_embeds)], dim=1
                        )
                        
                        uncond_teacher_output = teacher_unet(
                            concatenated_noisy_latents_uncond.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                            added_cond_kwargs={k: v.to(weight_dtype) for k, v in uncond_added_conditions.items()},
                        ).sample
                        
                        uncond_pred_x0 = get_predicted_original_sample(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_latents,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_noise = get_predicted_noise(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_latents,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # CFG
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                        
                        # DDIM step
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                # Target student prediction
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)

                    with autocast_ctx:
                        # Concatenate x_prev with original image embeds
                        concatenated_x_prev = torch.cat([x_prev, original_image_embeds], dim=1)
                        
                        target_noise_pred = target_unet(
                            concatenated_x_prev.float(),
                            timesteps,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=encoder_hidden_states.float(),
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                # Compute loss
                if args.loss_type == "l2":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.loss_type == "huber":
                    loss = torch.mean(
                        torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # EMA update and logging
            if accelerator.sync_gradients:
                update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # Checkpointing
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Validation
                    if global_step % args.validation_steps == 0:
                        if (args.val_image_url_or_path is not None) and (args.validation_prompt is not None):
                            # Use target UNet for validation
                            pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                                args.pretrained_teacher_model,
                                unet=unwrap_model(target_unet),
                                text_encoder=text_encoder_1,
                                text_encoder_2=text_encoder_2,
                                tokenizer=tokenizer_1,
                                tokenizer_2=tokenizer_2,
                                vae=vae,
                                scheduler=LCMScheduler.from_pretrained(
                                    args.pretrained_teacher_model, subfolder="scheduler"
                                ),
                                revision=args.revision,
                                variant=args.variant,
                                torch_dtype=weight_dtype,
                            )

                            log_validation(
                                pipeline,
                                args,
                                accelerator,
                                generator,
                                global_step,
                                is_final_validation=False,
                            )

                            del pipeline
                            torch.cuda.empty_cache()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # ========================================================================
    # 19. Save Final Model
    # ========================================================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        target_unet = accelerator.unwrap_model(target_unet)
        target_unet.save_pretrained(os.path.join(args.output_dir, "unet_target"))

        # Create final pipeline with LCM scheduler
        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            args.pretrained_teacher_model,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=vae,
            unet=target_unet,
            scheduler=LCMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler"),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        # Final validation
        if (args.val_image_url_or_path is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
                global_step,
                is_final_validation=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
