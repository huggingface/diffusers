#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Script to train a consistency model from scratch via (improved) consistency training."""

import argparse
import gc
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, resolve_interpolation_mode
from diffusers.utils import is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_discretization_steps(global_step: int, max_train_steps: int, s_0: int = 10, s_1: int = 1280, constant=False):
    """
    Calculates the current discretization steps at global step k using the discretization curriculum N(k).
    """
    if constant:
        return s_0 + 1

    k_prime = math.floor(max_train_steps / (math.log2(math.floor(s_1 / s_0)) + 1))
    num_discretization_steps = min(s_0 * 2 ** math.floor(global_step / k_prime), s_1) + 1

    return num_discretization_steps


def get_skip_steps(global_step, initial_skip: int = 1):
    # Currently only support constant skip curriculum.
    return initial_skip


def get_karras_sigmas(
    num_discretization_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    dtype=torch.float32,
):
    """
    Calculates the Karras sigmas timestep discretization of [sigma_min, sigma_max].
    """
    ramp = np.linspace(0, 1, num_discretization_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # Make sure sigmas are in increasing rather than decreasing order (see section 2 of the iCT paper)
    sigmas = sigmas[::-1].copy()
    sigmas = torch.from_numpy(sigmas).to(dtype=dtype)
    return sigmas


def get_discretized_lognormal_weights(noise_levels: torch.Tensor, p_mean: float = -1.1, p_std: float = 2.0):
    """
    Calculates the unnormalized weights for a 1D array of noise level sigma_i based on the discretized lognormal"
    " distribution used in the iCT paper (given in Equation 10).
    """
    upper_prob = torch.special.erf((torch.log(noise_levels[1:]) - p_mean) / (math.sqrt(2) * p_std))
    lower_prob = torch.special.erf((torch.log(noise_levels[:-1]) - p_mean) / (math.sqrt(2) * p_std))
    weights = upper_prob - lower_prob
    return weights


def get_loss_weighting_schedule(noise_levels: torch.Tensor):
    """
    Calculates the loss weighting schedule lambda given a set of noise levels.
    """
    return 1.0 / (noise_levels[1:] - noise_levels[:-1])


def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
    # Make sure timesteps (Karras sigmas) have the same device and dtype as original_samples
    sigmas = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
    while len(sigmas.shape) < len(original_samples.shape):
        sigmas = sigmas.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigmas

    return noisy_samples


def get_noise_preconditioning(sigmas, noise_precond_type: str = "cm"):
    """
    Calculates the noise preconditioning function c_noise, which is used to transform the raw Karras sigmas into the
    timestep input for the U-Net.
    """
    if noise_precond_type == "none":
        return sigmas
    elif noise_precond_type == "edm":
        return 0.25 * torch.log(sigmas)
    elif noise_precond_type == "cm":
        return 1000 * 0.25 * torch.log(sigmas + 1e-44)
    else:
        raise ValueError(
            f"Noise preconditioning type {noise_precond_type} is not current supported. Currently supported noise"
            f" preconditioning types are `none` (which uses the sigmas as is), `edm`, and `cm`."
        )


def get_input_preconditioning(sigmas, sigma_data=0.5, input_precond_type: str = "cm"):
    """
    Calculates the input preconditioning factor c_in, which is used to scale the U-Net image input.
    """
    if input_precond_type == "none":
        return 1
    elif input_precond_type == "cm":
        return 1.0 / (sigmas**2 + sigma_data**2)
    else:
        raise ValueError(
            f"Input preconditioning type {input_precond_type} is not current supported. Currently supported input"
            f" preconditioning types are `none` (which uses a scaling factor of 1.0) and `cm`."
        )


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=1.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def log_validation(unet, scheduler, args, accelerator, weight_dtype, step, name="teacher"):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    pipeline = ConsistencyModelPipeline(
        unet=unet,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device=accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    class_labels = [None]
    if args.class_conditional:
        if args.num_classes is not None:
            class_labels = list(range(args.num_classes))
        else:
            logger.warning(
                "The model is class-conditional but the number of classes is not set. The generated images will be"
                " unconditional rather than class-conditional."
            )

    image_logs = []

    for class_label in class_labels:
        images = []
        with torch.autocast("cuda"):
            images = pipeline(
                num_inference_steps=1,
                batch_size=args.eval_batch_size,
                class_labels=[class_label] * args.eval_batch_size,
                generator=generator,
            ).images
        log = {"images": images}
        if args.class_conditional and class_label is not None:
            log["class_label"] = str(class_label)
        else:
            log["class_label"] = "images"
        image_logs.append(log)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                class_label = log["class_label"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(class_label, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                class_label = log["class_label"]
                for image in images:
                    image = wandb.Image(image, caption=class_label)
                    formatted_images.append(image)

            tracker.log({f"validation/{name}": formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ------------Model Arguments-----------
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help=(
            "If initializing the weights from a pretrained model, the path to the pretrained model or model identifier"
            " from huggingface.co/models."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. `fp16`,"
            " `non_ema`, etc.",
        ),
    )
    # ------------Dataset Arguments-----------
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--dataset_image_column_name",
        type=str,
        default="image",
        help="The name of the image column in the dataset to use for training.",
    )
    parser.add_argument(
        "--dataset_class_label_column_name",
        type=str,
        default="label",
        help="If doing class-conditional training, the name of the class label column in the dataset to use.",
    )
    # ------------Image Processing Arguments-----------
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="bilinear",
        help=(
            "The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`,"
            " `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--class_conditional",
        action="store_true",
        help=(
            "Whether to train a class-conditional model. If set, the class labels will be taken from the `label`"
            " column of the provided dataset."
        ),
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="The number of classes in the training data, if training a class-conditional model.",
    )
    parser.add_argument(
        "--class_embed_type",
        type=str,
        default=None,
        help=(
            "The class embedding type to use. Choose from `None`, `identity`, and `timestep`. If `class_conditional`"
            " and `num_classes` and set, but `class_embed_type` is `None`, a embedding matrix will be used."
        ),
    )
    # ------------Dataloader Arguments-----------
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    # ------------Training Arguments-----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # ----Batch Size and Training Length----
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
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
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    # ----Optimizer (Adam) Arguments----
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        help=(
            "The optimizer algorithm to use for training. Choose between `radam` and `adamw`. The iCT paper uses"
            " RAdam."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ----Consistency Training (CT) Specific Arguments----
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        choices=["sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help=(
            "The lower boundary for the timestep discretization, which should be set to a small positive value close"
            " to zero to avoid numerical issues when solving the PF-ODE backwards in time."
        ),
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help=(
            "The upper boundary for the timestep discretization, which also determines the variance of the Gaussian"
            " prior."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="The rho parameter for the Karras sigmas timestep dicretization.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=None,
        help=(
            "The Pseudo-Huber loss parameter c. If not set, this will default to the value recommended in the Improved"
            " Consistency Training (iCT) paper of 0.00054 * sqrt(d), where d is the data dimensionality."
        ),
    )
    parser.add_argument(
        "--discretization_s_0",
        type=int,
        default=10,
        help=(
            "The s_0 parameter in the discretization curriculum N(k). This controls the number of training steps after"
            " which the number of discretization steps N will be doubled."
        ),
    )
    parser.add_argument(
        "--discretization_s_1",
        type=int,
        default=1280,
        help=(
            "The s_1 parameter in the discretization curriculum N(k). This controls the upper limit to the number of"
            " discretization steps used. Increasing this value will reduce the bias at the cost of higher variance."
        ),
    )
    parser.add_argument(
        "--constant_discretization_steps",
        action="store_true",
        help=(
            "Whether to set the discretization curriculum N(k) to be the constant value `discretization_s_0 + 1`. This"
            " is useful for testing when `max_number_steps` is small, when `k_prime` would otherwise be 0, causing"
            " a divide-by-zero error."
        ),
    )
    parser.add_argument(
        "--p_mean",
        type=float,
        default=-1.1,
        help=(
            "The mean parameter P_mean for the (discretized) lognormal noise schedule, which controls the probability"
            " of sampling a (discrete) noise level sigma_i."
        ),
    )
    parser.add_argument(
        "--p_std",
        type=float,
        default=2.0,
        help=(
            "The standard deviation parameter P_std for the (discretized) noise schedule, which controls the"
            " probability of sampling a (discrete) noise level sigma_i."
        ),
    )
    parser.add_argument(
        "--noise_precond_type",
        type=str,
        default="cm",
        help=(
            "The noise preconditioning function to use for transforming the raw Karras sigmas into the timestep"
            " argument of the U-Net. Choose between `none` (the identity function), `edm`, and `cm`."
        ),
    )
    parser.add_argument(
        "--input_precond_type",
        type=str,
        default="cm",
        help=(
            "The input preconditioning function to use for scaling the image input of the U-Net. Choose between `none`"
            " (a scaling factor of 1) and `cm`."
        ),
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=1,
        help=(
            "The gap in indices between the student and teacher noise levels. In the iCT paper this is always set to"
            " 1, but theoretically this could be greater than 1 and/or altered according to a curriculum throughout"
            " training, much like the number of discretization steps is."
        ),
    )
    parser.add_argument(
        "--cast_teacher",
        action="store_true",
        help="Whether to cast the teacher U-Net model to `weight_dtype` or leave it in full precision.",
    )
    # ----Exponential Moving Average (EMA) Arguments----
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_min_decay",
        type=float,
        default=None,
        help=(
            "The minimum decay magnitude for EMA. If not set, this will default to the value of `ema_max_decay`,"
            " resulting in a constant EMA decay rate."
        ),
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.99993,
        help=(
            "The maximum decay magnitude for EMA. Setting `ema_min_decay` equal to this value will result in a"
            " constant decay rate."
        ),
    )
    parser.add_argument(
        "--use_ema_warmup",
        action="store_true",
        help="Whether to use EMA warmup.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    # ----Training Optimization Arguments----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    # ----Distributed Training Arguments----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ------------Validation Arguments-----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help=(
            "The number of images to generate for evaluation. Note that if `class_conditional` and `num_classes` is"
            " set the effective number of images generated per evaluation step is `eval_batch_size * num_classes`."
        ),
    )
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    # ------------Validation Arguments-----------
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    # ------------Logging Arguments-----------
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    # ------------HuggingFace Hub Arguments-----------
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    # ------------Accelerate Arguments-----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="consistency-training",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # 1. Initialize the noise scheduler.
    initial_discretization_steps = get_discretization_steps(
        0,
        args.max_train_steps,
        s_0=args.discretization_s_0,
        s_1=args.discretization_s_1,
        constant=args.constant_discretization_steps,
    )
    noise_scheduler = CMStochasticIterativeScheduler(
        num_train_timesteps=initial_discretization_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
    )

    # 2. Initialize the student U-Net model.
    if args.pretrained_model_name_or_path is not None:
        logger.info(f"Loading pretrained U-Net weights from {args.pretrained_model_name_or_path}... ")
        unet = UNet2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
    elif args.model_config_name_or_path is None:
        # TODO: use default architectures from iCT paper
        if not args.class_conditional and (args.num_classes is not None or args.class_embed_type is not None):
            logger.warning(
                f"`--class_conditional` is set to `False` but `--num_classes` is set to {args.num_classes} and"
                f" `--class_embed_type` is set to {args.class_embed_type}. These values will be overridden to `None`."
            )
            args.num_classes = None
            args.class_embed_type = None
        elif args.class_conditional and args.num_classes is None and args.class_embed_type is None:
            logger.warning(
                "`--class_conditional` is set to `True` but neither `--num_classes` nor `--class_embed_type` is set."
                "`class_conditional` will be overridden to `False`."
            )
            args.class_conditional = False
        unet = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type=args.class_embed_type,
            num_class_embeds=args.num_classes,
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        unet = UNet2DModel.from_config(config)
    unet.train()

    # Create EMA for the student U-Net model.
    if args.use_ema:
        if args.ema_min_decay is None:
            args.ema_min_decay = args.ema_max_decay
        ema_unet = EMAModel(
            unet.parameters(),
            decay=args.ema_max_decay,
            min_decay=args.ema_min_decay,
            use_ema_warmup=args.use_ema_warmup,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=unet.config,
        )

    # 3. Initialize the teacher U-Net model from the student U-Net model.
    # Note that following the improved Consistency Training paper, the teacher U-Net is not updated via EMA (e.g. the
    # EMA decay rate is 0.)
    teacher_unet = UNet2DModel.from_config(unet.config)
    teacher_unet.load_state_dict(unet.state_dict())
    teacher_unet.train()
    teacher_unet.requires_grad_(False)

    # 4. Handle mixed precision and device placement
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Cast teacher_unet to weight_dtype if cast_teacher is set.
    if args.cast_teacher:
        teacher_dtype = weight_dtype
    else:
        teacher_dtype = torch.float32

    teacher_unet.to(accelerator.device)
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # 5. Handle saving and loading of checkpoints.
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                teacher_unet.save_pretrained(os.path.join(output_dir, "unet_teacher"))
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DModel.from_pretrained(os.path.join(input_dir, "unet_teacher"))
            teacher_unet.load_state_dict(load_model.state_dict())
            teacher_unet.to(accelerator.device)
            del load_model

            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 6. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            if args.use_ema:
                ema_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.optimizer_type == "radam":
        optimizer_class = torch.optim.RAdam
    elif args.optimizer_type == "adamw":
        # Use 8-bit Adam for lower memory usage or to fine-tune the model for 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(
            f"Optimizer type {args.optimizer_type} is not supported. Currently supported optimizer types are `radam`"
            f" and `adamw`."
        )

    # 7. Initialize the optimizer
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 8. Dataset creation and data preprocessing
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    interpolation_mode = resolve_interpolation_mode(args.interpolation_type)
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=interpolation_mode),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples[args.dataset_image_column_name]]
        batch_dict = {"images": images}
        if args.class_conditional:
            batch_dict["class_labels"] = examples[args.dataset_class_label_column_name]
        return batch_dict

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # 9. Initialize the learning rate scheduler
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 10. Prepare for training
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    def recalculate_num_discretization_step_values(discretization_steps, skip_steps):
        """
        Recalculates all quantities depending on the number of discretization steps N.
        """
        noise_scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=discretization_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
        )
        current_timesteps = get_karras_sigmas(discretization_steps, args.sigma_min, args.sigma_max, args.rho)
        valid_teacher_timesteps_plus_one = current_timesteps[: len(current_timesteps) - skip_steps + 1]
        # timestep_weights are the unnormalized probabilities of sampling the timestep/noise level at each index
        timestep_weights = get_discretized_lognormal_weights(
            valid_teacher_timesteps_plus_one, p_mean=args.p_mean, p_std=args.p_std
        )
        # timestep_loss_weights is the timestep-dependent loss weighting schedule lambda(sigma_i)
        timestep_loss_weights = get_loss_weighting_schedule(valid_teacher_timesteps_plus_one)

        current_timesteps = current_timesteps.to(accelerator.device)
        timestep_weights = timestep_weights.to(accelerator.device)
        timestep_loss_weights = timestep_loss_weights.to(accelerator.device)

        return noise_scheduler, current_timesteps, timestep_weights, timestep_loss_weights

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Function for unwraping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
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

    # Resolve the c parameter for the Pseudo-Huber loss
    if args.huber_c is None:
        args.huber_c = 0.00054 * args.resolution * math.sqrt(unet.config.in_channels)

    # Get current number of discretization steps N according to our discretization curriculum
    current_discretization_steps = get_discretization_steps(
        initial_global_step,
        args.max_train_steps,
        s_0=args.discretization_s_0,
        s_1=args.discretization_s_1,
        constant=args.constant_discretization_steps,
    )
    current_skip_steps = get_skip_steps(initial_global_step, initial_skip=args.skip_steps)
    if current_skip_steps >= current_discretization_steps:
        raise ValueError(
            f"The current skip steps is {current_skip_steps}, but should be smaller than the current number of"
            f" discretization steps {current_discretization_steps}"
        )
    # Recalculate all quantities depending on the number of discretization steps N
    (
        noise_scheduler,
        current_timesteps,
        timestep_weights,
        timestep_loss_weights,
    ) = recalculate_num_discretization_step_values(current_discretization_steps, current_skip_steps)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # 11. Train!
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # 1. Get batch of images from dataloader (sample x ~ p_data(x))
            clean_images = batch["images"].to(weight_dtype)
            if args.class_conditional:
                class_labels = batch["class_labels"]
            else:
                class_labels = None
            bsz = clean_images.shape[0]

            # 2. Sample a random timestep for each image according to the noise schedule.
            # Sample random indices i ~ p(i), where p(i) is the dicretized lognormal distribution in the iCT paper
            # NOTE: timestep_indices should be in the range [0, len(current_timesteps) - k - 1] inclusive
            timestep_indices = torch.multinomial(timestep_weights, bsz, replacement=True).long()
            teacher_timesteps = current_timesteps[timestep_indices]
            student_timesteps = current_timesteps[timestep_indices + current_skip_steps]

            # 3. Sample noise and add it to the clean images for both teacher and student unets
            # Sample noise z ~ N(0, I) that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            teacher_noisy_images = add_noise(clean_images, noise, teacher_timesteps)
            student_noisy_images = add_noise(clean_images, noise, student_timesteps)

            # 4. Calculate preconditioning and scalings for boundary conditions for the consistency model.
            teacher_rescaled_timesteps = get_noise_preconditioning(teacher_timesteps, args.noise_precond_type)
            student_rescaled_timesteps = get_noise_preconditioning(student_timesteps, args.noise_precond_type)

            c_in_teacher = get_input_preconditioning(teacher_timesteps, input_precond_type=args.input_precond_type)
            c_in_student = get_input_preconditioning(student_timesteps, input_precond_type=args.input_precond_type)

            c_skip_teacher, c_out_teacher = scalings_for_boundary_conditions(teacher_timesteps)
            c_skip_student, c_out_student = scalings_for_boundary_conditions(student_timesteps)

            c_skip_teacher, c_out_teacher, c_in_teacher = [
                append_dims(x, clean_images.ndim) for x in [c_skip_teacher, c_out_teacher, c_in_teacher]
            ]
            c_skip_student, c_out_student, c_in_student = [
                append_dims(x, clean_images.ndim) for x in [c_skip_student, c_out_student, c_in_student]
            ]

            with accelerator.accumulate(unet):
                # 5. Get the student unet denoising prediction on the student timesteps
                # Get rng state now to ensure that dropout is synced between the student and teacher models.
                dropout_state = torch.get_rng_state()
                student_model_output = unet(
                    c_in_student * student_noisy_images, student_rescaled_timesteps, class_labels=class_labels
                ).sample
                # NOTE: currently only support prediction_type == sample, so no need to convert model_output
                student_denoise_output = c_skip_student * student_noisy_images + c_out_student * student_model_output

                # 6. Get the teacher unet denoising prediction on the teacher timesteps
                with torch.no_grad(), torch.autocast("cuda", dtype=teacher_dtype):
                    torch.set_rng_state(dropout_state)
                    teacher_model_output = teacher_unet(
                        c_in_teacher * teacher_noisy_images, teacher_rescaled_timesteps, class_labels=class_labels
                    ).sample
                    # NOTE: currently only support prediction_type == sample, so no need to convert model_output
                    teacher_denoise_output = (
                        c_skip_teacher * teacher_noisy_images + c_out_teacher * teacher_model_output
                    )

                # 7. Calculate the weighted Pseudo-Huber loss
                if args.prediction_type == "sample":
                    # Note that the loss weights should be those at the (teacher) timestep indices.
                    lambda_t = _extract_into_tensor(
                        timestep_loss_weights, timestep_indices, (bsz,) + (1,) * (clean_images.ndim - 1)
                    )
                    loss = lambda_t * (
                        torch.sqrt(
                            (student_denoise_output.float() - teacher_denoise_output.float()) ** 2 + args.huber_c**2
                        )
                        - args.huber_c
                    )
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {args.prediction_type}. Currently, only `sample` is supported."
                    )

                # 8. Backpropagate on the consistency training loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 9. Update teacher_unet and ema_unet parameters using unet's parameters.
                teacher_unet.load_state_dict(unet.state_dict())
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # 10. Recalculate quantities depending on the global step, if necessary.
                    new_discretization_steps = get_discretization_steps(
                        global_step,
                        args.max_train_steps,
                        s_0=args.discretization_s_0,
                        s_1=args.discretization_s_1,
                        constant=args.constant_discretization_steps,
                    )
                    current_skip_steps = get_skip_steps(global_step, initial_skip=args.skip_steps)
                    if current_skip_steps >= new_discretization_steps:
                        raise ValueError(
                            f"The current skip steps is {current_skip_steps}, but should be smaller than the current"
                            f" number of discretization steps {new_discretization_steps}."
                        )
                    if new_discretization_steps != current_discretization_steps:
                        (
                            noise_scheduler,
                            current_timesteps,
                            timestep_weights,
                            timestep_loss_weights,
                        ) = recalculate_num_discretization_step_values(new_discretization_steps, current_skip_steps)
                        current_discretization_steps = new_discretization_steps

                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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

                    if global_step % args.validation_steps == 0:
                        # NOTE: since we do not use EMA for the teacher model, the teacher parameters and student
                        # parameters are the same at this point in time
                        log_validation(unet, noise_scheduler, args, accelerator, weight_dtype, global_step, "teacher")
                        # teacher_unet.to(dtype=teacher_dtype)

                        if args.use_ema:
                            # Store the student unet weights and load the EMA weights.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                            log_validation(
                                unet,
                                noise_scheduler,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                "ema_student",
                            )

                            # Restore student unet weights
                            ema_unet.restore(unet.parameters())

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_unet.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        # progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        pipeline = ConsistencyModelPipeline(unet=unet, scheduler=noise_scheduler)
        pipeline.save_pretrained(args.output_dir)

        # If using EMA, save EMA weights as well.
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

            unet.save_pretrained(os.path.join(args.output_dir, "ema_unet"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
