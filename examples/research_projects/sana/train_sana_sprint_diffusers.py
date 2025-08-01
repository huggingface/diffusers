#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Sana-Sprint team. All rights reserved.
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
import io
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from safetensors.torch import load_file
from torch.nn.utils.spectral_norm import SpectralNorm
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model

import diffusers
from diffusers import (
    AutoencoderDC,
    SanaPipeline,
    SanaSprintPipeline,
    SanaTransformer2DModel,
)
from diffusers.models.attention_processor import Attention
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


class SanaVanillaAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention to support JVP calculation during training.
    """

    def __init__(self):
        pass

    @staticmethod
    def scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ) -> torch.Tensor:
        B, H, L, S = *query.size()[:-1], key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class Text2ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading text-image pairs from a HuggingFace dataset.
    This dataset is designed for text-to-image generation tasks.
    Args:
        hf_dataset (datasets.Dataset):
            A HuggingFace dataset containing 'image' (bytes) and 'llava' (text) fields. Note that 'llava' is the field name for text descriptions in this specific dataset - you may need to adjust this key if using a different HuggingFace dataset with a different text field name.
            resolution (int, optional): Target resolution for image resizing. Defaults to 1024.
    Returns:
        dict: A dictionary containing:
            - 'text': The text description (str)
            - 'image': The processed image tensor (torch.Tensor) of shape [3, resolution, resolution]
    """

    def __init__(self, hf_dataset, resolution=1024):
        self.dataset = hf_dataset
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize(resolution),  # Image.BICUBIC
                T.CenterCrop(resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["llava"]
        image_bytes = item["image"]

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        image_tensor = self.transform(image)

        return {"text": text, "image": image_tensor}


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Sana Sprint - {repo_id}

<Gallery />

## Model description

These are {repo_id} Sana Sprint weights for {base_model}.

The weights were trained using [Sana-Sprint](https://nvlabs.github.io/Sana/Sprint/).

## License

TODO
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sana-sprint",
        "sana-sprint-diffusers",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    if args.enable_vae_tiling:
        pipeline.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_stride_width=1024)

    pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

    images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300,
        help="Maximum sequence length to use with with the Gemma model",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # ----Image Processing----
    parser.add_argument("--file_path", nargs="+", required=True, help="List of parquet files (space-separated)")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--use_fix_crop_and_size",
        action="store_true",
        help="Whether or not to use the fixed crop and size for the teacher model.",
        default=False,
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
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.2, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.6, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_mean_discriminator", type=float, default=-0.6, help="Logit mean for discriminator timestep sampling"
    )
    parser.add_argument(
        "--logit_std_discriminator", type=float, default=1.0, help="Logit std for discriminator timestep sampling"
    )
    parser.add_argument("--ladd_multi_scale", action="store_true", help="Whether to use multi-scale discriminator")
    parser.add_argument(
        "--head_block_ids",
        type=int,
        nargs="+",
        default=[2, 8, 14, 19],
        help="Specify which transformer blocks to use for discriminator heads",
    )
    parser.add_argument("--adv_lambda", type=float, default=0.5, help="Weighting coefficient for adversarial loss")
    parser.add_argument("--scm_lambda", type=float, default=1.0, help="Weighting coefficient for SCM loss")
    parser.add_argument("--gradient_clip", type=float, default=0.1, help="Threshold for gradient clipping")
    parser.add_argument(
        "--sigma_data", type=float, default=0.5, help="Standard deviation of data distribution is supposed to be 0.5"
    )
    parser.add_argument(
        "--tangent_warmup_steps", type=int, default=4000, help="Number of warmup steps for tangent vectors"
    )
    parser.add_argument(
        "--guidance_embeds_scale", type=float, default=0.1, help="Scaling factor for guidance embeddings"
    )
    parser.add_argument(
        "--scm_cfg_scale", type=float, nargs="+", default=[4, 4.5, 5], help="Range for classifier-free guidance scale"
    )
    parser.add_argument(
        "--train_largest_timestep", action="store_true", help="Whether to enable special training for large timesteps"
    )
    parser.add_argument("--largest_timestep", type=float, default=1.57080, help="Maximum timestep value")
    parser.add_argument(
        "--largest_timestep_prob", type=float, default=0.5, help="Sampling probability for large timesteps"
    )
    parser.add_argument(
        "--misaligned_pairs_D", action="store_true", help="Add misaligned sample pairs for discriminator"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_vae_tiling", action="store_true", help="Enabla vae tiling in log validation")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


# Adapted from https://github.com/autonomousvision/stylegan-t/blob/main/networks/discriminator.py
class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1), ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = nn.Linear(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class SanaMSCMDiscriminator(nn.Module):
    def __init__(self, pretrained_model, is_multiscale=False, head_block_ids=None):
        super().__init__()
        self.transformer = pretrained_model
        self.transformer.requires_grad_(False)

        if head_block_ids is None or len(head_block_ids) == 0:
            self.block_hooks = {2, 8, 14, 20, 27} if is_multiscale else {self.transformer.depth - 1}
        else:
            self.block_hooks = head_block_ids

        heads = []
        for i in range(len(self.block_hooks)):
            heads.append(DiscHead(self.transformer.hidden_size, 0, 0))
        self.heads = nn.ModuleList(heads)

    def get_head_inputs(self):
        return self.head_inputs

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, **kwargs):
        feat_list = []
        self.head_inputs = []

        def get_features(module, input, output):
            feat_list.append(output)
            return output

        hooks = []
        for i, block in enumerate(self.transformer.transformer_blocks):
            if i in self.block_hooks:
                hooks.append(block.register_forward_hook(get_features))

        self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_logvar=False,
            **kwargs,
        )

        for hook in hooks:
            hook.remove()

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            B, N, C = feat.shape
            feat = feat.transpose(1, 2)  # [B, C, N]
            self.head_inputs.append(feat)
            res_list.append(head(feat, None).reshape(feat.shape[0], -1))

        concat_res = torch.cat(res_list, dim=1)

        return concat_res

    @property
    def model(self):
        return self.transformer

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)


class DiscHeadModel:
    def __init__(self, disc):
        self.disc = disc

    def state_dict(self):
        return {name: param for name, param in self.disc.state_dict().items() if not name.startswith("transformer.")}

    def __getattr__(self, name):
        return getattr(self.disc, name)


class SanaTrigFlow(SanaTransformer2DModel):
    def __init__(self, original_model, guidance=False):
        self.__dict__ = original_model.__dict__
        self.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.guidance = guidance
        if self.guidance:
            hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
            self.logvar_linear = torch.nn.Linear(hidden_size, 1)
            torch.nn.init.xavier_uniform_(self.logvar_linear.weight)
            torch.nn.init.constant_(self.logvar_linear.bias, 0)

    def forward(
        self, hidden_states, encoder_hidden_states, timestep, guidance=None, jvp=False, return_logvar=False, **kwargs
    ):
        batch_size = hidden_states.shape[0]
        latents = hidden_states
        prompt_embeds = encoder_hidden_states
        t = timestep

        # TrigFlow --> Flow Transformation
        timestep = t.expand(latents.shape[0]).to(prompt_embeds.dtype)
        latents_model_input = latents

        flow_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))

        flow_timestep_expanded = flow_timestep.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2
        )
        latent_model_input = latent_model_input.to(prompt_embeds.dtype)

        # forward in original flow

        if jvp and self.gradient_checkpointing:
            self.gradient_checkpointing = False
            model_out = super().forward(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=flow_timestep,
                guidance=guidance,
                **kwargs,
            )[0]
            self.gradient_checkpointing = True
        else:
            model_out = super().forward(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=flow_timestep,
                guidance=guidance,
                **kwargs,
            )[0]

        # Flow --> TrigFlow Transformation
        trigflow_model_out = (
            (1 - 2 * flow_timestep_expanded) * latent_model_input
            + (1 - 2 * flow_timestep_expanded + 2 * flow_timestep_expanded**2) * model_out
        ) / torch.sqrt(flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2)

        if self.guidance and guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        if return_logvar:
            logvar = self.logvar_linear(embedded_timestep)
            return trigflow_model_out, logvar

        return (trigflow_model_out,)


def compute_density_for_timestep_sampling_scm(batch_size: int, logit_mean: float = None, logit_std: float = None):
    """Compute the density for sampling the timesteps when doing Sana-Sprint training."""
    sigma = torch.randn(batch_size, device="cpu")
    sigma = (sigma * logit_std + logit_mean).exp()
    u = torch.atan(sigma / 0.5)  # TODO: 0.5 should be a hyper-parameter

    return u


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
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
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
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
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load scheduler and models
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    ori_transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        guidance_embeds=True,
    )
    ori_transformer.set_attn_processor(SanaVanillaAttnProcessor())

    ori_transformer_no_guide = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        guidance_embeds=False,
    )

    original_state_dict = load_file(
        f"{args.pretrained_model_name_or_path}/transformer/diffusion_pytorch_model.safetensors"
    )

    param_mapping = {
        "time_embed.emb.timestep_embedder.linear_1.weight": "time_embed.timestep_embedder.linear_1.weight",
        "time_embed.emb.timestep_embedder.linear_1.bias": "time_embed.timestep_embedder.linear_1.bias",
        "time_embed.emb.timestep_embedder.linear_2.weight": "time_embed.timestep_embedder.linear_2.weight",
        "time_embed.emb.timestep_embedder.linear_2.bias": "time_embed.timestep_embedder.linear_2.bias",
    }

    for src_key, dst_key in param_mapping.items():
        if src_key in original_state_dict:
            ori_transformer.load_state_dict({dst_key: original_state_dict[src_key]}, strict=False, assign=True)

    guidance_embedder_module = ori_transformer.time_embed.guidance_embedder

    zero_state_dict = {}

    target_device = accelerator.device
    param_w1 = guidance_embedder_module.linear_1.weight
    zero_state_dict["linear_1.weight"] = torch.zeros(param_w1.shape, device=target_device)
    param_b1 = guidance_embedder_module.linear_1.bias
    zero_state_dict["linear_1.bias"] = torch.zeros(param_b1.shape, device=target_device)
    param_w2 = guidance_embedder_module.linear_2.weight
    zero_state_dict["linear_2.weight"] = torch.zeros(param_w2.shape, device=target_device)
    param_b2 = guidance_embedder_module.linear_2.bias
    zero_state_dict["linear_2.bias"] = torch.zeros(param_b2.shape, device=target_device)
    guidance_embedder_module.load_state_dict(zero_state_dict, strict=False, assign=True)

    transformer = SanaTrigFlow(ori_transformer, guidance=True).train()
    pretrained_model = SanaTrigFlow(ori_transformer_no_guide, guidance=False).eval()

    disc = SanaMSCMDiscriminator(
        pretrained_model,
        is_multiscale=args.ladd_multi_scale,
        head_block_ids=args.head_block_ids,
    ).train()

    transformer.requires_grad_(True)
    pretrained_model.requires_grad_(False)
    disc.model.requires_grad_(False)
    disc.heads.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # VAE should always be kept in fp32 for SANA (?)
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    pretrained_model.to(accelerator.device, dtype=weight_dtype)
    disc.to(accelerator.device, dtype=weight_dtype)
    # because Gemma2 is particularly suited for bfloat16.
    text_encoder.to(dtype=torch.bfloat16)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            for block in transformer.transformer_blocks:
                block.attn2.set_use_npu_flash_attention(True)
            for block in pretrained_model.transformer_blocks:
                block.attn2.set_use_npu_flash_attention(True)
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    unwrapped_model = unwrap_model(model)
                    # Handle transformer model
                    if isinstance(unwrapped_model, type(unwrap_model(transformer))):
                        model = unwrapped_model
                        model.save_pretrained(os.path.join(output_dir, "transformer"))
                    # Handle discriminator model (only save heads)
                    elif isinstance(unwrapped_model, type(unwrap_model(disc))):
                        # Save only the heads
                        torch.save(unwrapped_model.heads.state_dict(), os.path.join(output_dir, "disc_heads.pt"))
                    else:
                        raise ValueError(f"unexpected save model: {unwrapped_model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            transformer_ = None
            disc_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    unwrapped_model = unwrap_model(model)

                    if isinstance(unwrapped_model, type(unwrap_model(transformer))):
                        transformer_ = model  # noqa: F841
                    elif isinstance(unwrapped_model, type(unwrap_model(disc))):
                        # Load only the heads
                        heads_state_dict = torch.load(os.path.join(input_dir, "disc_heads.pt"))
                        unwrapped_model.heads.load_state_dict(heads_state_dict)
                        disc_ = model  # noqa: F841
                    else:
                        raise ValueError(f"unexpected save model: {unwrapped_model.__class__}")

            else:
                # DeepSpeed case
                transformer_ = SanaTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")  # noqa: F841
                disc_heads_state_dict = torch.load(os.path.join(input_dir, "disc_heads.pt"))  # noqa: F841
                # You'll need to handle how to load the heads in DeepSpeed case

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
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

    # Optimization parameters
    optimizer_G = optimizer_class(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_D = optimizer_class(
        disc.heads.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    hf_dataset = load_dataset(
        args.dataset_name,
        data_files=args.file_path,
        split="train",
    )

    train_dataset = Text2ImageDataset(
        hf_dataset=hf_dataset,
        resolution=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, pretrained_model, disc, optimizer_G, optimizer_D, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            transformer, pretrained_model, disc, optimizer_G, optimizer_D, train_dataloader, lr_scheduler
        )
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sana-sprint"
        config = {
            k: str(v) if not isinstance(v, (int, float, str, bool, torch.Tensor)) else v for k, v in vars(args).items()
        }
        accelerator.init_trackers(tracker_name, config=config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
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
            # Get the mos recent checkpoint
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
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    phase = "G"
    vae_config_scaling_factor = vae.config.scaling_factor
    sigma_data = args.sigma_data
    negative_prompt = [""] * args.train_batch_size
    negative_prompt_embeds, negative_prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
        prompt=negative_prompt,
        complex_human_instruction=False,
        do_classifier_free_guidance=False,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        disc.train()

        for step, batch in enumerate(train_dataloader):
            # text encoding
            prompts = batch["text"]
            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                    prompts, complex_human_instruction=COMPLEX_HUMAN_INSTRUCTION, do_classifier_free_guidance=False
                )

            # Convert images to latent space
            vae = vae.to(accelerator.device)
            pixel_values = batch["image"].to(dtype=vae.dtype)
            model_input = vae.encode(pixel_values).latent
            model_input = model_input * vae_config_scaling_factor * sigma_data
            model_input = model_input.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input) * sigma_data
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling_scm(
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
            ).to(accelerator.device)

            # Add noise according to TrigFlow.
            # zt = cos(t) * x + sin(t) * noise
            t = u.view(-1, 1, 1, 1)
            noisy_model_input = torch.cos(t) * model_input + torch.sin(t) * noise

            scm_cfg_scale = torch.tensor(
                np.random.choice(args.scm_cfg_scale, size=bsz, replace=True),
                device=accelerator.device,
            )

            def model_wrapper(scaled_x_t, t):
                pred, logvar = accelerator.unwrap_model(transformer)(
                    hidden_states=scaled_x_t,
                    timestep=t.flatten(),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=(scm_cfg_scale.flatten() * args.guidance_embeds_scale),
                    jvp=True,
                    return_logvar=True,
                )
                return pred, logvar

            if phase == "G":
                transformer.train()
                disc.eval()
                models_to_accumulate = [transformer]
                with accelerator.accumulate(models_to_accumulate):
                    with torch.no_grad():
                        cfg_x_t = torch.cat([noisy_model_input, noisy_model_input], dim=0)
                        cfg_t = torch.cat([t, t], dim=0)
                        cfg_y = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        cfg_y_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

                        cfg_pretrain_pred = pretrained_model(
                            hidden_states=(cfg_x_t / sigma_data),
                            timestep=cfg_t.flatten(),
                            encoder_hidden_states=cfg_y,
                            encoder_attention_mask=cfg_y_mask,
                        )[0]

                        cfg_dxt_dt = sigma_data * cfg_pretrain_pred

                        dxt_dt_uncond, dxt_dt = cfg_dxt_dt.chunk(2)

                        scm_cfg_scale = scm_cfg_scale.view(-1, 1, 1, 1)
                        dxt_dt = dxt_dt_uncond + scm_cfg_scale * (dxt_dt - dxt_dt_uncond)

                    v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
                    v_t = torch.cos(t) * torch.sin(t)

                    # Adapt from https://github.com/xandergos/sCM-mnist/blob/master/train_consistency.py
                    with torch.no_grad():
                        F_theta, F_theta_grad, logvar = torch.func.jvp(
                            model_wrapper, (noisy_model_input / sigma_data, t), (v_x, v_t), has_aux=True
                        )

                    F_theta, logvar = transformer(
                        hidden_states=(noisy_model_input / sigma_data),
                        timestep=t.flatten(),
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        guidance=(scm_cfg_scale.flatten() * args.guidance_embeds_scale),
                        return_logvar=True,
                    )

                    logvar = logvar.view(-1, 1, 1, 1)
                    F_theta_grad = F_theta_grad.detach()
                    F_theta_minus = F_theta.detach()

                    # Warmup steps
                    r = min(1, global_step / args.tangent_warmup_steps)

                    # Calculate gradient g using JVP rearrangement
                    g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
                    second_term = -r * (torch.cos(t) * torch.sin(t) * noisy_model_input + sigma_data * F_theta_grad)
                    g = g + second_term

                    # Tangent normalization
                    g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
                    g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper

                    sigma = torch.tan(t) * sigma_data
                    weight = 1 / sigma

                    l2_loss = torch.square(F_theta - F_theta_minus - g)

                    # Calculate loss with normalization factor
                    loss = (weight / torch.exp(logvar)) * l2_loss + logvar

                    loss = loss.mean()

                    loss_no_logvar = weight * torch.square(F_theta - F_theta_minus - g)
                    loss_no_logvar = loss_no_logvar.mean()
                    g_norm = g_norm.mean()

                    pred_x_0 = torch.cos(t) * noisy_model_input - torch.sin(t) * F_theta * sigma_data

                    if args.train_largest_timestep:
                        pred_x_0.detach()
                        u = compute_density_for_timestep_sampling_scm(
                            batch_size=bsz,
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                        ).to(accelerator.device)
                        t_new = u.view(-1, 1, 1, 1)

                        random_mask = torch.rand_like(t_new) < args.largest_timestep_prob

                        t_new = torch.where(random_mask, torch.full_like(t_new, args.largest_timestep), t_new)
                        z_new = torch.randn_like(model_input) * sigma_data
                        x_t_new = torch.cos(t_new) * model_input + torch.sin(t_new) * z_new

                        F_theta = transformer(
                            hidden_states=(x_t_new / sigma_data),
                            timestep=t_new.flatten(),
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            guidance=(scm_cfg_scale.flatten() * args.guidance_embeds_scale),
                            return_logvar=False,
                            jvp=False,
                        )[0]

                        pred_x_0 = torch.cos(t_new) * x_t_new - torch.sin(t_new) * F_theta * sigma_data

                    # Sample timesteps for discriminator
                    timesteps_D = compute_density_for_timestep_sampling_scm(
                        batch_size=bsz,
                        logit_mean=args.logit_mean_discriminator,
                        logit_std=args.logit_std_discriminator,
                    ).to(accelerator.device)
                    t_D = timesteps_D.view(-1, 1, 1, 1)

                    # Add noise to predicted x0
                    z_D = torch.randn_like(model_input) * sigma_data
                    noised_predicted_x0 = torch.cos(t_D) * pred_x_0 + torch.sin(t_D) * z_D

                    # Calculate adversarial loss
                    pred_fake = disc(
                        hidden_states=(noised_predicted_x0 / sigma_data),
                        timestep=t_D.flatten(),
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                    )
                    adv_loss = -torch.mean(pred_fake)

                    # Total loss = sCM loss + LADD loss

                    total_loss = args.scm_lambda * loss + adv_loss * args.adv_lambda

                    total_loss = total_loss / args.gradient_accumulation_steps

                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.gradient_clip)
                        if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                            optimizer_G.zero_grad(set_to_none=True)
                            optimizer_D.zero_grad(set_to_none=True)
                            logger.warning("NaN or Inf detected in grad_norm, skipping iteration...")
                            continue

                        # switch phase to D
                        phase = "D"

                        optimizer_G.step()
                        lr_scheduler.step()
                        optimizer_G.zero_grad(set_to_none=True)

            elif phase == "D":
                transformer.eval()
                disc.train()
                models_to_accumulate = [disc]
                with accelerator.accumulate(models_to_accumulate):
                    with torch.no_grad():
                        scm_cfg_scale = torch.tensor(
                            np.random.choice(args.scm_cfg_scale, size=bsz, replace=True),
                            device=accelerator.device,
                        )

                        if args.train_largest_timestep:
                            random_mask = torch.rand_like(t) < args.largest_timestep_prob
                            t = torch.where(random_mask, torch.full_like(t, args.largest_timestep_prob), t)

                            z_new = torch.randn_like(model_input) * sigma_data
                            noisy_model_input = torch.cos(t) * model_input + torch.sin(t) * z_new
                        # here
                        F_theta = transformer(
                            hidden_states=(noisy_model_input / sigma_data),
                            timestep=t.flatten(),
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            guidance=(scm_cfg_scale.flatten() * args.guidance_embeds_scale),
                            return_logvar=False,
                            jvp=False,
                        )[0]
                        pred_x_0 = torch.cos(t) * noisy_model_input - torch.sin(t) * F_theta * sigma_data

                    # Sample timesteps for fake and real samples
                    timestep_D_fake = compute_density_for_timestep_sampling_scm(
                        batch_size=bsz,
                        logit_mean=args.logit_mean_discriminator,
                        logit_std=args.logit_std_discriminator,
                    ).to(accelerator.device)
                    timesteps_D_real = timestep_D_fake

                    t_D_fake = timestep_D_fake.view(-1, 1, 1, 1)
                    t_D_real = timesteps_D_real.view(-1, 1, 1, 1)

                    # Add noise to predicted x0 and real x0
                    z_D_fake = torch.randn_like(model_input) * sigma_data
                    z_D_real = torch.randn_like(model_input) * sigma_data
                    noised_predicted_x0 = torch.cos(t_D_fake) * pred_x_0 + torch.sin(t_D_fake) * z_D_fake
                    noised_latents = torch.cos(t_D_real) * model_input + torch.sin(t_D_real) * z_D_real

                    # Add misaligned pairs if enabled and batch size > 1
                    if args.misaligned_pairs_D and bsz > 1:
                        # Create shifted pairs
                        shifted_x0 = torch.roll(model_input, 1, 0)
                        timesteps_D_shifted = compute_density_for_timestep_sampling_scm(
                            batch_size=bsz,
                            logit_mean=args.logit_mean_discriminator,
                            logit_std=args.logit_std_discriminator,
                        ).to(accelerator.device)
                        t_D_shifted = timesteps_D_shifted.view(-1, 1, 1, 1)

                        # Add noise to shifted pairs
                        z_D_shifted = torch.randn_like(shifted_x0) * sigma_data
                        noised_shifted_x0 = torch.cos(t_D_shifted) * shifted_x0 + torch.sin(t_D_shifted) * z_D_shifted

                        # Concatenate with original noised samples
                        noised_predicted_x0 = torch.cat([noised_predicted_x0, noised_shifted_x0], dim=0)
                        t_D_fake = torch.cat([t_D_fake, t_D_shifted], dim=0)
                        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                        prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask], dim=0)

                    # Calculate D loss

                    pred_fake = disc(
                        hidden_states=(noised_predicted_x0 / sigma_data),
                        timestep=t_D_fake.flatten(),
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                    )
                    pred_true = disc(
                        hidden_states=(noised_latents / sigma_data),
                        timestep=t_D_real.flatten(),
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                    )

                    # hinge loss
                    loss_real = torch.mean(F.relu(1.0 - pred_true))
                    loss_gen = torch.mean(F.relu(1.0 + pred_fake))
                    loss_D = 0.5 * (loss_real + loss_gen)

                    loss_D = loss_D / args.gradient_accumulation_steps

                    accelerator.backward(loss_D)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(disc.parameters(), args.gradient_clip)
                        if torch.logical_or(grad_norm.isnan(), grad_norm.isinf()):
                            optimizer_G.zero_grad(set_to_none=True)
                            optimizer_D.zero_grad(set_to_none=True)
                            logger.warning("NaN or Inf detected in grad_norm, skipping iteration...")
                            continue

                        # switch back to phase G and add global step by one.
                        phase = "G"

                        optimizer_D.step()
                        optimizer_D.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
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

            logs = {
                "scm_loss": loss.detach().item(),
                "adv_loss": adv_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = SanaSprintPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=accelerator.unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=torch.float32,
                )
                pipeline_args = {
                    "prompt": args.validation_prompt,
                    "complex_human_instruction": COMPLEX_HUMAN_INSTRUCTION,
                }
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                )
                free_memory()

                images = None
                del pipeline

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)

        # Save discriminator heads
        disc = unwrap_model(disc)
        disc_heads_state_dict = disc.heads.state_dict()

        # Save transformer model
        transformer.save_pretrained(os.path.join(args.output_dir, "transformer"))

        # Save discriminator heads
        torch.save(disc_heads_state_dict, os.path.join(args.output_dir, "disc_heads.pt"))

        # Final inference
        # Load previous pipeline
        pipeline = SanaSprintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=accelerator.unwrap_model(transformer),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=torch.float32,
        )

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {
                "prompt": args.validation_prompt,
                "complex_human_instruction": COMPLEX_HUMAN_INSTRUCTION,
            }
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
