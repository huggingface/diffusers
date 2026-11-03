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

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.11.1",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux3Transformer2DModel,
)
from diffusers.loaders import Flux3LoraLoaderMixin
from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.training_utils import (
    _collate_lora_metadata,
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    generate_aspect_ratio_buckets,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    parse_buckets_string,
    wrap_with_fsdp,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.40.0.dev0")

logger = get_logger(__name__)

# FLUX.3 ships no `scheduler` subfolder: the modular pipeline builds its scheduler `from_config` with a fixed
# shift of 6.93 (the `Flux3SetTimestepsStep` default, and the reference sampler's `alpha`). There is no
# resolution-dependent dynamic shifting. Constructed here for the same reason, rather than loaded.
FLOW_SHIFT = 6.93

# FLUX.3 image RoPE scale applied to the H/W position-id axes (see the flux3 modular pipeline).
IMAGE_ROPE_SCALE = 0.5

# The text window is right-padded to a multiple of this many tokens (the `Flux3TextEncoderStep` default).
TEXT_PAD_MULTIPLE = 80


# --- FLUX.3 latent / text helpers (bit-identical to the modular pipeline blocks) ---
def patchify_latents(latents):
    """(B, C, H, W) -> (B, C*4, H//2, W//2)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def pack_latents(latents):
    """(B, C, H, W) -> (B, H*W, C)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def prepare_latent_ids(height, width, device, image_rope_scale=IMAGE_ROPE_SCALE):
    """(H*W, 4) position ids (t, h, w, l) for the generated image; H/W axes scaled by the image RoPE scale."""
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    ids = ids.to(dtype=torch.float32)
    ids[:, 1:3] = ids[:, 1:3] * image_rope_scale
    return ids.to(device)


def prepare_image_cond_ids(height, width, device, image_rope_scale=IMAGE_ROPE_SCALE):
    """(H*W, 4) position ids for the conditioning image: reference sits at the L axis position -1; H/W scaled."""
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([-1]))
    ids = ids.to(dtype=torch.float32)
    ids[:, 1:3] = ids[:, 1:3] * image_rope_scale
    return ids.to(device)


def prepare_text_ids(seq_len, device):
    """(seq_len, 4) position ids for text tokens; text advances along the L axis."""
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len))
    return ids.to(dtype=torch.float32).to(device)


@torch.no_grad()
def get_qwen3vl_prompt_embeds(
    text_encoder, tokenizer, prompt, device, dtype, text_encoder_out_layers, max_sequence_length, pad_to_length=None
):
    """Stack the FLUX.3 Qwen3-VL hidden layers -> (B, seq, len(layers) * hidden_dim).

    Mirrors `Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds`: each prompt goes through the chat template and is
    right-padded to a multiple of `TEXT_PAD_MULTIPLE` (capped at `max_sequence_length`). Qwen3-VL masks the padding
    (so a prompt's real-token embeddings are independent of the window size), but the transformer attends to it, so
    the window is part of the recipe -- for a single prompt this reproduces the pipeline window exactly. When a batch
    of differing prompts is encoded together, they share the batch's largest window so the embeddings can be stacked;
    with `train_batch_size=1` or a shared instance prompt every prompt keeps its own pipeline-exact window.
    The position ids of the padding tokens are passed explicitly to match the Qwen3-VL 4.57 training environment.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompt
    ]
    real_lengths = [
        tokenizer.tokenizer(t, return_tensors="pt", truncation=True, max_length=max_sequence_length)[
            "input_ids"
        ].shape[1]
        for t in texts
    ]
    padded_length = min(math.ceil(max(real_lengths) / TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE, max_sequence_length)
    if pad_to_length is not None:
        # Force a specific window (caption dropout encodes the empty prompt at a dropped instruction's window). Never
        # shrink below the prompt's own real length, never exceed max_sequence_length.
        padded_length = min(max(padded_length, pad_to_length), max_sequence_length)
    enc = tokenizer.tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=padded_length,
    )
    input_ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    position_ids = torch.arange(padded_length, device=device)[None].masked_fill(attn == 0, 1)
    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attn,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack([out.hidden_states[k] for k in text_encoder_out_layers], dim=1).to(
        dtype=dtype, device=device
    )
    b, n, s, d = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(b, s, n * d)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    quant_training=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# FLUX.3 DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [FLUX.3 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux3_klein.md).

Quant training? {quant_training}

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import ModularPipeline
import torch
pipeline = ModularPipeline.from_pretrained("{base_model}")
pipeline.load_components(torch_dtype=torch.bfloat16)
pipeline.to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline(prompt='{validation_prompt if validation_prompt else instance_prompt}', output_type="pil").values["images"][0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described in the model repository.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux3",
        "flux3-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    args.num_validation_images = args.num_validation_images if args.num_validation_images else 1
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            output = pipeline(generator=generator, output_type="pil", **pipeline_args)
            images.append(output.values["images"][0])

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
    free_memory()

    return images


def make_validation_pipeline(args, transformer, weight_dtype, device):
    # Build a modular pipeline for validation that REUSES the in-memory trained transformer, so validation runs
    # on exactly the weights being trained and the DiT is never loaded a second time. Only the small auxiliary
    # components are (re)loaded (they are freed after latent caching).
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=torch.float32
    )
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype
    )
    tokenizer = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT, use_dynamic_shifting=False)
    pipeline = Flux3AutoBlocks().init_pipeline()
    pipeline.update_components(
        transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler
    )
    pipeline.to(device)
    return pipeline


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module (FLUX.3's output projection)
    if fqn.endswith("proj_out"):
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


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
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
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
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--caption_dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of replacing a sample's instruction with an empty string during training, so that fraction"
            " of samples is trained unconditionally on the text (the conditioning image is kept). Improves"
            " classifier-free guidance. A common value is 0.1."
        ),
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help=(
            "Cap on the number of Qwen3-VL prompt tokens; longer prompts are truncated. The text window is the "
            f"prompt's length rounded up to a multiple of {TEXT_PAD_MULTIPLE}, capped at this value."
        ),
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24, 28, 32],
        help="Text encoder hidden layers to compute the final text embeddings.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="path to an image that is used during validation as the condition image to verify that the model is learning.",
    )
    parser.add_argument(
        "--skip_final_inference",
        default=False,
        action="store_true",
        help="Whether to skip the final inference step with loaded lora weights upon training completion. This will run intermediate validation inference if `validation_prompt` is provided. Specify to reduce memory.",
    )
    parser.add_argument(
        "--final_validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during a final validation to verify that the model is learning. Ignored if `--validation_prompt` is provided.",
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
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'. "
            "Requires --use_aspect_ratio_buckets. Images are resized to cover and cropped to the nearest "
            "listed bucket (smaller images are upscaled). When set, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--use_aspect_ratio_buckets",
        action="store_true",
        help=(
            "Enable aspect-ratio bucketing. Without --aspect_ratio_buckets, the buckets are computed on the "
            "fly from --resolution and capped to each image's own resolution, so smaller images are assigned "
            "to a smaller bucket instead of being upscaled. Provide --aspect_ratio_buckets to use an explicit list."
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
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
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
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
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
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

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
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--fsdp_text_encoder", action="store_true", help="Use FSDP for text encoder")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.cond_image_column is None:
        raise ValueError(
            "you must provide --cond_image_column for image-to-image training. Otherwise use the FLUX.3 text-to-image training script (train_dreambooth_lora_flux3_klein.py)."
        )
    else:
        assert args.image_column is not None
        assert args.caption_column is not None

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        size=1024,
        repeats=1,
        center_crop=False,
        buckets=None,
        use_aspect_ratio_buckets=False,
        bucket_divisibility=16,
        bucket_base_resolutions=None,
    ):
        self.size = size
        self.resolution = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None

        # Explicit user-provided bucket list (or None). The concrete list of buckets actually used is
        # built from the data in `self.buckets` during preprocessing below.
        self._explicit_buckets = buckets
        self.use_aspect_ratio_buckets = use_aspect_ratio_buckets
        self.bucket_divisibility = bucket_divisibility
        self.bucket_base_resolutions = bucket_base_resolutions

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.cond_image_column is not None and args.cond_image_column not in column_names:
                raise ValueError(
                    f"`--cond_image_column` value '{args.cond_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]
            cond_images = None
            cond_image_column = args.cond_image_column
            if cond_image_column is not None:
                cond_images = [dataset["train"][i][cond_image_column] for i in range(len(dataset["train"]))]
                assert len(instance_images) == len(cond_images)

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        self.cond_images = []
        for i, img in enumerate(instance_images):
            self.instance_images.extend(itertools.repeat(img, repeats))
            if args.dataset_name is not None and cond_images is not None:
                self.cond_images.extend(itertools.repeat(cond_images[i], repeats))

        self.pixel_values = []
        self.cond_pixel_values = []
        self.buckets = []
        bucket_to_idx = {}
        for i, image in enumerate(self.instance_images):
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            dest_image = None
            if self.cond_images:
                dest_image = self.cond_images[i]
                image_width, image_height = dest_image.size
                if image_width * image_height > 1024 * 1024:
                    dest_image = Flux2ImageProcessor._resize_to_target_area(dest_image, 1024 * 1024)
                    image_width, image_height = dest_image.size

                # VAE spatial compression: 2 ** (len(vae.config.block_out_channels) - 1) = 8 for the Flux2 VAE
                multiple_of = 2 ** (4 - 1)
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                image_processor = Flux2ImageProcessor()
                dest_image = image_processor.preprocess(
                    dest_image, height=image_height, width=image_width, resize_mode="crop"
                )
                # Convert back to PIL
                dest_image = dest_image.squeeze(0)
                if dest_image.min() < 0:
                    dest_image = (dest_image + 1) / 2
                dest_image = (torch.clamp(dest_image, 0, 1) * 255).byte().cpu()

                if dest_image.shape[0] == 1:
                    # Gray scale image
                    dest_image = Image.fromarray(dest_image.squeeze().numpy(), mode="L")
                else:
                    # RGB scale image: (C, H, W) -> (H, W, C)
                    dest_image = TF.to_pil_image(dest_image)

                dest_image = exif_transpose(dest_image)
                if not dest_image.mode == "RGB":
                    dest_image = dest_image.convert("RGB")

            width, height = image.size

            # Assign the image to a bucket.
            target = self._bucket_for_image(height, width)
            if target not in bucket_to_idx:
                bucket_to_idx[target] = len(self.buckets)
                self.buckets.append(target)
            bucket_idx = bucket_to_idx[target]

            # based on the bucket assignment, define the transformations
            image, dest_image = self.paired_transform(
                image,
                dest_image=dest_image,
                size=target,
                center_crop=args.center_crop,
                random_flip=args.random_flip,
            )
            self.pixel_values.append((image, bucket_idx))
            if dest_image is not None:
                self.cond_pixel_values.append((dest_image, bucket_idx))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["bucket_idx"] = bucket_idx
        if self.cond_pixel_values:
            dest_image, _ = self.cond_pixel_values[index % self.num_instance_images]
            example["cond_images"] = dest_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        return example

    def _bucket_for_image(self, height, width):
        # An explicit bucket list takes priority: pick the nearest, upscaling smaller images to cover it.
        if self._explicit_buckets is not None:
            return self._explicit_buckets[find_nearest_bucket(height, width, self._explicit_buckets)]
        # On-the-fly bucketing: cap the ladder to the image's own resolution so smaller images are
        # assigned to a smaller bucket rather than being upscaled (mirrors ostris' bucketing).
        if self.use_aspect_ratio_buckets:
            resolution = min(self.resolution, round((height * width) ** 0.5))
            ladder = generate_aspect_ratio_buckets(
                resolution,
                divisibility=self.bucket_divisibility,
                base_resolutions=self.bucket_base_resolutions,
            )
            return ladder[find_nearest_bucket(height, width, ladder)]
        # No bucketing: a single square bucket reproduces the fixed-size resize + crop.
        return (self.resolution, self.resolution)

    def paired_transform(self, image, dest_image=None, size=(224, 224), center_crop=False, random_flip=False):
        # Resize preserving aspect ratio so the image covers the bucket, then crop to the bucket size.
        # The same geometry is applied to the conditioning image so the pair stays aligned.
        target_height, target_width = size
        width, height = image.size
        scale = max(target_height / height, target_width / width)
        new_size = [round(height * scale), round(width * scale)]
        image = TF.resize(image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        if dest_image is not None:
            dest_image = TF.resize(dest_image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        if center_crop:
            image = TF.center_crop(image, size)
            if dest_image is not None:
                dest_image = TF.center_crop(dest_image, size)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)
            if dest_image is not None:
                dest_image = TF.crop(dest_image, i, j, h, w)
        if random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            if dest_image is not None:
                dest_image = TF.hflip(dest_image)
        image = TF.normalize(TF.to_tensor(image), [0.5], [0.5])
        if dest_image is not None:
            dest_image = TF.normalize(TF.to_tensor(dest_image), [0.5], [0.5])
        return (image, dest_image) if dest_image is not None else (image, None)


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    if any("cond_images" in example for example in examples):
        cond_pixel_values = [example["cond_images"] for example in examples]
        cond_pixel_values = torch.stack(cond_pixel_values)
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch.update({"cond_pixel_values": cond_pixel_values})
    return batch


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: DreamBoothDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle_batches_each_epoch: bool = True,
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_batches_each_epoch = shuffle_batches_each_epoch

        # Group indices by bucket
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        for idx, (_, bucket_idx) in enumerate(self.dataset.pixel_values):
            self.bucket_indices[bucket_idx].append(idx)

        self.sampler_len = 0
        self.batches = []

        # Pre-generate batches for each bucket
        for indices_in_bucket in self.bucket_indices:
            # Shuffle indices within the bucket
            random.shuffle(indices_in_bucket)
            # Create batches
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                self.batches.append(batch)
                self.sampler_len += 1  # Count the number of batches

        if not self.shuffle_batches_each_epoch:
            # Shuffle the precomputed batches once to mix buckets while keeping
            # the order stable across epochs for step-indexed caches.
            random.shuffle(self.batches)

    def __iter__(self):
        if self.shuffle_batches_each_epoch:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len


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
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

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

    # Load the tokenizers
    tokenizer = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT, use_dynamic_shifting=False)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = Flux3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    text_encoder.to(**to_kwargs)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # Attention + feed-forward projections across every block (the generated-image, reference and text mode
        # blocks, and the joint single blocks). These are exactly the leaf modules the non-diffusers (ostris)
        # FLUX.3 LoRA converter maps, so adapters trained here stay interchangeable with that layout. The reference
        # ("image_cond") mode blocks are adapted too, so the LoRA can adjust how the condition image is consumed.
        # Pass `--lora_layers` to target a different set.
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.linear_in",
            "ff.linear_out",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        transformer_cls = type(unwrap_model(transformer))

        # 1) Validate and pick the transformer model
        modules_to_save: dict[str, Any] = {}
        transformer_model = None

        for model in models:
            if isinstance(unwrap_model(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        # 2) Optionally gather FSDP state dict once
        state_dict = accelerator.get_state_dict(model) if is_fsdp else None

        # 3) Only main process materializes the LoRA state dict
        transformer_lora_layers_to_save = None
        if accelerator.is_main_process:
            peft_kwargs = {}
            if is_fsdp:
                peft_kwargs["state_dict"] = state_dict

            transformer_lora_layers_to_save = get_peft_model_state_dict(
                unwrap_model(transformer_model) if is_fsdp else transformer_model,
                **peft_kwargs,
            )

            if is_fsdp:
                transformer_lora_layers_to_save = _to_cpu_contiguous(transformer_lora_layers_to_save)

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

            Flux3LoraLoaderMixin.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not is_fsdp:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux3Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = Flux3LoraLoaderMixin.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

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

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
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

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Resolve the bucketing mode. Bucketing must be enabled explicitly with --use_aspect_ratio_buckets;
    # a bucket list without that flag is an error. With the flag, an explicit --aspect_ratio_buckets list
    # drives assignment, otherwise buckets are computed on the fly inside the dataset. Without the flag a
    # single square bucket reproduces the fixed-size resize + crop.
    if args.aspect_ratio_buckets is not None and not args.use_aspect_ratio_buckets:
        raise ValueError("--aspect_ratio_buckets requires --use_aspect_ratio_buckets to be set.")
    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
        use_aspect_ratio_buckets = False
        logger.info(f"Using explicit aspect ratio buckets: {buckets}")
    elif args.use_aspect_ratio_buckets:
        buckets = None
        use_aspect_ratio_buckets = True
        logger.info(
            "No --aspect_ratio_buckets provided; auto-computing aspect ratio buckets on the fly from --resolution."
        )
    else:
        buckets = [(args.resolution, args.resolution)]
        use_aspect_ratio_buckets = False

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        buckets=buckets,
        use_aspect_ratio_buckets=use_aspect_ratio_buckets,
    )
    has_step_indexed_caches = precompute_latents = args.cache_latents or train_dataset.custom_instance_prompts
    batch_sampler = BucketBatchSampler(
        train_dataset,
        batch_size=args.train_batch_size,
        drop_last=True,
        shuffle_batches_each_epoch=not has_step_indexed_caches,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoder, tokenizer, pad_to_length=None):
        # FLUX.3 has no classic text-encoding pipeline; encode directly with the frozen Qwen3-VL encoder.
        with torch.no_grad():
            prompt_embeds = get_qwen3vl_prompt_embeds(
                text_encoder,
                tokenizer,
                prompt,
                accelerator.device,
                weight_dtype,
                text_encoder_out_layers=args.text_encoder_out_layers,
                max_sequence_length=args.max_sequence_length,
                pad_to_length=pad_to_length,
            )
            # FLUX.3 text ids are positional only (identical for every sample of a given length). We carry them
            # batched here so the repeat plumbing below is unchanged; the transformer consumes a single (seq, 4)
            # slice at the forward call.
            text_ids = prepare_text_ids(prompt_embeds.shape[1], accelerator.device)
            text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
        return prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not train_dataset.custom_instance_prompts:
        with offload_models(text_encoder, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoder, tokenizer
            )

    # FLUX.3 validation runs through the modular pipeline (which encodes prompts itself), so we only assemble
    # the generation kwargs here. The reference/condition image is routed to the pipeline's `image` input.
    validation_pipeline_args = {}
    if args.validation_prompt is not None:
        validation_image = load_image(args.validation_image).convert("RGB")
        validation_pipeline_args = {
            "prompt": args.validation_prompt,
            "image": validation_image,
            "negative_prompt": "",
            "height": args.resolution,
            "width": args.resolution,
            "num_inference_steps": 28,
            # Encode the validation prompt exactly as training did (same window cap and hidden layers), so the
            # modular pipeline's text embeddings match the ones the adapter was trained against.
            "max_sequence_length": args.max_sequence_length,
            "text_encoder_out_layers": tuple(args.text_encoder_out_layers),
        }

    # Init FSDP for text encoder
    if args.fsdp_text_encoder:
        fsdp_kwargs = get_fsdp_kwargs_from_accelerator(accelerator)
        text_encoder = wrap_with_fsdp(
            model=text_encoder,
            device=accelerator.device,
            offload=args.offload,
            limit_all_gathers=True,
            use_orig_params=True,
            fsdp_kwargs=fsdp_kwargs,
        )
        dist.barrier()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    if not train_dataset.custom_instance_prompts:
        prompt_embeds = instance_prompt_hidden_states
        text_ids = instance_text_ids

    # if cache_latents is set to True, we encode images to latents and store them.
    # Similar to pre-encoding in the case of a single instance prompt, if custom prompts are provided
    # we encode them in advance as well.
    if precompute_latents:
        prompt_embeds_cache = []
        text_ids_cache = []
        latents_cache = []
        cond_latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                if args.cache_latents:
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        batch["pixel_values"] = batch["pixel_values"].to(
                            accelerator.device, non_blocking=True, dtype=vae.dtype
                        )
                        latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                        batch["cond_pixel_values"] = batch["cond_pixel_values"].to(
                            accelerator.device, non_blocking=True, dtype=vae.dtype
                        )
                        cond_latents_cache.append(vae.encode(batch["cond_pixel_values"]).latent_dist)
                if train_dataset.custom_instance_prompts:
                    if args.fsdp_text_encoder:
                        prompt_embeds, text_ids = compute_text_embeddings(batch["prompts"], text_encoder, tokenizer)
                    else:
                        with offload_models(text_encoder, device=accelerator.device, offload=args.offload):
                            prompt_embeds, text_ids = compute_text_embeddings(
                                batch["prompts"], text_encoder, tokenizer
                            )
                    prompt_embeds_cache.append(prompt_embeds)
                    text_ids_cache.append(text_ids)

    # Caption dropout replaces the instruction with the empty prompt for a fraction of samples (the conditioning
    # image is kept). FLUX.3 pads each prompt to its own window (a multiple of TEXT_PAD_MULTIPLE) and the transformer
    # attends to it, so the empty prompt must match the dropped sample's window. Precompute it for every window in
    # use here, while the text encoder is still loaded (it is deleted below, so this cannot be done in the loop).
    empty_prompt_by_window = {}
    if args.caption_dropout > 0:
        dropout_windows = (
            {embeds.shape[1] for embeds in prompt_embeds_cache}
            if train_dataset.custom_instance_prompts
            else {instance_prompt_hidden_states.shape[1]}
        )
        for window in sorted(dropout_windows):
            with offload_models(text_encoder, device=accelerator.device, offload=args.offload):
                empty_prompt_by_window[window] = compute_text_embeddings(
                    "", text_encoder, tokenizer, pad_to_length=window
                )

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if args.cache_latents:
        vae = vae.to("cpu")
        del vae

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    text_encoder = text_encoder.to("cpu")
    del text_encoder, tokenizer
    free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux3-image2img-lora"
        args_cp = vars(args).copy()
        args_cp["text_encoder_out_layers"] = str(args_cp["text_encoder_out_layers"])
        accelerator.init_trackers(tracker_name, config=args_cp)

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

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            prompts = batch["prompts"]

            with accelerator.accumulate(models_to_accumulate):
                if train_dataset.custom_instance_prompts:
                    prompt_embeds = prompt_embeds_cache[step]
                    text_ids = text_ids_cache[step]
                else:
                    num_repeat_elements = len(prompts)
                    prompt_embeds = prompt_embeds.repeat(num_repeat_elements, 1, 1)
                    text_ids = text_ids.repeat(num_repeat_elements, 1, 1)

                # Caption dropout: replace the instruction with the empty prompt for a fraction of samples so they
                # train unconditionally on the text (the conditioning image is kept). Clone first so the in-place
                # write never mutates the cached embeddings.
                if args.caption_dropout > 0:
                    drop_mask = torch.rand(prompt_embeds.shape[0], device=prompt_embeds.device) < args.caption_dropout
                    if drop_mask.any():
                        empty_hidden_states, empty_text_ids = empty_prompt_by_window[prompt_embeds.shape[1]]
                        prompt_embeds = prompt_embeds.clone()
                        text_ids = text_ids.clone()
                        prompt_embeds[drop_mask] = empty_hidden_states.to(
                            device=prompt_embeds.device, dtype=prompt_embeds.dtype
                        )
                        text_ids[drop_mask] = empty_text_ids.to(device=text_ids.device, dtype=text_ids.dtype)

                # Convert images to latent space
                if args.cache_latents:
                    model_input = latents_cache[step].mode()
                    cond_model_input = cond_latents_cache[step].mode()
                else:
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                        cond_pixel_values = batch["cond_pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()

                model_input = patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std

                cond_model_input = patchify_latents(cond_model_input)
                cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std

                _, _, lh, lw = model_input.shape
                _, _, ch, cw = cond_model_input.shape

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]: the noisy generated-image tokens.
                packed_noisy_model_input = pack_latents(noisy_model_input)
                image_ids = prepare_latent_ids(lh, lw, accelerator.device)  # (image_seq_len, 4)
                # FLUX.3 is not guidance-distilled: no guidance argument. The flow timestep is a per-sample scalar in
                # [0, 1] (the transformer scales it by 1000 and broadcasts it over the image tokens internally).
                timestep = sigmas.reshape(bsz)

                # The clean reference image is fed as a SEPARATE conditioning stream: its own packed tokens and
                # position ids at the l=-1 axis. The transformer holds the reference (and the text) at timestep 0
                # internally, so no cond timestep is passed. It does not concatenate the reference onto the image
                # stream and returns only the generated image tokens, so there is nothing to slice off afterwards.
                image_cond_hidden_states = pack_latents(cond_model_input)
                image_cond_ids = prepare_image_cond_ids(ch, cw, accelerator.device)  # (cond_seq_len, 4)

                # Predict the noise residual. The transformer returns only the generated image tokens, so there is
                # nothing to slice/unpack afterwards.
                with accelerator.autocast():
                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
                        img_ids=image_ids,  # (image_seq_len, 4)
                        timestep=timestep,  # (B,)
                        image_cond_hidden_states=image_cond_hidden_states,  # (B, cond_seq_len, C)
                        image_cond_ids=image_cond_ids,  # (cond_seq_len, 4)
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids[0],  # (text_seq_len, 4), shared across the batch
                        return_dict=False,
                    )[0]

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas.reshape(bsz, 1, 1)
                )

                # flow matching loss (velocity target) on the packed image tokens
                target = pack_latents(noise) - pack_latents(model_input)

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or is_fsdp:
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

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                transformer_ = unwrap_model(transformer)
                transformer_.eval()
                pipeline = make_validation_pipeline(args, transformer_, weight_dtype, accelerator.device)
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )
                transformer_.train()
                free_memory()

    # Save the lora layers
    accelerator.wait_for_everyone()

    if is_fsdp:
        transformer = unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process:
        modules_to_save = {}
        if is_fsdp:
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    state_dict = {
                        k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }
                else:
                    state_dict = {
                        k: v.to(weight_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }

            transformer_lora_layers = get_peft_model_state_dict(
                transformer,
                state_dict=state_dict,
            )
            transformer_lora_layers = {
                k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in transformer_lora_layers.items()
            }

        else:
            transformer = unwrap_model(transformer)
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    transformer.to(torch.float32)
                else:
                    transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)

        modules_to_save["transformer"] = transformer

        Flux3LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        images = []
        run_validation = (args.validation_prompt and args.num_validation_images > 0) or (args.final_validation_prompt)
        should_run_final_inference = not args.skip_final_inference and run_validation
        if should_run_final_inference:
            pipeline = make_validation_pipeline(args, transformer, weight_dtype, accelerator.device)

            # run inference
            images = []
            if args.validation_prompt and args.num_validation_images > 0:
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_pipeline_args,
                    epoch=epoch,
                    is_final_validation=True,
                    torch_dtype=weight_dtype,
                )
            del pipeline
            free_memory()

        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt
        quant_training = None
        if args.do_fp8_training:
            quant_training = "FP8 TorchAO"
        elif args.bnb_quantization_config_path:
            quant_training = "BitsandBytes"
        save_model_card(
            (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            instance_prompt=args.instance_prompt,
            validation_prompt=validation_prompt,
            repo_folder=args.output_dir,
            quant_training=quant_training,
        )

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
