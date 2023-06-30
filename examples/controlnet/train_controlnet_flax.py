#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset, load_from_disk
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxControlNetModel,
    FlaxDDPMScheduler,
    FlaxStableDiffusionControlNetPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available


# To prevent an error that occurs when there are abnormally large compressed data chunk in the png image
# see more https://github.com/python-pillow/Pillow/issues/5610
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = logging.getLogger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(pipeline, pipeline_params, controlnet_params, tokenizer, args, rng, weight_dtype):
    logger.info("Running validation...")

    pipeline_params = pipeline_params.copy()
    pipeline_params["controlnet"] = controlnet_params

    num_samples = jax.device_count()
    prng_seed = jax.random.split(rng, jax.device_count())

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        prompts = num_samples * [validation_prompt]
        prompt_ids = pipeline.prepare_text_inputs(prompts)
        prompt_ids = shard(prompt_ids)

        validation_image = Image.open(validation_image).convert("RGB")
        processed_image = pipeline.prepare_image_inputs(num_samples * [validation_image])
        processed_image = shard(processed_image)
        images = pipeline(
            prompt_ids=prompt_ids,
            image=processed_image,
            params=pipeline_params,
            prng_seed=prng_seed,
            num_inference_steps=50,
            jit=True,
        ).images

        images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
        images = pipeline.numpy_to_pil(images)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    if args.report_to == "wandb":
        formatted_images = []
        for log in image_logs:
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]

            formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
            for image in images:
                image = wandb.Image(image, caption=validation_prompt)
                formatted_images.append(image)

        wandb.log({"validation": formatted_images})
    else:
        logger.warn(f"image logging not implemented for {args.report_to}")

    return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
- jax-diffusers-event
inference: true
---
    """
    model_card = f"""
# controlnet- {repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--from_pt",
        action="store_true",
        help="Load the pretrained model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--controlnet_revision",
        type=str,
        default=None,
        help="Revision of controlnet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=0,
        help="How many training steps to profile in the beginning.",
    )
    parser.add_argument(
        "--profile_validation",
        action="store_true",
        help="Whether to profile the (last) validation.",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Whether to dump an initial (before training loop) and a final (at program end) memory profile.",
    )
    parser.add_argument(
        "--ccache",
        type=str,
        default=None,
        help="Enables compilation cache.",
    )
    parser.add_argument(
        "--controlnet_from_pt",
        action="store_true",
        help="Load the controlnet model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/{timestamp}",
        help="The output directory where the model predictions and checkpoints will be written. "
        "Can contain placeholders: {timestamp}.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
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
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=("Save a checkpoint of the training state every X updates."),
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
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
        "--logging_steps",
        type=int,
        default=100,
        help=("log training metric every X steps to `--report_t`"),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=('The integration to report the results and logs to. Currently only supported platforms are `"wandb"`'),
    )
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument("--streaming", action="store_true", help="To stream a large dataset from Hub.")
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training dataset. By default it will use `load_dataset` method to load a custom dataset from the folder."
            "Folder must contain a dataset script as described here https://huggingface.co/docs/datasets/dataset_script) ."
            "If `--load_from_disk` flag is passed, it will use `load_from_disk` method instead. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        help=(
            "If True, will load a dataset that was previously saved using `save_to_disk` from `--train_data_dir`"
            "See more https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.load_from_disk"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set. Needed if `streaming` is set to True."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` and logging the images."
        ),
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help=("The wandb entity to use (for teams)."))
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet_flax",
        help=("The `project` argument passed to wandb"),
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients over"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    args.output_dir = args.output_dir.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S"))

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    # This idea comes from
    # https://github.com/borisdayma/dalle-mini/blob/d2be512d4a6a9cda2d63ba04afc33038f98f705f/src/dalle_mini/data.py#L370
    if args.streaming and args.max_train_samples is None:
        raise ValueError("You must specify `max_train_samples` when using dataset streaming.")

    return args


def make_train_dataset(args, tokenizer, batch_size=None):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
    else:
        if args.train_data_dir is not None:
            if args.load_from_disk:
                dataset = load_from_disk(
                    args.train_data_dir,
                )
            else:
                dataset = load_dataset(
                    args.train_data_dir,
                    cache_dir=args.cache_dir,
                )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if isinstance(dataset["train"], IterableDataset):
        column_names = next(iter(dataset["train"])).keys()
    else:
        column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {caption_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    if jax.process_index() == 0:
        if args.max_train_samples is not None:
            if args.streaming:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).take(args.max_train_samples)
            else:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        if args.streaming:
            train_dataset = dataset["train"].map(
                preprocess_train,
                batched=True,
                batch_size=batch_size,
                remove_columns=list(dataset["train"].features.keys()),
            )
        else:
            train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    batch = {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }
    batch = {k: v.numpy() for k, v in batch.items()}
    return batch


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb init
    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.init(
            entity=args.wandb_entity,
            project=args.tracker_project_name,
            job_type="train",
            config=args,
        )

    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(0)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
    else:
        raise NotImplementedError("No tokenizer specified!")

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    total_train_batch_size = args.train_batch_size * jax.local_device_count() * args.gradient_accumulation_steps
    train_dataset = make_train_dataset(args, tokenizer, batch_size=total_train_batch_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=not args.streaming,
        collate_fn=collate_fn,
        batch_size=total_train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        subfolder="vae",
        dtype=weight_dtype,
        from_pt=args.from_pt,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            revision=args.controlnet_revision,
            from_pt=args.controlnet_from_pt,
            dtype=jnp.float32,
        )
    else:
        logger.info("Initializing controlnet weights from unet")
        rng, rng_params = jax.random.split(rng)

        controlnet = FlaxControlNetModel(
            in_channels=unet.config.in_channels,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            attention_head_dim=unet.config.attention_head_dim,
            cross_attention_dim=unet.config.cross_attention_dim,
            use_linear_projection=unet.config.use_linear_projection,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
        )
        controlnet_params = controlnet.init_weights(rng=rng_params)
        controlnet_params = unfreeze(controlnet_params)
        for key in [
            "conv_in",
            "time_embedding",
            "down_blocks_0",
            "down_blocks_1",
            "down_blocks_2",
            "down_blocks_3",
            "mid_block",
        ]:
            controlnet_params[key] = unet_params[key]

    pipeline, pipeline_params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        controlnet=controlnet,
        safety_checker=None,
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    pipeline_params = jax_utils.replicate(pipeline_params)

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    state = train_state.TrainState.create(apply_fn=controlnet.__call__, params=controlnet_params, tx=optimizer)

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Initialize our training
    validation_rng, train_rngs = jax.random.split(rng)
    train_rngs = jax.random.split(train_rngs, jax.local_device_count())

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler_state.common.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        alpha = sqrt_alphas_cumprod[timesteps]
        sigma = sqrt_one_minus_alphas_cumprod[timesteps]
        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def train_step(state, unet_params, text_encoder_params, vae_params, batch, train_rng):
        # reshape batch, add grad_step_dim if gradient_accumulation_steps > 1
        if args.gradient_accumulation_steps > 1:
            grad_steps = args.gradient_accumulation_steps
            batch = jax.tree_map(lambda x: x.reshape((grad_steps, x.shape[0] // grad_steps) + x.shape[1:]), batch)

        def compute_loss(params, minibatch, sample_rng):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, minibatch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                minibatch["input_ids"],
                params=text_encoder_params,
                train=False,
            )[0]

            controlnet_cond = minibatch["conditioning_pixel_values"]

            # Predict the noise residual and compute loss
            down_block_res_samples, mid_block_res_sample = controlnet.apply(
                {"params": params},
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                controlnet_cond,
                train=True,
                return_dict=False,
            )

            model_pred = unet.apply(
                {"params": unet_params},
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = (target - model_pred) ** 2

            if args.snr_gamma is not None:
                snr = jnp.array(compute_snr(timesteps))
                snr_loss_weights = jnp.where(snr < args.snr_gamma, snr, jnp.ones_like(snr) * args.snr_gamma) / snr
                loss = loss * snr_loss_weights

            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)

        # get a minibatch (one gradient accumulation slice)
        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_index_in_dim(x, grad_idx, keepdims=False),
                batch,
            )

        def loss_and_grad(grad_idx, train_rng):
            # create minibatch for the grad step
            minibatch = get_minibatch(batch, grad_idx) if grad_idx is not None else batch
            sample_rng, train_rng = jax.random.split(train_rng, 2)
            loss, grad = grad_fn(state.params, minibatch, sample_rng)
            return loss, grad, train_rng

        if args.gradient_accumulation_steps == 1:
            loss, grad, new_train_rng = loss_and_grad(None, train_rng)
        else:
            init_loss_grad_rng = (
                0.0,  # initial value for cumul_loss
                jax.tree_map(jnp.zeros_like, state.params),  # initial value for cumul_grad
                train_rng,  # initial value for train_rng
            )

            def cumul_grad_step(grad_idx, loss_grad_rng):
                cumul_loss, cumul_grad, train_rng = loss_grad_rng
                loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
                cumul_loss, cumul_grad = jax.tree_map(jnp.add, (cumul_loss, cumul_grad), (loss, grad))
                return cumul_loss, cumul_grad, new_train_rng

            loss, grad, new_train_rng = jax.lax.fori_loop(
                0,
                args.gradient_accumulation_steps,
                cumul_grad_step,
                init_loss_grad_rng,
            )
            loss, grad = jax.tree_map(lambda x: x / args.gradient_accumulation_steps, (loss, grad))

        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        def l2(xs):
            return jnp.sqrt(sum([jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(xs)]))

        metrics["l2_grads"] = l2(jax.tree_util.tree_leaves(grad))

        return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)
    unet_params = jax_utils.replicate(unet_params)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

    # Train!
    if args.streaming:
        dataset_length = args.max_train_samples
    else:
        dataset_length = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(dataset_length / args.gradient_accumulation_steps)

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {args.max_train_samples if args.streaming else len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs * num_update_steps_per_epoch}")

    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")
        wandb.config.update(
            {
                "num_train_examples": args.max_train_samples if args.streaming else len(train_dataset),
                "total_train_batch_size": total_train_batch_size,
                "total_optimization_step": args.num_train_epochs * num_update_steps_per_epoch,
                "num_devices": jax.device_count(),
                "controlnet_params": sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params)),
            }
        )

    global_step = step0 = 0
    epochs = tqdm(
        range(args.num_train_epochs),
        desc="Epoch ... ",
        position=0,
        disable=jax.process_index() > 0,
    )
    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.output_dir, "memory_initial.prof"))
    t00 = t0 = time.monotonic()
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []
        train_metric = None

        steps_per_epoch = (
            args.max_train_samples // total_train_batch_size
            if args.streaming or args.max_train_samples
            else len(train_dataset) // total_train_batch_size
        )
        train_step_progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Training...",
            position=1,
            leave=False,
            disable=jax.process_index() > 0,
        )
        # train
        for batch in train_dataloader:
            if args.profile_steps and global_step == 1:
                train_metric["loss"].block_until_ready()
                jax.profiler.start_trace(args.output_dir)
            if args.profile_steps and global_step == 1 + args.profile_steps:
                train_metric["loss"].block_until_ready()
                jax.profiler.stop_trace()

            batch = shard(batch)
            with jax.profiler.StepTraceAnnotation("train", step_num=global_step):
                state, train_metric, train_rngs = p_train_step(
                    state, unet_params, text_encoder_params, vae_params, batch, train_rngs
                )
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= args.max_train_steps:
                break

            if (
                args.validation_prompt is not None
                and global_step % args.validation_steps == 0
                and jax.process_index() == 0
            ):
                _ = log_validation(
                    pipeline, pipeline_params, state.params, tokenizer, args, validation_rng, weight_dtype
                )

            if global_step % args.logging_steps == 0 and jax.process_index() == 0:
                if args.report_to == "wandb":
                    train_metrics = jax_utils.unreplicate(train_metrics)
                    train_metrics = jax.tree_util.tree_map(lambda *m: jnp.array(m).mean(), *train_metrics)
                    wandb.log(
                        {
                            "walltime": time.monotonic() - t00,
                            "train/step": global_step,
                            "train/epoch": global_step / dataset_length,
                            "train/steps_per_sec": (global_step - step0) / (time.monotonic() - t0),
                            **{f"train/{k}": v for k, v in train_metrics.items()},
                        }
                    )
                t0, step0 = time.monotonic(), global_step
                train_metrics = []
            if global_step % args.checkpointing_steps == 0 and jax.process_index() == 0:
                controlnet.save_pretrained(
                    f"{args.output_dir}/{global_step}",
                    params=get_params_to_save(state.params),
                )

        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    # Final validation & store model.
    if jax.process_index() == 0:
        if args.validation_prompt is not None:
            if args.profile_validation:
                jax.profiler.start_trace(args.output_dir)
            image_logs = log_validation(
                pipeline, pipeline_params, state.params, tokenizer, args, validation_rng, weight_dtype
            )
            if args.profile_validation:
                jax.profiler.stop_trace()
        else:
            image_logs = None

        controlnet.save_pretrained(
            args.output_dir,
            params=get_params_to_save(state.params),
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.output_dir, "memory_final.prof"))
    logger.info("Finished training.")


if __name__ == "__main__":
    main()
