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
import contextlib
import gc
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from taming.modules.losses.vqperceptual import NLayerDiscriminator, hinge_d_loss, vanilla_d_loss, weights_init
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


@torch.no_grad()
def log_validation(vae, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        vae = accelerator.unwrap_model(vae)
    else:
        vae = AutoencoderKL.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    images = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    for i, validation_image in enumerate(args.validation_image):
        validation_image = Image.open(validation_image).convert("RGB")
        targets = image_transforms(validation_image).to(accelerator.device, weight_dtype)
        targets = targets.unsqueeze(0)

        with inference_ctx:
            reconstructions = vae(targets).sample

        images.append(torch.cat([targets.cpu(), reconstructions.cpu()], axis=0))

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(f"{tracker_key}: Original (left), Reconstruction (right)", np_images, step)
        elif tracker.name == "wandb":
            tracker.log(
                {
                    f"{tracker_key}: Original (left), Reconstruction (right)": [
                        wandb.Image(torchvision.utils.make_grid(image)) for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        gc.collect()
        torch.cuda.empty_cache()

    return images


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    if images is not None:
        img_str = "You can find some example images below.\n\n"
        make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, "images.png"))
        img_str += "![images](./images.png)\n"

    model_description = f"""
# autoencoderkl-{repo_id}

These are autoencoderkl weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "image-to-image",
        "diffusers",
        "autoencoderkl",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a AutoencoderKL training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the VAE model to train, leave as None to use standard VAE model configuration.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="autoencoderkl-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
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
        default=4.5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--disc_learning_rate",
        type=float,
        default=4.5e-6,
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
        "--disc_lr_scheduler",
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
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
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
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
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help="A set of paths to the image be evaluated every `--validation_steps` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_autoencoderkl",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--rec_loss",
        type=str,
        default="l2",
        help="The loss function for VAE reconstruction loss.",
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    )
    parser.add_argument(
        "--perceptual_scale",
        type=float,
        default=0.5,
        help="Scaling factor for the LPIPS metric",
    )
    parser.add_argument(
        "--disc_start",
        type=int,
        default=50001,
        help="Start for the discriminator",
    )
    parser.add_argument(
        "--disc_factor",
        type=float,
        default=1.0,
        help="Scaling factor for the discriminator",
    )
    parser.add_argument(
        "--disc_scale",
        type=float,
        default=1.0,
        help="Scaling factor for the discriminator",
    )
    parser.add_argument(
        "--disc_loss",
        type=str,
        default="hinge",
        help="Loss function for the discriminator",
    )
    parser.add_argument(
        "--decoder_only",
        action="store_true",
        help="Only train the VAE decoder.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.pretrained_model_name_or_path is not None and args.model_config_name_or_path is not None:
        raise ValueError("Cannot specify both `--pretrained_model_name_or_path` and `--model_config_name_or_path`")

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the diffusion model."
        )

    return args


def make_train_dataset(args, accelerator):
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
            data_dir=args.train_data_dir,
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
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
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

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        examples["pixel_values"] = images

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values}


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

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
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load AutoencoderKL
    if args.pretrained_model_name_or_path is None and args.model_config_name_or_path is None:
        config = AutoencoderKL.load_config("stabilityai/sd-vae-ft-mse")
        vae = AutoencoderKL.from_config(config)
    elif args.pretrained_model_name_or_path is not None:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision)
    else:
        config = AutoencoderKL.load_config(args.model_config_name_or_path)
        vae = AutoencoderKL.from_config(config)
    if args.use_ema:
        ema_vae = EMAModel(vae.parameters(), model_cls=AutoencoderKL, model_config=vae.config)
    perceptual_loss = lpips.LPIPS(net="vgg").eval()
    discriminator = NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=False).apply(weights_init)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    sub_dir = "autoencoderkl_ema"
                    ema_vae.save_pretrained(os.path.join(output_dir, sub_dir))

                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    if isinstance(model, AutoencoderKL):
                        sub_dir = "autoencoderkl"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    else:
                        sub_dir = "discriminator"
                        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(output_dir, sub_dir, "pytorch_model.bin"))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                if args.use_ema:
                    sub_dir = "autoencoderkl_ema"
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, sub_dir), AutoencoderKL)
                    ema_vae.load_state_dict(load_model.state_dict())
                    ema_vae.to(accelerator.device)
                    del load_model

                # pop models so that they are not loaded again
                model = models.pop()
                load_model = NLayerDiscriminator(input_nc=3, n_layers=3, use_actnorm=False).load_state_dict(
                    os.path.join(input_dir, "discriminator", "pytorch_model.bin")
                )
                model.load_state_dict(load_model.state_dict())
                del load_model

                model = models.pop()
                load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="autoencoderkl")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(True)
    if args.decoder_only:
        vae.encoder.requires_grad_(False)
        if getattr(vae, "quant_conv", None):
            vae.quant_conv.requires_grad_(False)
    vae.train()
    discriminator.requires_grad_(True)
    discriminator.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(vae).dtype != torch.float32:
        raise ValueError(f"VAE loaded as datatype {unwrap_model(vae).dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
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

    params_to_optimize = filter(lambda p: p.requires_grad, vae.parameters())
    disc_params_to_optimize = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    disc_optimizer = optimizer_class(
        disc_params_to_optimize,
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
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
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    disc_lr_scheduler = get_scheduler(
        args.disc_lr_scheduler,
        optimizer=disc_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    (
        vae,
        discriminator,
        optimizer,
        disc_optimizer,
        train_dataloader,
        lr_scheduler,
        disc_lr_scheduler,
    ) = accelerator.prepare(
        vae, discriminator, optimizer, disc_optimizer, train_dataloader, lr_scheduler, disc_lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move VAE, perceptual loss and discriminator to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    perceptual_loss.to(accelerator.device, dtype=weight_dtype)
    discriminator.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_vae.to(accelerator.device, dtype=weight_dtype)

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

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space and reconstruct from them
            targets = batch["pixel_values"].to(dtype=weight_dtype)
            posterior = accelerator.unwrap_model(vae).encode(targets).latent_dist
            latents = posterior.sample()
            reconstructions = accelerator.unwrap_model(vae).decode(latents).sample

            if (step // args.gradient_accumulation_steps) % 2 == 0 or global_step < args.disc_start:
                with accelerator.accumulate(vae):
                    # reconstruction loss. Pixel level differences between input vs output
                    if args.rec_loss == "l2":
                        rec_loss = F.mse_loss(reconstructions.float(), targets.float(), reduction="none")
                    elif args.rec_loss == "l1":
                        rec_loss = F.l1_loss(reconstructions.float(), targets.float(), reduction="none")
                    else:
                        raise ValueError(f"Invalid reconstruction loss type: {args.rec_loss}")
                    # perceptual loss. The high level feature mean squared error loss
                    with torch.no_grad():
                        p_loss = perceptual_loss(reconstructions, targets)

                    rec_loss = rec_loss + args.perceptual_scale * p_loss
                    nll_loss = rec_loss
                    nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

                    kl_loss = posterior.kl()
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                    logits_fake = discriminator(reconstructions)
                    g_loss = -torch.mean(logits_fake)
                    last_layer = accelerator.unwrap_model(vae).decoder.conv_out.weight
                    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                    disc_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                    disc_weight = torch.clamp(disc_weight, 0.0, 1e4).detach()
                    disc_weight = disc_weight * args.disc_scale
                    disc_factor = args.disc_factor if global_step >= args.disc_start else 0.0

                    loss = nll_loss + args.kl_scale * kl_loss + disc_weight * disc_factor * g_loss

                    logs = {
                        "loss": loss.detach().mean().item(),
                        "nll_loss": nll_loss.detach().mean().item(),
                        "rec_loss": rec_loss.detach().mean().item(),
                        "p_loss": p_loss.detach().mean().item(),
                        "kl_loss": kl_loss.detach().mean().item(),
                        "disc_weight": disc_weight.detach().mean().item(),
                        "disc_factor": disc_factor,
                        "g_loss": g_loss.detach().mean().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = vae.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            else:
                with accelerator.accumulate(discriminator):
                    logits_real = discriminator(targets)
                    logits_fake = discriminator(reconstructions)
                    disc_loss = hinge_d_loss if args.disc_loss == "hinge" else vanilla_d_loss
                    disc_factor = args.disc_factor if global_step >= args.disc_start else 0.0
                    d_loss = disc_factor * disc_loss(logits_real, logits_fake)
                    logs = {
                        "disc_loss": d_loss.detach().mean().item(),
                        "logits_real": logits_real.detach().mean().item(),
                        "logits_fake": logits_fake.detach().mean().item(),
                        "disc_lr": disc_lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        params_to_clip = discriminator.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    disc_optimizer.step()
                    disc_lr_scheduler.step()
                    disc_optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.use_ema:
                    ema_vae.step(vae.parameters())

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

                    if global_step == 1 or global_step % args.validation_steps == 0:
                        if args.use_ema:
                            ema_vae.store(vae.parameters())
                            ema_vae.copy_to(vae.parameters())
                        image_logs = log_validation(
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        if args.use_ema:
                            ema_vae.restore(vae.parameters())

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        discriminator = accelerator.unwrap_model(discriminator)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
        vae.save_pretrained(args.output_dir)
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        # Run a final round of validation.
        image_logs = None
        image_logs = log_validation(
            vae=vae,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            step=global_step,
            is_final_validation=True,
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

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
