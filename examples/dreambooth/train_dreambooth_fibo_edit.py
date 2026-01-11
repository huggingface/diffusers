import abc
import argparse
import io
import itertools
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch
import transformers
import ujson
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import HfFolder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import diffusers
from diffusers import AutoencoderKLWan, BriaFiboEditPipeline
from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
from diffusers.models.transformers.transformer_bria_fibo import (
    BriaFiboTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft


# Set Logger
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="briaai/Fibo-edit",
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fibo-edit-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=3000,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1501,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "cosine_with_warmup", "constant_with_warmup_cosine_decay"'
        ),
    )
    parser.add_argument(
        "--constant_steps",
        type=int,
        default=-1,
        help=("Amount of constsnt lr steps"),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-15,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
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
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--drop_rate_cfg",
        type=float,
        default=0.0,
        help="Rate for Classifier Free Guidance dropping.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="no",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=int,
        default=1,
        required=False,
        help="Path to pretrained ELLA model",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
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
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing the instance prompt for each image",
    )
    (
        parser.add_argument(
            "--repeats",
            type=int,
            default=1,
            help="How many times to repeat the training data.",
        ),
    )
    parser.add_argument(
        "--input_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the source image.",
    )
    args = parser.parse_args()
    return args


# Resolution mapping for dynamic aspect ratio selection
RESOLUTIONS_1k = {
    0.67: (832, 1248),
    0.778: (896, 1152),
    0.883: (960, 1088),
    1.000: (1024, 1024),
    1.133: (1088, 960),
    1.286: (1152, 896),
    1.462: (1216, 832),
    1.600: (1280, 800),
    1.750: (1344, 768),
}


def find_closest_resolution(image_width, image_height):
    """Find the closest aspect ratio from RESOLUTIONS_1k and return the target dimensions."""
    image_aspect = image_width / image_height
    aspect_ratios = list(RESOLUTIONS_1k.keys())
    closest_ratio = min(aspect_ratios, key=lambda x: abs(x - image_aspect))
    return RESOLUTIONS_1k[closest_ratio]


def create_attention_matrix(attention_mask):
    attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)
    # convert to 0 - keep, -inf ignore
    attention_matrix = torch.where(
        attention_matrix == 1, 0.0, -torch.inf
    )  # Apply -inf to ignored tokens for nulling softmax score
    return attention_matrix


@torch.no_grad()
def get_smollm_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: AutoModelForCausalLM,
    prompts: Union[str, List[str]] = None,
    max_sequence_length: int = 2048,
):
    prompts = [prompts] if isinstance(prompts, str) else prompts
    bot_token_id = 128000  # same as Llama

    if prompts[0] == "":
        bs = len(prompts)
        assert all(p == "" for p in prompts)
        text_input_ids = torch.zeros([bs, 1], dtype=torch.int64, device=text_encoder.device) + bot_token_id
        attention_mask = torch.ones([bs, 1], dtype=torch.int64, device=text_encoder.device)
    else:
        text_inputs = tokenizer(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    if len(prompts) == 1:
        assert (attention_mask == 1).all()

    hidden_states = text_encoder(
        text_input_ids, attention_mask=attention_mask, output_hidden_states=True
    ).hidden_states
    # We need a 4096 dim so since we have 2048 we take last 2 layers
    prompt_embeds = torch.concat([hidden_states[-1], hidden_states[-2]], dim=-1)

    return prompt_embeds, hidden_states, attention_mask


def open_image_from_binary(binary_data):
    return Image.open(io.BytesIO(binary_data))


def pad_embedding(prompt_embeds, max_tokens):
    # Pads a tensor which is not masked, i.e. the "initial" tensor mask is 1's
    # We extend the tokens to max tokens and provide a mask to differentiate the masked areas
    b, seq_len, dim = prompt_embeds.shape
    padding = torch.zeros(
        (b, max_tokens - seq_len, dim),
        dtype=prompt_embeds.dtype,
        device=prompt_embeds.device,
    )
    attentions_mask = torch.zeros((b, max_tokens), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    attentions_mask[:, :seq_len] = 1  # original tensor is not masked
    prompt_embeds = torch.concat([prompt_embeds, padding], dim=1)

    return prompt_embeds, attentions_mask


class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        normal_samples = torch.randn((batch_size,), device=device) * self.std + shift
        sigmas = torch.sigmoid(normal_samples)
        return sigmas

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 256,
        max_tokens: int = 4096,
        min_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift


class TimestepSampler(abc.ABC):
    """Base class for timestep samplers.

    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(
        self,
        batch_size: int,
        seq_length: int | None = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(
        self,
        batch_size: int,
        seq_length: int | None = None,
        device: torch.device = None,
    ) -> torch.Tensor:  # noqa: ARG002
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)


class ShiftedStretchedLogitNormalTimestepSampler:
    """
    Samples timesteps from a stretched logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0, uniform_prob: float = 0.1):
        self.std = std
        self.shifted_logit_normal_sampler = ShiftedLogitNormalTimestepSampler(std=std)
        self.uniform_sampler = UniformTimestepSampler()
        self.uniform_prob = uniform_prob

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        # Determine which sampler to use for each batch element
        should_use_uniform = torch.rand(batch_size, device=device) < self.uniform_prob

        # Initialize an empty tensor for the results
        timesteps = torch.empty(batch_size, device=device)

        # Sample from uniform sampler where should_use_uniform is True
        num_uniform = should_use_uniform.sum().item()
        if num_uniform > 0:
            timesteps[should_use_uniform] = self.uniform_sampler.sample(
                batch_size=num_uniform, seq_length=seq_length, device=device
            )

        # Sample from shifted logit-normal sampler where should_use_uniform is False
        should_use_shifted = ~should_use_uniform
        num_shifted = should_use_shifted.sum().item()
        if num_shifted > 0:
            timesteps[should_use_shifted] = self.shifted_logit_normal_sampler.sample(
                batch_size=num_shifted, seq_length=seq_length, device=device
            )
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size=batch_size, seq_length=seq_length, device=batch.device)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance images with the prompts for fine-tuning the model.
    Images are dynamically resized and center-cropped to the closest aspect ratio from RESOLUTIONS_1k.
    """

    def __init__(
        self,
        instance_data_root,
        repeats=1,
    ):
        self.custom_instance_prompts = None

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

            if args.input_image_column is None:
                input_image_column = column_names[0]
                logger.info(f"source image column defaulting to {input_image_column}")
            else:
                input_image_column = args.input_image_column
                if input_image_column not in column_names:
                    raise ValueError(
                        f"`--input_image_column` value '{args.input_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            source_images = dataset["train"][input_image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided. If your dataset "
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
                    # Validate and normalize the JSON caption (raises error if invalid)
                    cleaned_caption = clean_json_caption(caption)
                    self.custom_instance_prompts.extend(itertools.repeat(cleaned_caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir()) if path.is_file()]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            img = open_image_from_binary(img)
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        self.source_images = []
        for img in source_images:
            img = open_image_from_binary(img)
            self.source_images.extend(itertools.repeat(img, repeats))

        self.num_source_images = len(self.source_images)
        self._length = self.num_source_images
        # Normalization transform (applied after resize/crop)
        self.to_tensor_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def _process_image(self, image):
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def __getitem__(self, index):
        example = {}
        # Get the original image
        instance_image = self.instance_images[index % self.num_instance_images]
        source_image = self.source_images[index % self.num_source_images]
        instance_image = self._process_image(instance_image)
        source_image = self._process_image(source_image)

        # Get image dimensions and find closest resolution
        img_width, img_height = instance_image.size
        target_width, target_height = find_closest_resolution(img_width, img_height)

        # Resize and center crop to target dimensions
        # Calculate scale factor to ensure we can center crop to target dimensions
        target_aspect = target_width / target_height
        img_aspect = img_width / img_height

        if img_aspect > target_aspect:
            # Image is wider than target, resize based on height
            scale = target_height / img_height
        else:
            # Image is taller than target, resize based on width
            scale = target_width / img_width

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize maintaining aspect ratio
        instance_image = transforms.Resize(
            (new_height, new_width), interpolation=transforms.InterpolationMode.BILINEAR
        )(instance_image)
        source_image = transforms.Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BILINEAR)(
            source_image
        )

        # Center crop to exact target dimensions
        instance_image = transforms.CenterCrop((target_height, target_width))(instance_image)
        source_image = transforms.CenterCrop((target_height, target_width))(source_image)
        # Convert to tensor and normalize
        instance_image = self.to_tensor_normalize(instance_image)
        source_image = self.to_tensor_normalize(source_image)
        example["instance_images"] = instance_image
        example["source_images"] = source_image
        example["target_width"] = target_width
        example["target_height"] = target_height

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                raise ValueError("Caption cannot be empty when custom_instance_prompts is provided")
        else:
            raise ValueError(
                "Captions must be provided via --caption_column when using --dataset_name, or via dataset metadata when loading from directory"
            )

        return example


def clean_json_caption(caption):
    """Validate and normalize JSON caption format. Raises ValueError if caption is not valid JSON."""
    try:
        caption = json.loads(caption)
        return ujson.dumps(caption, escape_forward_slashes=False)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(
            f"Caption must be in valid JSON format. Error: {e}. Caption: {caption[:100] if len(str(caption)) > 100 else caption}"
        )


def add_lora(transformer, lora_rank):
    target_modules = [
        # HF Lora Layers
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
        "proj_mlp",
        # +  layers that exist on ostris ai-toolkit / replicate trainer
        "norm1_context.linear",
        "norm1.linear",
        "norm.linear",
        "proj_out",
    ]
    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)


def set_lora_training(accelerator, transformer, lora_rank):
    add_lora(transformer, lora_rank)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxLoraLoaderMixin.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        load_lora(transformer=transformer_, input_dir=input_dir)
        # Make sure the trainable params are in float32. This is again needed since the base models
        cast_training_params([transformer_], dtype=torch.float32)

    if accelerator:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format


def load_lora(transformer, input_dir):
    lora_state_dict = FluxLoraLoaderMixin.lora_state_dict(input_dir)

    transformer_state_dict = {
        f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            raise Exception(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: {unexpected_keys}. "
            )


# Not really cosine but with decay
def get_cosine_schedule_with_warmup_and_decay(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    constant_steps=-1,
    eps=1e-5,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        constant_steps (`int`):
            The total number of constant lr steps following a warmup

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if constant_steps <= 0:
        constant_steps = num_training_steps - num_warmup_steps

    def lr_lambda(current_step):
        # Accelerate sends current_step*num_processes
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + constant_steps:
            return 1

        return max(
            eps,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps - constant_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(name, optimizer, num_warmup_steps, num_training_steps, constant_steps):
    if name != "constant_with_warmup_cosine_decay":
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # Using custom warmup+constant+decay scheduler
    return get_cosine_schedule_with_warmup_and_decay(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        constant_steps=constant_steps,
    )


def load_checkpoint(accelerator, args):
    # Load from local checkpoint that sage maker synced to s3 prefix
    global_step = 0
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        args.resume_from_checkpoint = None
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
        global_step = int(path.split("_")[-1])

    return global_step


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    input_images = [example["source_images"] for example in examples]
    captions = [example["instance_prompt"] for example in examples]
    # Get target dimensions (assuming batch_size=1, so we can get from first example)
    target_width = examples[0]["target_width"]
    target_height = examples[0]["target_height"]

    input_images = torch.stack(input_images)
    input_images = input_images.to(memory_format=torch.contiguous_format).float()

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return pixel_values, input_images, captions, target_width, target_height


def get_accelerator(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    # Set huggingface token key if provided
    with accelerator.main_process_first():
        if accelerator.is_local_main_process:
            if os.environ.get("HF_API_TOKEN"):
                HfFolder.save_token(os.environ.get("HF_API_TOKEN"))

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return accelerator


def main(args):
    try:
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA Version: {cuda_version}")
    except Exception as e:
        print(f"Error checking CUDA version: {e}")
        raise e

    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))

    seed = args.seed + RANK
    set_seed(seed)
    random.seed(seed)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Set accelerator with fsdp/data-parallel
    accelerator = get_accelerator(args)

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TOTAL_BATCH_NO_ACC = args.train_batch_size

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"TORCH_VERSION {torch.__version__}")
    logger.info(f"DIFFUSERS_VERSION {diffusers.__version__}")

    logger.info("using precompted datasets")

    transformer = BriaFiboTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        low_cpu_mem_usage=False,  # critical: avoid meta tensors
        weight_dtype=weight_dtype,
    )
    transformer = transformer.to(accelerator.device).eval()
    total_num_layers = transformer.config["num_layers"] + transformer.config["num_single_layers"]

    logger.info(f"Using precision of {weight_dtype}")
    if args.lora_rank > 0:
        logger.info(f"Using LORA with rank {args.lora_rank}")
        transformer.requires_grad_(False)
        transformer.to(dtype=weight_dtype)
        set_lora_training(accelerator, transformer, args.lora_rank)
        # Upcast trainable parameters (LoRA) into fp32 for mixed precision training
        cast_training_params([transformer], dtype=torch.float32)
    else:
        transformer.requires_grad_(True)
        assert transformer.dtype == torch.float32

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    get_prompt_embeds_lambda = get_smollm_prompt_embeds
    print("Loading smolLM text encoder")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            dtype=weight_dtype,
        )
        .to(accelerator.device)
        .eval()
        .requires_grad_(False)
    )

    vae_model = AutoencoderKLWan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    vae_model = vae_model.to(accelerator.device).requires_grad_(False)
    # Read vae config
    vae_config = vae_model.config
    vae_config["shift_factor"] = (
        torch.tensor(vae_model.config["latents_mean"]).reshape((1, 48, 1, 1)).to(device=accelerator.device)
    )
    vae_config["scaling_factor"] = 1 / torch.tensor(vae_model.config["latents_std"]).reshape((1, 48, 1, 1)).to(
        device=accelerator.device
    )
    vae_config["compression_rate"] = 16
    vae_config["latent_channels"] = 48

    def get_prompt_embeds(prompts):
        prompt_embeddings, text_encoder_layers, attentions_masks = get_prompt_embeds_lambda(
            tokenizer,
            text_encoder,
            prompts=prompts,
            max_sequence_length=args.max_sequence_length,
        )
        return prompt_embeddings, text_encoder_layers, attentions_masks

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to prodigy"
        )
        args.optimizer = "prodigy"

    if args.lora_rank > 0:
        parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    else:
        parameters = transformer.parameters()

    if args.optimizer.lower() == "adamw":
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_cls = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_cls(
            parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.lr_scheduler == "cosine_with_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_lr_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            constant_steps=args.constant_steps * accelerator.num_processes,
        )

    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
    fibo_edit_pipeline = BriaFiboEditPipeline(
        transformer=transformer,
        scheduler=lr_scheduler,
        vae=vae_model,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    logger.info("***** Running training *****")

    logger.info(f"diffusers version: {diffusers.__version__}")

    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    if args.resume_from_checkpoint != "no":
        global_step = load_checkpoint(accelerator, args)
    logger.info(f"Using {args.optimizer} with lr: {args.learning_rate}, beta2: {args.adam_beta2}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    now = datetime.now()
    times_arr = []
    # Init dynamic scheduler (resolution will be determined per batch)
    noise_scheduler = ShiftedStretchedLogitNormalTimestepSampler()

    # encode null prompt ""
    null_conditioning, null_conditioning_layers, _ = get_prompt_embeds([""])
    logger.info("Using empty prompt for null embeddings")
    assert null_conditioning.shape[0] == 1
    null_conditioning = null_conditioning.repeat(args.train_batch_size, 1, 1).to(dtype=torch.float32)
    null_conditioning_layers = [
        layer.repeat(args.train_batch_size, 1, 1).to(dtype=torch.float32) for layer in null_conditioning_layers
    ]

    vae_scale_factor = (
        2 ** (len(vae_config["block_out_channels"]) - 1)
        if "compression_rate" not in vae_config
        else vae_config["compression_rate"]
    )
    transformer.train()
    train_loss = 0.0
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        repeats=args.repeats,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    iter_ = iter(train_dataloader)
    for step in range(
        global_step * args.gradient_accumulation_steps,
        args.max_train_steps * args.gradient_accumulation_steps,
    ):
        have_batch = False

        while not have_batch:
            try:
                fetch_time = datetime.now()
                batch = next(iter_)
                fetch_time = datetime.now() - fetch_time
                have_batch = True
            except StopIteration:
                iter_ = iter(train_dataloader)
                logger.info(f"Rank {RANK} reinit iterator")

        target_pixel_values, input_pixel_values, captions, target_width, target_height = (
            batch  # Get batch with dynamic resolution
        )
        height, width = target_height, target_width
        target_latents, target_latent_image_ids = fibo_edit_pipeline.prepare_image_latents(
            image=target_pixel_values,
            batch_size=args.train_batch_size,
            num_channels_latents=vae_config["latent_channels"],
            height=height,
            width=width,
            dtype=torch.float32,
            device=accelerator.device,
            generator=generator,
        )
        # target latents of the target image should be 0
        target_latent_image_ids[..., 0] = 0

        context_latents, context_latent_image_ids = fibo_edit_pipeline.prepare_image_latents(
            image=input_pixel_values,
            batch_size=args.train_batch_size,
            num_channels_latents=vae_config["latent_channels"],
            height=height,
            width=width,
            dtype=torch.float32,
            device=accelerator.device,
            generator=generator,
        )

        # Get Captions
        encoder_hidden_states, text_encoder_layers, prompt_attention_mask = get_prompt_embeds(captions)
        text_encoder_layers = list(text_encoder_layers)
        # make sure that the number of text encoder layers is equal to the total number of layers in the transformer
        assert len(text_encoder_layers) <= total_num_layers
        text_encoder_layers = text_encoder_layers + [text_encoder_layers[-1]] * (
            total_num_layers - len(text_encoder_layers)
        )
        null_conditioning_layers = null_conditioning_layers + [null_conditioning_layers[-1]] * (
            total_num_layers - len(null_conditioning_layers)
        )

        target_pixel_values = target_pixel_values.to(device=accelerator.device, dtype=torch.float32)
        input_pixel_values = input_pixel_values.to(device=accelerator.device, dtype=torch.float32)
        encoder_hidden_states = encoder_hidden_states.to(device=accelerator.device, dtype=torch.float32)
        prompt_attention_mask = prompt_attention_mask.to(device=accelerator.device, dtype=torch.float32)

        # create attention mask for the target and context latents
        target_latents_attention_mask = torch.ones(
            [target_latents.shape[0], target_latents.shape[1]],
            dtype=target_latents.dtype,
            device=target_latents.device,
        )

        context_latents_attention_mask = torch.ones(
            [context_latents.shape[0], context_latents.shape[1]],
            dtype=context_latents.dtype,
            device=context_latents.device,
        )

        attention_mask = torch.cat(
            [prompt_attention_mask, target_latents_attention_mask, context_latents_attention_mask], dim=1
        )

        with accelerator.accumulate(transformer):
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(target_latents)

            bsz = target_pixel_values.shape[0]

            seq_len = (height // vae_scale_factor) * (width // vae_scale_factor)

            sigmas = noise_scheduler.sample(bsz, seq_len, device=accelerator.device)
            timesteps = sigmas * 1000
            while len(sigmas.shape) < len(noise.shape):
                sigmas = sigmas.unsqueeze(-1)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * target_latents

            # input for rope positional embeddings for text
            num_text_tokens = encoder_hidden_states.shape[1]
            text_ids = torch.zeros(num_text_tokens, 3).to(device=accelerator.device, dtype=encoder_hidden_states.dtype)

            # Sample masks for the edit prompts.
            if args.drop_rate_cfg > 0:
                null_embedding, null_attention_mask = pad_embedding(null_conditioning, max_tokens=num_text_tokens)
                # null embedding for 10% of the images
                random_p = torch.rand(bsz, device=target_latents.device, generator=generator)

                prompt_mask = random_p < args.drop_rate_cfg

                prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                encoder_hidden_states = torch.where(prompt_mask, null_embedding, encoder_hidden_states)

                text_encoder_layers = [
                    torch.where(
                        prompt_mask,
                        pad_embedding(null_conditioning_layers[i], max_tokens=num_text_tokens)[0],
                        text_encoder_layers[i],
                    )
                    for i in range(len(text_encoder_layers))
                ]

                prompt_mask = prompt_mask.reshape(bsz, 1)
                prompt_attention_mask = torch.where(prompt_mask, null_attention_mask, prompt_attention_mask)

            # Get the target for loss depending on the prediction type
            target = noise - target_latents  # V pred
            latent_height = int(height) // vae_scale_factor
            latent_width = int(width) // vae_scale_factor

            patched_latent_image_ids = fibo_edit_pipeline._prepare_latent_image_ids(
                noisy_latents.shape[0],
                latent_height,
                latent_width,
                accelerator.device,
                noisy_latents.dtype,
            )

            latent_attention_mask = torch.ones(
                [noisy_latents.shape[0], noisy_latents.shape[1]],
                dtype=target_latents.dtype,
                device=target_latents.device,
            )
            patched_latent_image_ids = torch.cat([patched_latent_image_ids, context_latent_image_ids], dim=0)
            noisy_latents = torch.cat([noisy_latents, context_latents], dim=1)
            attention_mask = torch.cat(
                [prompt_attention_mask, latent_attention_mask, context_latents_attention_mask], dim=1
            )

            # Prepare attention_matrix
            attention_mask = create_attention_matrix(attention_mask)  # batch, seq => batch, seq, seq

            attention_mask = attention_mask.unsqueeze(dim=1)  # for brodoacast to attention heads
            joint_attention_kwargs = {"attention_mask": attention_mask}

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,  # [batch,128,height/patch*width/patch]
                text_encoder_layers=text_encoder_layers,
                txt_ids=text_ids,
                img_ids=patched_latent_image_ids,
                return_dict=False,
                joint_attention_kwargs=joint_attention_kwargs,
            )[0]
            model_pred = model_pred[:, : target_latents.shape[1]]
            # Un-Patchify latent  (4 -> 1)
            loss_coeff = WORLD_SIZE / TOTAL_BATCH_NO_ACC

            denoising_loss = torch.mean(
                ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            ).sum()
            denoising_loss = loss_coeff * denoising_loss

            loss = denoising_loss

            train_loss += accelerator.gather(loss.detach()).mean().item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(parameters, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                logger.info(f"train_loss: {train_loss}")
                after = datetime.now() - now
                now = datetime.now()

                times_arr += [after.total_seconds()]

            train_loss = 0.0

        if (global_step - 1) % args.checkpointing_steps == 0 and (global_step - 1) > 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_{global_step - 1}")

            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")
            now = datetime.now()

        if global_step == args.max_train_steps:
            save_path = os.path.join(args.output_dir, "checkpoint_final")

            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")
            now = datetime.now()

        logs = {"step_loss": loss.detach().item()}

        progress_bar.set_postfix(**logs)
        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    logger.info("Waiting for everyone :)")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
