import argparse
from diffusers.models.retriever import Retriever
from datasets import load_dataset, Image, load_dataset_builder, load_from_disk
import copy
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from rdm_dataset import RDMDataset
import datasets
import diffusers
import transformers
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, UNet2DConditionModel
from diffusers.pipelines.rdm.pipeline_rdm_colossalai import RDMPipelineColossal
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPModel, CLIPFeatureExtractor, CLIPTokenizer
from packaging import version
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
from typing import Callable, List, Optional, Union
import requests
import faiss
import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
from colossalai.nn.parallel.utils import get_static_torch_model
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
import importlib
_wandb_available = importlib.util.find_spec("wandb") is not None
def is_wandb_available():
    return _wandb_available
wandb = None
if is_wandb_available():
    import wandb

def wandb_setup(
    args: dict,
    project_name: str = "glide-text2im-finetune",
):
    return wandb.init(
        project=project_name,
        config=args,
    )

def get_pipeline(vae, clip_model, unet, tokenizer, feature_extractor):
    # I disabled safety checker as it causes an oom
    pipeline = RDMPipelineColossal(
        vae=vae,
        clip=clip_model,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True, steps_offset=1
        ),
        feature_extractor=feature_extractor
    )
    return pipeline

def log_progress(pipeline, args, step, prompt, save_path, wandb_run=None, logs={}, retrieved_images=None):
    logger.info("Running pipeline")
    image = pipeline(
        prompt, retrieved_images=retrieved_images, height=args.resolution, width=args.resolution, num_inference_steps=50, guidance_scale=args.guidance_scale
    ).images[0]

    image.save(save_path)
    if is_wandb_available():
        wandb_run.log(
            {
                **logs,
                "iter": step,
                "samples": wandb.Image(save_path, caption=prompt),
            }
        )

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_dist_logger()

def preprocess_images(images, feature_extractor: CLIPFeatureExtractor) -> torch.FloatTensor:
    """
    Preprocesses a list of images into a batch of tensors.

    Args:
        images (:obj:`List[Image.Image]`):
            A list of images to preprocess.

    Returns:
        :obj:`torch.FloatTensor`: A batch of tensors.
    """
    images = [np.array(image) for image in images]
#     print(images[0].shape)
    images = [(image + 1.) / 2. for image in images]
    images = feature_extractor(images, return_tensors="pt").pixel_values
    return images

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--placement",
        type=str,
        default="cpu",
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for inference.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="huggingface_rdm_training",
        help="Name of wandb run",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=100,
        help="Frequency to log/save the model.",
    )
    parser.add_argument(
        "--unet_config",
        type=str,
        default=None,
        help="The configuration file for the unet. Defaults to getting the pretrained model",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=20,
        help="Number of query images.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="Name of the clip model.",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        required=True,
        help="Location to save dataset with embedding.",
    )
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
        "--dataset_name",
        type=str,
        default="Isamu136/oxford_pets_with_l14_emb",
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
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
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
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
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
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.collected_params = None

        self.decay = decay
        self.optimization_step = 0

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        value = (1 + self.optimization_step) / (10 + self.optimization_step)
        one_minus_decay = 1 - min(self.decay, value)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict.
        This method is used by accelerate during checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Loads the ExponentialMovingAverage state.
        This method is used by accelerate during checkpointing to save the ema state dict.
        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.optimization_step = state_dict["optimization_step"]
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.shadow_params = state_dict["shadow_params"]
        if not isinstance(self.shadow_params, list):
            raise ValueError("shadow_params must be a list")
        if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
            raise ValueError("shadow_params must all be Tensors")

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            if not isinstance(self.collected_params, list):
                raise ValueError("collected_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.collected_params):
                raise ValueError("collected_params must all be Tensors")
            if len(self.collected_params) != len(self.shadow_params):
                raise ValueError("collected_params and shadow_params must have the same length")

# Gemini + ZeRO DDP
def gemini_zero_dpp(model: torch.nn.Module, placememt_policy: str = "auto"):
    from colossalai.nn.parallel import GeminiDDP

    model = GeminiDDP(
        model, device=get_current_device(), placement_policy=placememt_policy, pin_memory=True, search_range_mb=64
    )
    return model

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    if args.seed is None:
        colossalai.launch_from_torch(config={'gradient_clipping': args.max_grad_norm}, backend="gloo")
    else:
        colossalai.launch_from_torch(config={'gradient_clipping': args.max_grad_norm}, seed=args.seed, backend="gloo")
    if args.seed is not None:
        gpc.set_seed(args.seed)
    revision_dtype = torch.float32
    if args.revision == "fp16":
        revision_dtype = torch.float16
    elif args.revision == "bf16":
        revision_dtype = torch.bfloat16
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    wandb_run = None
    
    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        diffusers.utils.logging.set_verbosity_info()
        if is_wandb_available():
            wandb_run = wandb_setup(args, args.project_name)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    with ColoInitContext(device=get_current_device()):
        if args.unet_config:
            unet = UNet2DConditionModel.from_config(args.unet_config, low_cpu_mem_usage=False).to(dtype=revision_dtype)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, low_cpu_mem_usage=False
            )
    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(dtype=revision_dtype)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    clip_model.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters())

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * gpc.get_world_size(ParallelMode.DATA)
        )
    unet = gemini_zero_dpp(unet, args.placement)
    # Initialize the optimizer
    optimizer = GeminiAdamOptimizer(
        unet, lr=args.learning_rate, initial_scale=2**5, clipping_norm=args.max_grad_norm
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if os.path.exists(args.dataset_save_path):
        dataset = {}
        dataset["train"] = load_from_disk(args.dataset_save_path)
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
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
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    retriever = Retriever(clip_model, tokenizer, feature_extractor, dataset["train"], args.dataset_save_path, get_current_device(), args.image_column)
    train_dataset = RDMDataset(dataset["train"],args.image_column,args.caption_column, tokenizer,feature_extractor,retriever, clip_model, center_crop=args.center_crop, size=512, num_queries=args.num_queries)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
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
    

    # Move text_encode and vae to gpu and cast to weight_dtype
    clip_model.to(get_current_device(), dtype=weight_dtype)
    vae.to(get_current_device(), dtype=weight_dtype)
    if args.use_ema:
        ema_unet.to(get_current_device())

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # Train!
    total_batch_size = args.train_batch_size * gpc.get_world_size(ParallelMode.DATA) * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not gpc.get_local_rank(ParallelMode.DATA) == 0)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            torch.cuda.reset_peak_memory_stats()
            # Move batch to gpu
            for key, value in batch.items():
                batch[key] = value.to(get_current_device(), non_blocking=True)
            # Skip steps until we reach the resumed step
            optimizer.zero_grad()
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            text_embeddings = clip_model.get_text_features(batch["input_ids"])
            text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
            text_embeddings = text_embeddings[:, None, :]

            retrieved_images = batch["nearest_neighbors"]

            if retrieved_images is not None:
                clip_hidden_states = []
                for i in range(retrieved_images.shape[0]):
                    # preprocess retrieved images
                    processed_images = retrieved_images[i].to(get_current_device(), dtype=weight_dtype, non_blocking=True)
                    image_embeddings = clip_model.get_image_features(processed_images)
                    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)

                    clip_hidden_states.append(torch.cat([text_embeddings[i], image_embeddings]))
                text_embeddings = torch.stack(clip_hidden_states)
            encoder_hidden_states = text_embeddings
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            train_loss += loss.detach().item()
            # Backpropagate=
            optimizer.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            logger.info(f"max GPU_mem cost is {torch.cuda.max_memory_allocated()/2**20} MB", ranks=[0])
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if args.use_ema:
                ema_unet.step(unet.parameters())
            progress_bar.update(1)
            global_step += 1
            wandb_run.log({"train_loss": train_loss, "iter": global_step})
            train_loss = 0.0
            if global_step % args.log_frequency == 0:
                torch.cuda.synchronize()
                torch_unet = get_static_torch_model(unet)
                if gpc.get_local_rank(ParallelMode.DATA) == 0:
                    prompt = dataset['train'][args.caption_column][0]
                    pipeline = get_pipeline(vae, clip_model, unet, tokenizer, feature_extractor)
                    os.makedirs(os.path.join(args.output_dir, "imgs"), exist_ok=True)
                    save_path = os.path.join(args.output_dir, f"imgs/{global_step}.jpg")
                    embedding = map_txt_to_clip_feature(clip_model, tokenizer, prompt)
                    retrieved_images = retrieved_images = retriever.retrieve_imgs(embedding, k=num_queries).examples[image_column]
                    for i in range(len(retrieved_images)):
                        if not retrieved_images[i].mode == "RGB":
                            retrieved_images[i] = retrieved_images[i].convert("RGB")
                        retrieved_images[i] = retrieved_images[i].resize((args.resolution, args.resolution), resample=PIL_INTERPOLATION['bicubic'])
                    log_progress(pipeline, args, global_step, prompt, save_path, wandb_run, retrieved_images=retrieved_images)
                    del pipeline
            if global_step % args.checkpointing_steps == 0:
                torch.cuda.synchronize()
                torch_unet = get_static_torch_model(unet)
                if gpc.get_local_rank(ParallelMode.DATA) == 0:
                    save_path = os.path.join(args.output_dir, f"unet")
                    torch.save(torch_unet, save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break
    torch.cuda.synchronize()
    unet = get_static_torch_model(unet)
    # Create the pipeline using the trained modules and save it.
    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = RDMPipelineColossal.from_pretrained(
            args.pretrained_model_name_or_path,
            clip=clip_model,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)


if __name__ == "__main__":
    main()
