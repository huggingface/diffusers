import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import diffusers
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.optimization import get_linear_schedule_with_warmup
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    export_to_video,
    load_video,
)
from diffusers.video_processor import VideoProcessor


logger = get_logger(__name__, log_level="INFO")


class MockSafetyChecker:
    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, *args, **kwargs):
        return True

    def check_video_safety(self, video):
        return video


def arch_invariant_rand(shape, dtype, device, seed=None):
    rng = np.random.RandomState(seed)
    random_array = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="nvidia/Cosmos-Predict2.5-2B",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="diffusers/base/post-trained",
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
        "--text_encoder_attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="The attention implementation to use for the text encoder (Qwen2.5 VL).",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="datasets/cosmos_nemo_assets",
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuned-lora",
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
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
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
        "--conditional_frame_timestep",
        type=float,
        default=0.0001,
        help="0.0001 for post-trained model. Set to < 0 to disable.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=20,
        help="Save a checkpoint of the training state every X epochs.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help=("The alpha parameter for Lora scaling."),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        help="Whether or not to use DoRA (Weight-Decomposed Low-Rank Adaptation).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=36,
        help="Number of denoising steps during final eval inference.",
    )
    parser.add_argument("--height", type=int, default=704, help="Height of the training videos in pixels.")
    parser.add_argument("--width", type=int, default=1280, help="Width of the training videos in pixels.")
    parser.add_argument("--num_frames", type=int, default=93, help="Number of frames per training video.")
    parser.add_argument(
        "--cfg_dropout_prob",
        type=float,
        default=0.2,
        help="Probability of dropping text or video conditioning per sample for CFG training.",
    )
    parser.add_argument(
        "--conditional_frames_probs",
        type=json.loads,
        default={1: 0.5, 2: 0.5},
        help=(
            "JSON dict mapping number of conditional frames to sampling probability. "
            "Default {1: 0.5, 2: 0.5} trains Image2World and Video2World equally."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2 ** (-14.5),
        help="Learning rate for the AdamW optimizer used in build_optimizer_and_scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay for the AdamW optimizer used in build_optimizer_and_scheduler.",
    )
    parser.add_argument(
        "--scheduler_warm_up_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for the linear LR scheduler.",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=100000,
        help="Total number of training steps for the LR scheduler.",
    )
    parser.add_argument(
        "--scheduler_f_max",
        type=float,
        default=0.5,
        help="Maximum LR multiplier (peak after warmup) for the linear scheduler.",
    )
    parser.add_argument(
        "--scheduler_f_min",
        type=float,
        default=0.2,
        help="Minimum LR multiplier (floor of linear decay) for the linear scheduler.",
    )
    parser.add_argument(
        "--do_final_eval",
        action="store_true",
        help="Whether to run inference on a training sample after training completes.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.use_dora:
        args.output_dir = args.output_dir + "-dora"

    return args


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_frames = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")

        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        logger.info(f"{len(self.video_paths)} videos in total", main_process_only=True)

        self.video_size = video_size
        self.video_processor = VideoProcessor(vae_scale_factor=8, resample="bilinear")
        self.num_failed_loads = 0

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> list:
        frames = load_video(video_path)
        total_frames = len(frames)
        if total_frames < self.num_frames:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, at least {self.num_frames} frames are required."
            )

        # randomly sample a consecutive window of frames
        max_start_idx = total_frames - self.num_frames
        start_frame = np.random.randint(0, max_start_idx + 1)
        return frames[start_frame : start_frame + self.num_frames]

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            print(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    print(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            print(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> torch.Tensor:
        frames = self._load_video(video_path)  # list of PIL images
        video = self.video_processor.preprocess_video(frames, height=self.video_size[0], width=self.video_size[1])
        # video: [1, C, T, H, W] in [-1, 1]
        return video.squeeze(0)  # [C, T, H, W]

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = {}
            video = self._get_frames(self.video_paths[index])  # [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.splitext(os.path.basename(video_path))[0]

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["caption"] = caption

            return data
        except Exception as e:
            self.num_failed_loads += 1
            print(f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n")
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def build_dataloader(args):
    dataset = VideoDataset(
        video_paths=None,
        num_frames=args.num_frames,
        video_size=[args.height, args.width],
        dataset_dir=args.train_data_dir,
    )

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    return dataloader


def get_flow_xt_and_target_v(clean_latent, t, cond_mask):
    # https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py#L779
    noise = torch.randn_like(clean_latent)
    target_velocity = noise - clean_latent
    xt_B_C_T_H_W = noise * t + clean_latent * (1 - t)

    # https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py#L104
    xt_B_C_T_H_W = clean_latent * cond_mask + xt_B_C_T_H_W * (1 - cond_mask)
    return xt_B_C_T_H_W, target_velocity


def sample_train_sigma_t(batch_size, distribution, device, dtype=torch.float32, shift=5):
    if distribution == "uniform":
        t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
    elif distribution == "logitnormal":
        t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Time distribution {distribution} is not implemented.")
    sigma_t = shift * t / (1 + (shift - 1) * t)  # 0.0 <= sigma_t <= 1.0
    return sigma_t.view(batch_size, 1, 1, 1, 1)


def main():
    args = parse_args()

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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        print("-" * 100)
        print(args)
        print("-" * 100)

    # Initialize models
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        text_encoder_attn_implementation=args.text_encoder_attn_implementation,
        safety_checker=MockSafetyChecker(),
    )

    dit = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder

    dit.set_autocast_fp32(True)
    dit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    target_modules_list = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    dit_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=target_modules_list,
        use_dora=args.use_dora,
    )
    logger.info(
        f"Add LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, targets={target_modules_list}, use_dora={args.use_dora}"
    )

    device = accelerator.device
    dit.to(device)
    vae.to(device)
    text_encoder.to(device)
    dit_dtype = dit.dtype

    # Add adapter and make sure the trainable params are in float32.
    dit.add_adapter(dit_lora_config)

    if accelerator.mixed_precision in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(dit, dtype=torch.float32)

    lora_params = [p for p in dit.parameters() if p.requires_grad]
    num_trainable_params = sum(p.numel() for p in lora_params)

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.scheduler_warm_up_steps,
        num_training_steps=args.num_training_steps,
        f_min=args.scheduler_f_min,
        f_max=args.scheduler_f_max,
    )

    train_dataloader = build_dataloader(args)

    # Prepare everything with our `accelerator`.
    dit, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        dit, optimizer, train_dataloader, lr_scheduler
    )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            assert len(models) == 1, f"Expected only one model to save, got {len(models)}"
            dit_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            weights.pop()
            Cosmos2_5_PredictBasePipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=dit_lora_state_dict,
                safe_serialization=True,
            )

    accelerator.register_save_state_pre_hook(save_model_hook)

    if accelerator.is_main_process:
        accelerator.init_trackers("diffusers-lora", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Video shape = {(args.height, args.width, args.num_frames)}")
    logger.info(f"  Total Trainable Parameters: {num_trainable_params / 10**9:.2f}B")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Checkpointing = {args.gradient_checkpointing}, allow_tf32 = {args.allow_tf32}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    padding_mask = torch.zeros(1, 1, args.height, args.width, dtype=dit_dtype, device=device)
    latent_shape = pipe.get_latent_shape_cthw(args.height, args.width, args.num_frames)
    latents_mean = pipe.latents_mean.float().to(device)
    latents_std = pipe.latents_std.float().to(device)  # 1/σ
    # Start training
    torch.set_grad_enabled(True)  # re-enable grad disabled by Cosmos2_5_PredictBasePipeline
    for epoch in range(first_epoch, args.num_train_epochs):
        dit.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                # Encode ground-truth video to latents
                # https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py#L532
                raw_state = batch["video"].to(device=device, dtype=vae.dtype)
                mu = vae.encode(raw_state).latent_dist.mean  # deterministic
                clean_latent = ((mu - latents_mean) * latents_std).contiguous().float()
                assert not clean_latent.requires_grad
                torch.cuda.empty_cache()

                # Encode text to text embeddings
                prompt_embeds = pipe._get_prompt_embeds(
                    prompt=batch["caption"],
                    device=device,
                )
                assert not prompt_embeds.requires_grad

                # CFG dropout: independently zero out text conditioning per sample
                bsz = clean_latent.shape[0]
                is_drop = torch.rand(bsz, device=device) < args.cfg_dropout_prob
                prompt_embeds[is_drop] = 0.0

                # Create indicator and mask to make the first few frames of x_t be the ground truth frames
                frames_options = list(args.conditional_frames_probs.keys())
                weights = list(args.conditional_frames_probs.values())
                num_conditional_frames = random.choices(frames_options, weights=weights, k=bsz)
                cond_indicator, cond_mask = pipe.create_condition_mask(
                    (bsz, *latent_shape),
                    device=device,
                    dtype=torch.float32,
                    num_cond_latent_frames=num_conditional_frames,
                )

                # Sample a random timestep
                sigma_t = sample_train_sigma_t(bsz, distribution="logitnormal", device=device)
                # 1. Sample noise 2. Get the target velocity 3. Get xt by interpolation between noise and clean
                xt_B_C_T_H_W, target_velocity = get_flow_xt_and_target_v(clean_latent, sigma_t, cond_mask)

                # Denoise
                if args.conditional_frame_timestep >= 0:
                    in_timestep = cond_indicator * args.conditional_frame_timestep + (1 - cond_indicator) * sigma_t

                pred_velocity = dit(
                    hidden_states=xt_B_C_T_H_W,
                    condition_mask=cond_mask,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                # Loss is only calculated on the non-conditioned frames
                pred_velocity = target_velocity * cond_mask + pred_velocity * (1 - cond_mask)
                loss = F.mse_loss(pred_velocity.float(), target_velocity.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if (epoch + 1) % args.checkpointing_epochs == 0 and (epoch + 1) < args.num_train_epochs:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    # After Training
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the lora layers
        unwrapped_dit = accelerator.unwrap_model(dit)
        dit_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_dit))
        Cosmos2_5_PredictBasePipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=dit_lora_state_dict,
            safe_serialization=True,
        )

        if args.do_final_eval:
            noises = arch_invariant_rand((1, *latent_shape), dtype=torch.float32, device=device, seed=args.seed)
            inputs = train_dataloader.dataset[0]

            pipe.transformer.eval()
            with torch.inference_mode():
                frames = pipe(
                    image=None,
                    video=inputs["video"].unsqueeze(0).to(device),
                    prompt=inputs["caption"],
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    latents=noises,  # ensure architecture invariant generation
                    height=args.height,
                    width=args.width,
                ).frames[0]

            export_to_video(frames, os.path.join(args.output_dir, "eval_output.mp4"), fps=16)

    accelerator.end_training()


if __name__ == "__main__":
    main()
