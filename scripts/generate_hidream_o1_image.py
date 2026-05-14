# Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32
T_EPS = 0.001
FULL_NOISE_SCALE = 8.0
DEV_FLASH_NOISE_SCALE = 7.5
DEV_FLASH_NOISE_CLIP_STD = 2.5
DEFAULT_TIMESTEPS = [
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
]


def parse_args():
    parser = argparse.ArgumentParser("Generate an image with HiDreamO1Transformer2DModel")
    parser.add_argument("--model_path", default="HiDream-ai/HiDream-O1-Image")
    parser.add_argument(
        "--official_repo",
        default=os.environ.get("HIDREAM_O1_OFFICIAL_REPO", "/tmp/HiDream-O1-Image"),
        help="Path to the official HiDream-O1-Image repo. The script reuses its schedulers and RoPE helper.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic portrait of a glass astronaut standing in a neon-lit botanical garden, "
            "highly detailed, sharp focus, natural skin tones, 35mm film still."
        ),
    )
    parser.add_argument("--output_image", default="hidream_o1_output.png")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--model_type", choices=["full", "dev"], default="full")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--scheduler", choices=["default", "flow_match", "flash"], default=None)
    parser.add_argument("--noise_scale_start", type=float, default=None)
    parser.add_argument("--noise_scale_end", type=float, default=None)
    parser.add_argument("--noise_clip_std", type=float, default=None)
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional device_map passed to from_pretrained, for example `cuda` or `auto`.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--use_flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the optimized flash-attn kernel for O1 two-pass attention. Disable to use PyTorch SDPA.",
    )
    parser.add_argument(
        "--use_resolution_binning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Snap the requested size to the official predefined high-resolution buckets.",
    )
    return parser.parse_args()


def import_runtime_dependencies():
    global AutoProcessor
    global FlowMatchEulerDiscreteScheduler
    global HiDreamO1Transformer2DModel
    global Image
    global np
    global torch

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from diffusers import FlowMatchEulerDiscreteScheduler, HiDreamO1Transformer2DModel


def import_official_helpers(official_repo: str):
    if not os.path.isdir(official_repo):
        raise FileNotFoundError(
            f"Official repo not found at {official_repo!r}. "
            "Set HIDREAM_O1_OFFICIAL_REPO or pass --official_repo."
        )

    if official_repo not in sys.path:
        sys.path.insert(0, official_repo)

    from models.flash_scheduler import FlashFlowMatchEulerDiscreteScheduler
    from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from models.utils import find_closest_resolution, get_rope_index_fix_point

    return (
        FlowUniPCMultistepScheduler,
        FlashFlowMatchEulerDiscreteScheduler,
        find_closest_resolution,
        get_rope_index_fix_point,
    )


def add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def get_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def get_module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, channels, height, width = image.shape
    image = image.reshape(
        batch_size,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    image = image.permute(0, 2, 4, 1, 3, 5)
    return image.reshape(batch_size, -1, channels * patch_size * patch_size)


def unpatchify(patches: torch.Tensor, height: int, width: int, patch_size: int) -> torch.Tensor:
    batch_size, _, patch_dim = patches.shape
    channels = patch_dim // (patch_size * patch_size)
    h_patches = height // patch_size
    w_patches = width // patch_size
    patches = patches.reshape(batch_size, h_patches, w_patches, channels, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.reshape(batch_size, channels, height, width)


def build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config, get_rope_index_fix_point):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + tokenizer.boi_token
        + tokenizer.tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
    vision_tokens = torch.full((1, image_len), image_token_id, dtype=input_ids.dtype)
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]
    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    start = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, start : start + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": (token_types > 0).to(token_types.dtype),
        "vinput_mask": token_types == 1,
    }


def build_scheduler(
    scheduler_name,
    num_inference_steps,
    timesteps_list,
    shift,
    device,
    FlowUniPCMultistepScheduler,
    FlashFlowMatchEulerDiscreteScheduler,
):
    if scheduler_name == "flash":
        scheduler = FlashFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False
        )
    elif scheduler_name == "flow_match":
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    else:
        scheduler = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)

    scheduler.set_timesteps(num_inference_steps, device=device)
    if timesteps_list is not None:
        scheduler.timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in scheduler.timesteps]
        sigmas.append(0.0)
        scheduler.sigmas = torch.tensor(sigmas, device=device)
    return scheduler


def to_device(sample, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in sample.items()}

def generate_text_to_image(
    transformer,
    processor,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    scheduler_name: str,
    timesteps_list: Optional[list[int]],
    seed: int,
    attention_kwargs: Optional[dict],
    noise_scale_start: float,
    noise_scale_end: float,
    noise_clip_std: float,
    FlowUniPCMultistepScheduler,
    FlashFlowMatchEulerDiscreteScheduler,
    get_rope_index_fix_point,
) -> Image.Image:
    device = get_module_device(transformer)
    dtype = next(transformer.parameters()).dtype
    model_config = transformer.qwen_config
    tokenizer = get_tokenizer(processor)

    cond_sample = build_t2i_text_sample(
        prompt, height, width, tokenizer, processor, model_config, get_rope_index_fix_point
    )
    samples = [to_device(cond_sample, device)]
    if guidance_scale > 1.0:
        uncond_sample = build_t2i_text_sample(
            " ", height, width, tokenizer, processor, model_config, get_rope_index_fix_point
        )
        samples.append(to_device(uncond_sample, device))

    noise = noise_scale_start * torch.randn(
        (1, 3, height, width),
        generator=torch.Generator("cpu").manual_seed(seed + 1),
    ).to(device=device, dtype=dtype)
    z = patchify(noise, PATCH_SIZE)

    scheduler = build_scheduler(
        scheduler_name,
        num_inference_steps,
        timesteps_list,
        shift,
        device,
        FlowUniPCMultistepScheduler,
        FlashFlowMatchEulerDiscreteScheduler,
    )

    if len(scheduler.timesteps) > 1:
        noise_scale_schedule = [
            noise_scale_start + (noise_scale_end - noise_scale_start) * step / (len(scheduler.timesteps) - 1)
            for step in range(len(scheduler.timesteps))
        ]
    else:
        noise_scale_schedule = [noise_scale_start]

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    def forward_once(sample, z_in, t_pixeldit):
        autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled, cache_enabled=False):
            outputs = transformer(
                input_ids=sample["input_ids"],
                position_ids=sample["position_ids"],
                vinputs=z_in,
                timestep=t_pixeldit.reshape(-1).to(device),
                token_types=sample["token_types"],
                attention_kwargs=attention_kwargs,
            )
        return outputs.sample[0, sample["vinput_mask"][0]].unsqueeze(0)

    for step_idx, step_t in enumerate(tqdm(scheduler.timesteps, desc="Generating")):
        t_pixeldit = 1.0 - step_t.float() / 1000.0
        sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

        x_pred_cond = forward_once(samples[0], z.clone(), t_pixeldit)
        v_cond = (x_pred_cond.float() - z.float()) / sigma

        if len(samples) > 1:
            x_pred_uncond = forward_once(samples[1], z.clone(), t_pixeldit)
            v_uncond = (x_pred_uncond.float() - z.float()) / sigma
            v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v_guided = v_cond

        model_output = -v_guided
        if scheduler_name == "flash":
            z = scheduler.step(
                model_output.float(),
                step_t.to(dtype=torch.float32),
                z.float(),
                s_noise=noise_scale_schedule[step_idx],
                noise_clip_std=noise_clip_std,
                return_dict=False,
            )[0].to(dtype)
        else:
            z = scheduler.step(model_output.float(), step_t.to(dtype=torch.float32), z.float(), return_dict=False)[
                0
            ].to(dtype)

    image = (z + 1) / 2
    image = unpatchify(image.float().cpu(), height, width, PATCH_SIZE)
    array = np.round(np.clip(image[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def main():
    args = parse_args()
    import_runtime_dependencies()

    (
        FlowUniPCMultistepScheduler,
        FlashFlowMatchEulerDiscreteScheduler,
        find_closest_resolution,
        get_rope_index_fix_point,
    ) = import_official_helpers(args.official_repo)

    if args.use_resolution_binning:
        width, height = find_closest_resolution(args.width, args.height)
        if (width, height) != (args.width, args.height):
            print(f"[hidream-o1] Resolution snapped from {args.width}x{args.height} to {width}x{height}")
    else:
        width, height = args.width, args.height
        if width % PATCH_SIZE != 0 or height % PATCH_SIZE != 0:
            raise ValueError(
                f"Width and height must be divisible by {PATCH_SIZE} when resolution binning is disabled."
            )

    if args.model_type == "dev":
        num_inference_steps = args.num_inference_steps or 28
        guidance_scale = 0.0 if args.guidance_scale is None else args.guidance_scale
        shift = 1.0 if args.shift is None else args.shift
        scheduler_name = args.scheduler or "flash"
        timesteps_list = DEFAULT_TIMESTEPS
    else:
        num_inference_steps = args.num_inference_steps or 50
        guidance_scale = 5.0 if args.guidance_scale is None else args.guidance_scale
        shift = 3.0 if args.shift is None else args.shift
        scheduler_name = args.scheduler or "default"
        timesteps_list = None

    if args.noise_scale_start is None:
        noise_scale_start = (
            DEV_FLASH_NOISE_SCALE if args.model_type == "dev" and scheduler_name == "flash" else FULL_NOISE_SCALE
        )
    else:
        noise_scale_start = args.noise_scale_start
    if args.noise_scale_end is None:
        noise_scale_end = (
            DEV_FLASH_NOISE_SCALE if args.model_type == "dev" and scheduler_name == "flash" else FULL_NOISE_SCALE
        )
    else:
        noise_scale_end = args.noise_scale_end
    if args.noise_clip_std is None:
        noise_clip_std = DEV_FLASH_NOISE_CLIP_STD if args.model_type == "dev" and scheduler_name == "flash" else 0.0
    else:
        noise_clip_std = args.noise_clip_std

    dtype = get_torch_dtype(args.torch_dtype)
    load_kwargs = {"torch_dtype": dtype, "local_files_only": args.local_files_only}
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map

    print(f"[hidream-o1] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    add_special_tokens(get_tokenizer(processor))

    print(f"[hidream-o1] Loading transformer from {args.model_path}")
    transformer = HiDreamO1Transformer2DModel.from_pretrained(args.model_path, **load_kwargs).eval()
    if args.device_map is None:
        transformer.to(torch.device(args.device))

    attention_kwargs = {"use_flash_attn": args.use_flash_attn}
    if not attention_kwargs["use_flash_attn"] and (height * width) >= 1024 * 1024:
        print("[hidream-o1] Warning: PyTorch SDPA attention at high resolution can be slower than flash-attn.")

    with torch.no_grad():
        image = generate_text_to_image(
            transformer=transformer,
            processor=processor,
            prompt=args.prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            scheduler_name=scheduler_name,
            timesteps_list=timesteps_list,
            seed=args.seed,
            attention_kwargs=attention_kwargs,
            noise_scale_start=noise_scale_start,
            noise_scale_end=noise_scale_end,
            noise_clip_std=noise_clip_std,
            FlowUniPCMultistepScheduler=FlowUniPCMultistepScheduler,
            FlashFlowMatchEulerDiscreteScheduler=FlashFlowMatchEulerDiscreteScheduler,
            get_rope_index_fix_point=get_rope_index_fix_point,
        )

    output_dir = os.path.dirname(os.path.abspath(args.output_image))
    os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_image)
    print(f"[hidream-o1] Saved image to {args.output_image}")


if __name__ == "__main__":
    main()
