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

import inspect
import os
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoProcessor

from ...models import HiDreamO1Transformer2DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TIMESTEP_TOKEN_NUM = 1
PATCH_SIZE = 32
T_EPS = 0.001
FULL_NOISE_SCALE = 8.0
DEV_FLASH_NOISE_SCALE = 7.5
DEV_FLASH_NOISE_CLIP_STD = 2.5

PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]

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

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import HiDreamO1ImagePipeline

        >>> pipe = HiDreamO1ImagePipeline.from_pretrained(
        ...     "HiDream-ai/HiDream-O1-Image",
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe.to("cuda")
        >>> image = pipe(
        ...     "A cinematic portrait of a glass astronaut standing in a neon-lit botanical garden.",
        ...     generator=torch.Generator("cuda").manual_seed(32),
        ... ).images[0]
        >>> image.save("hidream_o1.png")
        ```
"""


def _find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    image_ratio = width / height
    best_resolution = None
    min_diff = float("inf")
    for candidate_width, candidate_height in PREDEFINED_RESOLUTIONS:
        ratio = candidate_width / candidate_height
        diff = abs(ratio - image_ratio)
        if diff < min_diff:
            min_diff = diff
            best_resolution = (candidate_width, candidate_height)
    return best_resolution


def _patchify(image: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
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


def _unpatchify(patches: torch.Tensor, height: int, width: int, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    batch_size, _, patch_dim = patches.shape
    channels = patch_dim // (patch_size * patch_size)
    height_patches = height // patch_size
    width_patches = width // patch_size
    patches = patches.reshape(batch_size, height_patches, width_patches, channels, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.reshape(batch_size, channels, height, width)


def _get_rope_index_fix_point(
    spatial_merge_size,
    image_token_id,
    video_token_id,
    vision_start_token_id,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    skip_vision_start_token=None,
    fix_point=4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fix_point > 0:
                        fix_point = fix_point - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fix_point + st_idx)
                    fix_point = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(
            3, input_ids.shape[0], -1
        )
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
    return position_ids, mrope_position_deltas


def _add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in sample.items()}


def _get_module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


def _get_module_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def _maybe_set_scheduler_shift(scheduler, shift: float):
    if hasattr(scheduler, "set_shift"):
        scheduler.set_shift(shift)
    elif hasattr(scheduler, "register_to_config") and hasattr(scheduler, "config"):
        if hasattr(scheduler.config, "flow_shift"):
            scheduler.register_to_config(flow_shift=shift)
        elif hasattr(scheduler.config, "shift"):
            scheduler.register_to_config(shift=shift)


def _set_timesteps(scheduler, num_inference_steps: int, timesteps: Optional[list[int]], device: torch.device):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if accepts_timesteps:
            scheduler.set_timesteps(timesteps=timesteps, device=device)
        else:
            scheduler.set_timesteps(len(timesteps), device=device)
            scheduler.timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
            sigmas = [float(timestep) / 1000.0 for timestep in timesteps]
            sigmas.append(0.0)
            scheduler.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler.timesteps


class HiDreamO1ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for HiDream-O1 text-to-image generation.

    HiDream-O1 predicts raw RGB image patches directly and therefore does not use a VAE. This pipeline prepares the
    Qwen3-VL chat prompt, constructs O1 multimodal RoPE positions, denoises patchified RGB noise, and unpatchifies the
    final patch tensor into images.

    Args:
        processor (`AutoProcessor`):
            Qwen3-VL processor used for the chat template and tokenizer.
        transformer ([`HiDreamO1Transformer2DModel`]):
            O1-compatible Qwen3-VL transformer that predicts RGB patches.
        scheduler ([`SchedulerMixin`], *optional*):
            Scheduler used to update the raw RGB patch tensor. Defaults to [`UniPCMultistepScheduler`] configured for
            flow prediction with `flow_shift=3.0`.
    """

    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["patches"]

    def __init__(
        self,
        processor: AutoProcessor,
        transformer: HiDreamO1Transformer2DModel,
        scheduler: Optional[UniPCMultistepScheduler] = None,
    ):
        super().__init__()

        if scheduler is None:
            scheduler = UniPCMultistepScheduler(
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=3.0,
            )

        self.register_modules(
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        if processor is not None:
            _add_special_tokens(_get_tokenizer(processor))
        self.default_sample_size = 2048
        self._attention_kwargs = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Load either a native Diffusers pipeline directory or the official Transformers-style HiDream-O1 checkpoint.
        """
        processor = kwargs.pop("processor", None)
        transformer = kwargs.pop("transformer", None)
        scheduler = kwargs.pop("scheduler", None)

        path = os.fspath(pretrained_model_name_or_path)
        is_local_diffusers_pipeline = os.path.isdir(path) and os.path.isfile(os.path.join(path, "model_index.json"))
        if is_local_diffusers_pipeline and processor is None and transformer is None:
            passed_components = {}
            if scheduler is not None:
                passed_components["scheduler"] = scheduler
            return super().from_pretrained(pretrained_model_name_or_path, **passed_components, **kwargs)

        if processor is None or transformer is None:
            try:
                passed_components = {}
                if processor is not None:
                    passed_components["processor"] = processor
                if transformer is not None:
                    passed_components["transformer"] = transformer
                if scheduler is not None:
                    passed_components["scheduler"] = scheduler
                return super().from_pretrained(pretrained_model_name_or_path, **passed_components, **kwargs)
            except (OSError, ValueError) as error:
                if "model_index.json" not in str(error):
                    raise
                logger.info(
                    "No Diffusers model_index.json found for HiDream-O1. Falling back to official checkpoint loading."
                )

        shared_load_keys = (
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "token",
            "trust_remote_code",
        )
        model_load_keys = shared_load_keys + (
            "device_map",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
            "torch_dtype",
            "variant",
            "use_safetensors",
        )
        processor_kwargs = {key: kwargs[key] for key in shared_load_keys if key in kwargs}
        transformer_kwargs = {key: kwargs[key] for key in model_load_keys if key in kwargs}

        if processor is None:
            processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
        if transformer is None:
            transformer = HiDreamO1Transformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                **transformer_kwargs,
            )
        if scheduler is None:
            scheduler = UniPCMultistepScheduler(
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=3.0,
            )

        return cls(processor=processor, transformer=transformer, scheduler=scheduler)

    def _build_text_to_image_sample(
        self,
        prompt: str,
        height: int,
        width: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        tokenizer = _get_tokenizer(self.processor)
        model_config = self.transformer.qwen_config
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

        messages = [{"role": "user", "content": prompt}]
        template_caption = (
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + tokenizer.boi_token
            + tokenizer.tms_token * TIMESTEP_TOKEN_NUM
        )
        input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

        image_grid_thw = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
        vision_tokens = torch.full((1, image_len), image_token_id, dtype=input_ids.dtype)
        vision_tokens[0, 0] = vision_start_token_id
        input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

        position_ids, _ = _get_rope_index_fix_point(
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

        text_seq_len = input_ids.shape[-1]
        all_seq_len = position_ids.shape[-1]
        token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
        start = text_seq_len - TIMESTEP_TOKEN_NUM
        token_types[0, start : start + image_len + TIMESTEP_TOKEN_NUM] = 1
        token_types[0, text_seq_len - TIMESTEP_TOKEN_NUM : text_seq_len] = 3

        sample = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_types": (token_types > 0).to(token_types.dtype),
            "vinput_mask": token_types == 1,
        }
        return _to_device(sample, device)

    def check_inputs(
        self,
        prompt: str,
        height: int,
        width: int,
        output_type: str,
        use_resolution_binning: bool,
    ):
        if not isinstance(prompt, str):
            raise TypeError("`prompt` must be a string. Batched prompts are not implemented for HiDreamO1ImagePipeline.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("`output_type` must be one of 'pil', 'np', or 'pt'.")
        if height <= 0 or width <= 0:
            raise ValueError("`height` and `width` must be positive.")
        if not use_resolution_binning and (height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0):
            raise ValueError(f"`height` and `width` must be divisible by {PATCH_SIZE} when resolution binning is off.")

    def prepare_image_size(self, height: int, width: int, use_resolution_binning: bool) -> tuple[int, int]:
        if use_resolution_binning:
            width, height = _find_closest_resolution(width, height)
        return height, width

    def _prepare_generation_defaults(
        self,
        model_type: str,
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        shift: Optional[float],
        timesteps: Optional[list[int]],
        noise_scale_start: Optional[float],
        noise_scale_end: Optional[float],
        noise_clip_std: Optional[float],
    ):
        if model_type not in {"full", "dev"}:
            raise ValueError("`model_type` must be 'full' or 'dev'.")

        if model_type == "dev":
            num_inference_steps = 28 if num_inference_steps is None else num_inference_steps
            guidance_scale = 0.0 if guidance_scale is None else guidance_scale
            shift = 1.0 if shift is None else shift
            timesteps = DEFAULT_TIMESTEPS if timesteps is None else timesteps
        else:
            num_inference_steps = 50 if num_inference_steps is None else num_inference_steps
            guidance_scale = 5.0 if guidance_scale is None else guidance_scale
            shift = 3.0 if shift is None else shift

        if noise_scale_start is None:
            noise_scale_start = DEV_FLASH_NOISE_SCALE if model_type == "dev" else FULL_NOISE_SCALE
        if noise_scale_end is None:
            noise_scale_end = DEV_FLASH_NOISE_SCALE if model_type == "dev" else noise_scale_start
        if noise_clip_std is None:
            noise_clip_std = DEV_FLASH_NOISE_CLIP_STD if model_type == "dev" else 0.0

        return num_inference_steps, guidance_scale, shift, timesteps, noise_scale_start, noise_scale_end, noise_clip_std

    def _forward_transformer(
        self,
        sample: dict[str, torch.Tensor],
        patches: torch.Tensor,
        timestep: torch.Tensor,
        attention_kwargs: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=sample["input_ids"],
            position_ids=sample["position_ids"],
            vinputs=patches,
            timestep=timestep.reshape(-1),
            token_types=sample["token_types"],
            attention_kwargs=attention_kwargs,
        )
        return outputs.sample[0, sample["vinput_mask"][0]].unsqueeze(0)

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        shift: Optional[float] = None,
        timesteps: Optional[list[int]] = None,
        generator: Optional[torch.Generator] = None,
        model_type: str = "full",
        noise_scale_start: Optional[float] = None,
        noise_scale_end: Optional[float] = None,
        noise_clip_std: Optional[float] = None,
        attention_kwargs: Optional[dict[str, Any]] = None,
        use_flash_attn: Optional[bool] = None,
        use_resolution_binning: bool = True,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> ImagePipelineOutput | tuple:
        r"""
        Generate an image from a text prompt.

        Args:
            prompt (`str`):
                Text prompt to guide image generation.
            height (`int`, defaults to 2048):
                Requested output height. When `use_resolution_binning=True`, this is snapped to a supported bucket.
            width (`int`, defaults to 2048):
                Requested output width. When `use_resolution_binning=True`, this is snapped to a supported bucket.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Defaults to 50 for `model_type="full"` and 28 for `model_type="dev"`.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance scale. Defaults to 5.0 for `model_type="full"` and 0.0 for
                `model_type="dev"`.
            shift (`float`, *optional*):
                Flow matching timestep shift. Defaults to 3.0 for `model_type="full"` and 1.0 for `model_type="dev"`.
            timesteps (`list[int]`, *optional*):
                Optional custom timestep schedule.
            generator (`torch.Generator`, *optional*):
                Random generator for deterministic noise sampling.
            model_type (`str`, defaults to `"full"`):
                Generation preset. Use `"full"` for the released full model and `"dev"` for the dev preset.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed to [`HiDreamO1AttnProcessor`].
            use_flash_attn (`bool`, *optional*):
                Deprecated convenience flag. Pass `attention_kwargs={"use_flash_attn": ...}` instead.
            use_resolution_binning (`bool`, defaults to `True`):
                Whether to snap `height` and `width` to one of the official high-resolution buckets.
            output_type (`str`, defaults to `"pil"`):
                Output format. One of `"pil"`, `"np"`, or `"pt"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return an [`ImagePipelineOutput`] instead of a tuple.

        Examples:

        Returns:
            [`ImagePipelineOutput`] or `tuple`:
                Generated images.
        """
        self.check_inputs(prompt, height, width, output_type, use_resolution_binning)
        height, width = self.prepare_image_size(height, width, use_resolution_binning)
        attention_kwargs = {} if attention_kwargs is None else dict(attention_kwargs)
        if use_flash_attn is not None:
            attention_kwargs["use_flash_attn"] = use_flash_attn
        self._attention_kwargs = attention_kwargs
        (
            num_inference_steps,
            guidance_scale,
            shift,
            timesteps,
            noise_scale_start,
            noise_scale_end,
            noise_clip_std,
        ) = self._prepare_generation_defaults(
            model_type=model_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            timesteps=timesteps,
            noise_scale_start=noise_scale_start,
            noise_scale_end=noise_scale_end,
            noise_clip_std=noise_clip_std,
        )

        device = _get_module_device(self.transformer)
        dtype = _get_module_dtype(self.transformer)
        cond_sample = self._build_text_to_image_sample(prompt, height, width, device)
        samples = [cond_sample]
        if guidance_scale > 1.0:
            samples.append(self._build_text_to_image_sample(" ", height, width, device))

        image_noise = randn_tensor(
            (1, 3, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        image_noise = noise_scale_start * image_noise.to(device=device, dtype=dtype)
        patches = _patchify(image_noise, PATCH_SIZE)

        _maybe_set_scheduler_shift(self.scheduler, shift)
        scheduler_timesteps = _set_timesteps(self.scheduler, num_inference_steps, timesteps, device)
        if len(scheduler_timesteps) > 1:
            noise_scale_schedule = [
                noise_scale_start + (noise_scale_end - noise_scale_start) * step / (len(scheduler_timesteps) - 1)
                for step in range(len(scheduler_timesteps))
            ]
        else:
            noise_scale_schedule = [noise_scale_start]

        autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        step_kwargs = {}
        step_signature = set(inspect.signature(self.scheduler.step).parameters.keys())
        if "generator" in step_signature:
            step_kwargs["generator"] = generator

        with self.progress_bar(total=len(scheduler_timesteps)) as progress_bar:
            for step_idx, step_t in enumerate(scheduler_timesteps):
                step_t = step_t.to(device=device, dtype=torch.float32)
                t_pixeldit = 1.0 - step_t / 1000.0
                sigma = (step_t / 1000.0).clamp_min(T_EPS)

                with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled, cache_enabled=False):
                    x_pred_cond = self._forward_transformer(
                        samples[0], patches.clone(), t_pixeldit, self.attention_kwargs
                    )
                v_cond = (x_pred_cond.float() - patches.float()) / sigma

                if len(samples) > 1:
                    with torch.autocast(device.type, dtype=dtype, enabled=autocast_enabled, cache_enabled=False):
                        x_pred_uncond = self._forward_transformer(
                            samples[1], patches.clone(), t_pixeldit, self.attention_kwargs
                        )
                    v_uncond = (x_pred_uncond.float() - patches.float()) / sigma
                    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_guided = v_cond

                model_output = -v_guided
                current_step_kwargs = dict(step_kwargs)
                if "s_noise" in step_signature:
                    current_step_kwargs["s_noise"] = noise_scale_schedule[step_idx]
                if "noise_clip_std" in step_signature:
                    current_step_kwargs["noise_clip_std"] = noise_clip_std

                patches = self.scheduler.step(
                    model_output.float(),
                    step_t,
                    patches.float(),
                    return_dict=False,
                    **current_step_kwargs,
                )[0].to(dtype)
                progress_bar.update()

        image = (patches + 1) / 2
        image = _unpatchify(image.float(), height, width, PATCH_SIZE)

        if output_type == "pt":
            images = image
        else:
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = np.round(np.clip(image * 255, 0, 255)).astype(np.uint8)
            if output_type == "pil":
                images = self.numpy_to_pil(image)
            else:
                images = image

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
