# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# This pipeline is adapted from https://github.com/jd-opensource/JoyAI-Image

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen3VLForConditionalGeneration

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_joyai_image import JoyAIImageVAE
from diffusers.models.transformers.transformer_joyai_image import JoyAIImageTransformer3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, empty_device_cache, get_device
from diffusers.schedulers.scheduling_joyai_flow_match_discrete import JoyAIFlowMatchDiscreteScheduler
from diffusers.utils import is_accelerate_available, is_accelerate_version, logging
from diffusers.utils.torch_utils import randn_tensor

from .pipeline_output import JoyAIImagePipelineOutput


logger = logging.get_logger(__name__)

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


PROMPT_TEMPLATE_ENCODE = {
    "image": "<|im_start|>system\n \nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    "multiple_images": "<|im_start|>system\n \nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n{}<|im_start|>assistant\n",
    "video": "<|im_start|>system\n \nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
}

PROMPT_TEMPLATE_START_IDX = {
    "image": 34,
    "multiple_images": 34,
    "video": 91,
}


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class JoyAIImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["processor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: JoyAIImageVAE,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: PreTrainedTokenizerBase,
        transformer: JoyAIImageTransformer3DModel,
        scheduler: JoyAIFlowMatchDiscreteScheduler,
        processor: Any | None = None,
        args: Any | None = None,
    ):
        super().__init__()
        self.args = args

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.enable_multi_task = bool(getattr(self.args, "enable_multi_task_training", False))
        if hasattr(self.vae, "ffactor_spatial"):
            self.vae_scale_factor = self.vae.ffactor_spatial
            self.vae_scale_factor_temporal = self.vae.ffactor_temporal
        else:
            self.vae_scale_factor = 8
            self.vae_scale_factor_temporal = 4
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.qwen_processor = processor
        text_encoder_ckpt = None
        text_encoder_cfg = getattr(self.args, "text_encoder_arch_config", None)
        if isinstance(text_encoder_cfg, dict):
            text_encoder_params = text_encoder_cfg.get("params", {})
            text_encoder_ckpt = text_encoder_params.get("text_encoder_ckpt")
        if self.qwen_processor is None and text_encoder_ckpt is not None:
            self.qwen_processor = AutoProcessor.from_pretrained(
                text_encoder_ckpt,
                local_files_only=True,
                trust_remote_code=True,
            )

        self.text_token_max_length = int(getattr(self.args, "text_token_max_length", 2048))
        self.prompt_template_encode = PROMPT_TEMPLATE_ENCODE
        self.prompt_template_encode_start_idx = PROMPT_TEMPLATE_START_IDX
        self._joyai_force_vae_fp32 = True

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        template_type: str = "image",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._get_runtime_execution_device()
        dtype = dtype or next(self.text_encoder.parameters()).dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]
        formatted_prompts = [template.format(prompt_text) for prompt_text in prompt]
        txt_tokens = self.tokenizer(
            formatted_prompts,
            max_length=self.text_token_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self._run_text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = min(
            self.text_token_max_length,
            max(u.size(0) for u in split_hidden_states),
            max(u.size(0) for u in attn_mask_list),
        )
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [
                        hidden_state,
                        hidden_state.new_zeros(max_seq_len - hidden_state.size(0), hidden_state.size(1)),
                    ]
                )
                for hidden_state in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([attention_mask_row, attention_mask_row.new_zeros(max_seq_len - attention_mask_row.size(0))])
                for attention_mask_row in attn_mask_list
            ]
        )
        return prompt_embeds.to(dtype=dtype, device=device), encoder_attention_mask

    def encode_prompt_multiple_images(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        images: Optional[List[Any]] = None,
        template_type: str = "multiple_images",
        max_sequence_length: Optional[int] = None,
        drop_vit_feature: bool = False,
    ):
        if self.qwen_processor is None:
            raise ValueError("Qwen processor is required for JoyAI image-edit prompt encoding.")
        device = device or self._get_runtime_execution_device()
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]
        prompt = [p.replace("<image>\n", "<|vision_start|><|image_pad|><|vision_end|>") for p in prompt]
        prompt = [template.format(p) for p in prompt]
        inputs = self.qwen_processor(text=prompt, images=images, padding=True, return_tensors="pt").to(device)
        encoder_hidden_states = self._run_text_encoder(**inputs)
        last_hidden_states = encoder_hidden_states.hidden_states[-1]
        if drop_vit_feature:
            input_ids = inputs["input_ids"]
            vlm_image_end_idx = torch.where(input_ids[0] == 151653)[0][-1]
            drop_idx = int(vlm_image_end_idx.item()) + 1
        prompt_embeds = last_hidden_states[:, drop_idx:]
        prompt_embeds_mask = inputs["attention_mask"][:, drop_idx:]
        if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]
        return prompt_embeds, prompt_embeds_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        images: Optional[List[Any]] = None,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        template_type: str = "image",
        drop_vit_feature: bool = False,
    ):
        if images is not None:
            return self.encode_prompt_multiple_images(
                prompt=prompt,
                images=images,
                device=device,
                max_sequence_length=max_sequence_length,
                drop_vit_feature=drop_vit_feature,
            )

        device = device or self._get_runtime_execution_device()
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, template_type, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_videos_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        height: int,
        width: int,
        images: Optional[List[Any]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ) -> None:
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("If `prompt_embeds` are provided, `prompt_embeds_mask` must also be passed.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` must also be passed."
            )

    def _vae_compute_dtype(self) -> torch.dtype:
        if getattr(self, "_joyai_force_vae_fp32", False):
            return torch.float32
        if hasattr(self.vae, "model"):
            return next(self.vae.model.parameters()).dtype
        return next(self.vae.parameters()).dtype

    def _get_runtime_execution_device(self) -> torch.device:
        override = getattr(self, "_joyai_execution_device_override", None)
        if override is not None:
            return torch.device(override)
        return self._execution_device

    def _is_sequential_cpu_offload_enabled(self) -> bool:
        return bool(getattr(self, "_joyai_sequential_cpu_offload_enabled", False))

    def enable_manual_cpu_offload(
        self,
        device: torch.device | str,
        components: Optional[List[str]] = None,
    ) -> None:
        """Enable manual CPU offload for selected components."""
        runtime_device = torch.device(device)
        component_names = set(components or ["text_encoder", "vae"])

        invalid_components = [name for name in component_names if name not in self.components]
        if invalid_components:
            raise ValueError(f"Unknown components for manual cpu offload: {invalid_components}")

        self._joyai_execution_device_override = runtime_device
        self._joyai_sequential_cpu_offload_enabled = True
        self._joyai_manual_offload_components = component_names

        for name in component_names:
            component = getattr(self, name, None)
            if isinstance(component, torch.nn.Module):
                component.to("cpu")

    def _uses_manual_sequential_offload(self, component_name: str) -> bool:
        manual_components = getattr(self, "_joyai_manual_offload_components", set())
        return self._is_sequential_cpu_offload_enabled() and component_name in manual_components

    def _offload_component_to_cpu(self, component_name: str):
        component = getattr(self, component_name, None)
        if component is None:
            return
        component.to("cpu")
        empty_device_cache(getattr(self._get_runtime_execution_device(), "type", "cuda"))

    def _run_text_encoder(self, **inputs):
        if self._uses_manual_sequential_offload("text_encoder"):
            self.text_encoder.to(self._get_runtime_execution_device())
            try:
                return self.text_encoder(**inputs, output_hidden_states=True)
            finally:
                self._offload_component_to_cpu("text_encoder")
        return self.text_encoder(**inputs, output_hidden_states=True)

    def _get_vae_scale(self, device: torch.device, dtype: torch.dtype):
        mean = getattr(self.vae, "mean", None)
        std = getattr(self.vae, "std", None)
        if mean is None or std is None:
            return None
        mean = mean.to(device=device, dtype=dtype)
        std = std.to(device=device, dtype=dtype)
        return [mean, 1.0 / std]

    def _encode_with_vae(self, videos: torch.Tensor) -> torch.Tensor:
        device = self._get_runtime_execution_device()
        vae_dtype = PRECISION_TO_TYPE.get(getattr(self.args, "vae_precision", "bf16"), videos.dtype)
        videos = videos.to(device=device, dtype=vae_dtype)

        if self._uses_manual_sequential_offload("vae") and hasattr(self.vae, "model"):
            scale = self._get_vae_scale(device=device, dtype=vae_dtype)
            self.vae.model.to(device=device, dtype=vae_dtype)
            try:
                return self.vae.model.encode(videos, scale=scale)
            finally:
                self.vae.model.to("cpu")
                empty_device_cache(device.type)

        if hasattr(self.vae, "mean"):
            self.vae.mean = self.vae.mean.to(device=device, dtype=vae_dtype)
        if hasattr(self.vae, "std"):
            self.vae.std = self.vae.std.to(device=device, dtype=vae_dtype)
        if hasattr(self.vae, "scale"):
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        if hasattr(self.vae, "config"):
            if hasattr(self.vae.config, "latents_mean"):
                self.vae.config.latents_mean = self.vae.mean
            if hasattr(self.vae.config, "latents_std"):
                self.vae.config.latents_std = self.vae.std

        self.vae.to(device=device, dtype=vae_dtype)
        encoded = self.vae.encode(videos)
        if hasattr(encoded, "latent_dist"):
            return encoded.latent_dist.sample()
        return encoded

    def _decode_with_vae(self, latents: torch.Tensor):
        device = self._get_runtime_execution_device()
        vae_dtype = self._vae_compute_dtype()
        latents = latents.to(device=device, dtype=vae_dtype)

        if self._uses_manual_sequential_offload("vae") and hasattr(self.vae, "model"):
            scale = self._get_vae_scale(device=device, dtype=vae_dtype)
            self.vae.model.to(device=device, dtype=vae_dtype)
            try:
                videos = [self.vae.model.decode(u.unsqueeze(0), scale=scale).clamp_(-1, 1).squeeze(0) for u in latents]
                return torch.stack(videos, dim=0)
            finally:
                self.vae.model.to("cpu")
                empty_device_cache(device.type)

        if hasattr(self.vae, "mean"):
            self.vae.mean = self.vae.mean.to(device=device, dtype=vae_dtype)
        if hasattr(self.vae, "std"):
            self.vae.std = self.vae.std.to(device=device, dtype=vae_dtype)
        if hasattr(self.vae, "scale"):
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        if hasattr(self.vae, "config"):
            if hasattr(self.vae.config, "latents_mean"):
                self.vae.config.latents_mean = self.vae.mean
            if hasattr(self.vae.config, "latents_std"):
                self.vae.config.latents_std = self.vae.std

        self.vae.to(device=device, dtype=vae_dtype)
        return self.vae.decode(latents, return_dict=False)[0]

    def prepare_latents(
        self,
        batch_size,
        num_items,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
        reference_images=None,
    ):
        shape = (
            batch_size,
            num_items,
            num_channels_latents,
            (video_length - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}."
            )

        if latents is None:
            if reference_images is not None and len(reference_images) > 0:
                ref_img = [torch.from_numpy(np.array(x.convert("RGB"))) for x in reference_images]
                ref_img = torch.stack(ref_img).to(device=device, dtype=dtype)
                ref_img = ref_img / 127.5 - 1.0
                ref_img = ref_img.permute(0, 3, 1, 2).unsqueeze(2)
                ref_vae = self._encode_with_vae(ref_img)
                ref_vae = ref_vae.reshape(shape[0], num_items - 1, *ref_vae.shape[1:])
                noise = randn_tensor((shape[0], 1, *shape[2:]), generator=generator, device=device, dtype=dtype)
                latents = torch.cat([ref_vae, noise], dim=1)
            else:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        if not self.enable_multi_task:
            return latents, None
        raise NotImplementedError("JoyAI multi-task conditioning is not implemented in the diffusers adaptation yet.")

    def enable_sequential_cpu_offload(self, gpu_id: int | None = None, device: torch.device | str = None):
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        self._maybe_raise_error_if_group_offload_active(raise_error=True)
        self.remove_all_hooks()

        is_pipeline_device_mapped = self._is_pipeline_device_mapped()
        if is_pipeline_device_mapped:
            raise ValueError(
                "It seems like you have activated a device mapping strategy on the pipeline so calling `enable_sequential_cpu_offload()` isn't allowed. You can call `reset_device_map()` first and then call `enable_sequential_cpu_offload()`."
            )

        if device is None:
            device = get_device()
            if device == "cpu":
                raise RuntimeError("`enable_sequential_cpu_offload` requires accelerator, but not found")

        torch_device = torch.device(device)
        device_index = torch_device.index
        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)
        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        if self.device.type != "cpu":
            orig_device_type = self.device.type
            self.to("cpu", silence_dtype_warnings=True)
            empty_device_cache(orig_device_type)

        self._joyai_manual_offload_components = {"text_encoder", "vae"}

        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
                continue

            if name in self._joyai_manual_offload_components:
                model.to("cpu")
                continue

            offload_buffers = len(model._parameters) > 0
            params = list(model.parameters())
            on_cpu = len(params) == 0 or all(param.device.type == "cpu" for param in params)
            state_dict = model.state_dict() if on_cpu else None
            cpu_offload(model, device, offload_buffers=offload_buffers, state_dict=state_dict)

        self._joyai_sequential_cpu_offload_enabled = True

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def pad_sequence(self, x: torch.Tensor, target_length: int):
        current_length = x.shape[1]
        if current_length >= target_length:
            return x[:, -target_length:]
        padding_length = target_length - current_length
        if x.ndim >= 3:
            padding = torch.zeros((x.shape[0], padding_length, *x.shape[2:]), dtype=x.dtype, device=x.device)
        else:
            padding = torch.zeros((x.shape[0], padding_length), dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=1)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_frames: int = 1,
        images: Optional[List[Any]] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 4096,
        drop_vit_feature: bool = False,
        **kwargs,
    ):
        self.check_inputs(
            prompt,
            height,
            width,
            images=images,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._get_runtime_execution_device()
        template_type = "image" if num_frames == 1 else "video"
        num_items = 1 if images is None or len(images) == 0 else 1 + len(images)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            images=images,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            template_type=template_type,
            drop_vit_feature=drop_vit_feature,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                default_negative_prompt = ""
                if num_items <= 1:
                    negative_prompt = [f"<|im_start|>user\n{default_negative_prompt}<|im_end|>\n"] * batch_size
                else:
                    image_tokens = "<image>\n" * (num_items - 1)
                    negative_prompt = [
                        f"<|im_start|>user\n{image_tokens}{default_negative_prompt}<|im_end|>\n"
                    ] * batch_size

            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                images=images,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                template_type=template_type,
            )

            max_seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            prompt_embeds = torch.cat(
                [
                    self.pad_sequence(negative_prompt_embeds, max_seq_len),
                    self.pad_sequence(prompt_embeds, max_seq_len),
                ]
            )
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = torch.cat(
                    [
                        self.pad_sequence(negative_prompt_embeds_mask, max_seq_len),
                        self.pad_sequence(prompt_embeds_mask, max_seq_len),
                    ]
                )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )

        num_channels_latents = self.vae.config.latent_channels
        latents, condition = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_items,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            reference_images=images,
        )

        target_dtype = PRECISION_TO_TYPE.get(getattr(self.args, "dit_precision", "bf16"), prompt_embeds.dtype)
        autocast_enabled = target_dtype != torch.float32 and device.type == "cuda"
        vae_dtype = self._vae_compute_dtype()
        vae_autocast_enabled = vae_dtype != torch.float32 and device.type == "cuda"

        self._num_timesteps = len(timesteps)
        if num_items > 1:
            ref_latents = latents[:, : (num_items - 1)].clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if num_items > 1:
                    latents[:, : (num_items - 1)] = ref_latents.clone()

                latents_ = torch.cat([latents, condition], dim=2) if condition is not None else latents
                latent_model_input = torch.cat([latents_] * 2) if self.do_classifier_free_guidance else latents_
                latent_model_input = latent_model_input.to(device=device, dtype=target_dtype)
                prompt_embeds_input = prompt_embeds.to(device=device, dtype=target_dtype)
                t_expand = t.repeat(latent_model_input.shape[0])

                with torch.autocast(device_type=device.type, dtype=target_dtype, enabled=autocast_enabled):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_expand,
                        encoder_hidden_states=prompt_embeds_input,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    cond_norm = torch.norm(noise_pred_text, dim=2, keepdim=True)
                    noise_norm = torch.norm(noise_pred, dim=2, keepdim=True)
                    noise_pred = noise_pred * (cond_norm / noise_norm)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.reshape(-1, *latents.shape[2:])
            with torch.autocast(device_type=device.type, dtype=vae_dtype, enabled=vae_autocast_enabled):
                decoded = self._decode_with_vae(latents)
            decoded = decoded.reshape(batch_size, num_items, *decoded.shape[1:])
            image = decoded[:, -1, :, 0]
            image = (image / 2 + 0.5).clamp(0, 1)

        self.maybe_free_model_hooks()

        if output_type == "pt":
            output_image = image.cpu().float()
        elif output_type == "pil":
            output_image = self.image_processor.numpy_to_pil(image.cpu().permute(0, 2, 3, 1).float().numpy())
        else:
            output_image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if not return_dict:
            return (output_image,)
        return JoyAIImagePipelineOutput(images=output_image)


__all__ = ["JoyAIImagePipeline"]
