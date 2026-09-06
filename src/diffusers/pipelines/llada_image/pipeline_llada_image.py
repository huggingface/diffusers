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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import (
    AutoencoderKLFlux2,
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQModel,
    LLaDAImageTextProjectionModel,
    LLaDAImageTransformer2DModel,
)
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LLaDAImagePipelineOutput


logger = logging.get_logger(__name__)


class LLaDAImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for LLaDA-Image text-to-image generation, VQ-conditioned generation, and single-image editing.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used for denoising.
        vae ([`AutoencoderKLFlux2`]):
            Flux2 VAE used to encode reference images and decode generated latents.
        text_encoder (`transformers.PreTrainedModel`):
            LLaDA2 conditional-generation model. It must expose `get_input_embeddings()` and its language backbone as
            `model`.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            Tokenizer paired with the LLaDA2 text encoder.
        queryformer ([`LLaDAImageQueryFormerModel`]):
            QueryFormer that refines the learnable generation queries.
        text_projection ([`LLaDAImageTextProjectionModel`]):
            Connector and projector that map LLaDA2 hidden states to denoiser caption features.
        sigvq ([`LLaDAImageSigVQModel`]):
            GLM SigVQ component that embeds MLLM-generated VQ tokens and encodes editing reference images.
        transformer ([`LLaDAImageTransformer2DModel`]):
            Denoising transformer.
    """

    model_cpu_offload_seq = "text_encoder->queryformer->text_projection->sigvq->transformer->vae"
    _exclude_from_cpu_offload = ["text_encoder"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        queryformer: LLaDAImageQueryFormerModel,
        text_projection: LLaDAImageTextProjectionModel,
        sigvq: LLaDAImageSigVQModel,
        transformer: LLaDAImageTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            queryformer=queryformer,
            text_projection=text_projection,
            sigvq=sigvq,
            transformer=transformer,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if self.vae is not None else 8
        self.latent_scale_factor = self.vae_scale_factor * 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    def _encode_text(
        self,
        prompts: list[str],
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        formatted_prompts = [
            "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>"
            if prompt is None
            else f"<role>HUMAN</role> Generate an image: {prompt.strip()}\n<role>ASSISTANT</role>\n<IMAGE1>"
            for prompt in prompts
        ]
        text_inputs = self.tokenizer(
            formatted_prompts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        text_encoder_device = self.text_encoder.get_input_embeddings().weight.device
        input_ids = text_inputs.input_ids.to(text_encoder_device)
        attention_mask = text_inputs.attention_mask.to(text_encoder_device).bool()
        inputs_embeds = self.text_encoder.get_input_embeddings()(input_ids)

        query_embeds = self.queryformer(
            inputs_embeds.to(device=self._execution_device, dtype=self.queryformer.dtype),
            attention_mask.to(self._execution_device),
        ).query_embeds.to(device=text_encoder_device, dtype=inputs_embeds.dtype)
        text_length = inputs_embeds.shape[1]
        inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(attention_mask.shape[0], query_embeds.shape[1])],
            dim=1,
        )
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)

        mask_value = torch.finfo(inputs_embeds.dtype).min
        backbone_attention_mask = attention_mask[:, None, None, :].expand(-1, 1, attention_mask.shape[1], -1)
        backbone_attention_mask = torch.where(
            backbone_attention_mask,
            torch.zeros((), dtype=inputs_embeds.dtype, device=text_encoder_device),
            torch.full((), mask_value, dtype=inputs_embeds.dtype, device=text_encoder_device),
        )
        backbone_attention_mask[:, :, :text_length, text_length:] = mask_value

        hidden_states = self.text_encoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=backbone_attention_mask,
            position_ids=position_ids,
            return_dict=True,
        ).last_hidden_state
        prompt_embeds = self.text_projection(
            hidden_states.to(device=self._execution_device, dtype=self.text_projection.dtype)
        ).hidden_states
        return prompt_embeds, attention_mask.to(prompt_embeds.device)

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 2048,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        device = device or self._execution_device

        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt_embeds, prompt_attention_mask = self._encode_text(prompt, max_sequence_length)
        else:
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_mask = prompt_attention_mask.to(device).bool()

        batch_size = prompt_embeds.shape[0]
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [None] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_attention_mask = self._encode_text(
                negative_prompt, max_sequence_length
            )
        elif do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device).bool()

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat_interleave(
                num_images_per_prompt, dim=0
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def generate_vq_tokens(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
    ) -> torch.Tensor:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        image_token_offset = 157184
        frontend_scale = max(max(height, width) / 512, 1.0)
        frontend_height = int(height / frontend_scale)
        frontend_width = int(width / frontend_scale)
        vq_height = frontend_height // 16
        vq_width = frontend_width // 16
        image_token_count = vq_height * vq_width
        system_prompt = "You are a text-to-image generation assistant."
        generated_tokens = []

        for prompt in prompts:
            text_prompt = f"<role>SYSTEM</role> {system_prompt} <role>HUMAN</role>{prompt}<role>ASSISTANT</role>"
            text_ids = self.tokenizer(text_prompt).input_ids
            image_info_ids = self.tokenizer(
                f"<|image|><|reserved_token_{vq_height}|><|reserved_token_{vq_width}|><boi><|/image|>"
            ).input_ids
            input_ids = text_ids + image_info_ids[:-1]

            uncond_prompt = (
                f"<role>SYSTEM</role> {system_prompt} <role>HUMAN</role><uncondition><role>ASSISTANT</role>"
            )
            uncond_ids = self.tokenizer(uncond_prompt).input_ids + image_info_ids[:-1]
            output_ids = self.text_encoder.generate_bd_image_logic(
                data={
                    "input_ids": torch.tensor(
                        input_ids, device=self.text_encoder.get_input_embeddings().weight.device
                    ).unsqueeze(0),
                    "uncond_ids": uncond_ids,
                },
                block_length=32,
                steps=8,
                gen_length=image_token_count,
                cfg_scale=2.0,
            )
            token_ids = output_ids[0, len(input_ids) : len(input_ids) + image_token_count] - image_token_offset
            if len(token_ids) != image_token_count:
                raise ValueError(f"The MLLM generated {len(token_ids)} VQ tokens, expected {image_token_count}.")
            if torch.any((token_ids < 0) | (token_ids >= self.sigvq.config.codebook_size)):
                raise ValueError("The MLLM generated token IDs outside the SigVQ codebook.")
            generated_tokens.append(token_ids)

        return torch.stack(generated_tokens)

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        image: PipelineImageInput | None,
        generation_mode: str,
        height: int,
        width: int,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        callback_on_step_end_tensor_inputs: list[str],
        num_inference_steps: int,
    ) -> None:
        if generation_mode not in {"text", "vq", "editing"}:
            raise ValueError("`generation_mode` must be one of 'text', 'vq', or 'editing'.")
        if generation_mode in {"text", "vq"} and image is not None:
            raise ValueError(f"`image` must be omitted when `generation_mode='{generation_mode}'`.")
        if generation_mode == "vq" and prompt is None:
            raise ValueError("`prompt` is required when `generation_mode='vq'`.")
        if generation_mode == "editing" and image is None:
            raise ValueError("`image` is required when `generation_mode='editing'`.")
        if generation_mode == "vq" and (height % 16 != 0 or width % 16 != 0):
            raise ValueError("`height` and `width` must be divisible by 16 in VQ mode.")

        required_multiple = self.latent_scale_factor * (2 if generation_mode == "editing" else 1)
        if height <= 0 or width <= 0 or height % required_multiple != 0 or width % required_multiple != 0:
            raise ValueError(f"`height` and `width` must be divisible by {required_multiple}.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be at least 1.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide only one of `prompt` or `prompt_embeds`.")
        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("`prompt_attention_mask` is required with `prompt_embeds`.")
        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("`negative_prompt_attention_mask` is required with `negative_prompt_embeds`.")
        if num_images_per_prompt < 1:
            raise ValueError("`num_images_per_prompt` must be at least 1.")
        if not all(name in self._callback_tensor_inputs for name in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` must be chosen from {self._callback_tensor_inputs}."
            )

    def _encode_source_image(
        self,
        image: PipelineImageInput,
        height: int,
        width: int,
        batch_size: int,
        num_images_per_prompt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image_processor.preprocess(image, height=height, width=width)
        if image.shape[0] == 1 and batch_size > 1:
            image = image.repeat(batch_size, 1, 1, 1)
        if image.shape[0] != batch_size:
            raise ValueError(f"The image batch size must be 1 or {batch_size}, but is {image.shape[0]}.")
        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        sigvq_pixel_values = F.interpolate(
            image.float(),
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=False,
        )
        semantic_features = self.sigvq(
            sigvq_pixel_values.to(device=self._execution_device, dtype=self.sigvq.dtype)
        ).semantic_features

        source_latents = self.vae.encode(
            image.to(device=self._execution_device, dtype=self.vae.dtype)
        ).latent_dist.mode()
        source_latents = self._patchify_latents(source_latents)
        latent_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(source_latents)
        latent_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            source_latents
        )
        source_latents = (source_latents - latent_mean) / latent_std
        return source_latents, semantic_features

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: PipelineImageInput | None = None,
        generation_mode: str = "text",
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 2048,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[["LLaDAImagePipeline", int, torch.Tensor, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> LLaDAImagePipelineOutput | tuple:
        r"""
        Generate images using text-only, VQ-conditioned, or editing inference.

        The timestep schedule is selected by the scheduler configuration. `use_uniform_sigmas=True` uses a uniform
        pre-shift grid; otherwise the source Kumaraswamy schedule is used.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompts that describe the generated image or requested edit.
            image (`PipelineImageInput`, *optional*):
                Reference image or image batch. Required in `"editing"` mode and rejected in other modes.
            generation_mode (`str`, defaults to `"text"`):
                Inference path. `"text"` uses only the text prompt. `"vq"` uses the MLLM to generate VQ tokens from the
                prompt at a maximum frontend resolution of 512 before diffusion. `"editing"` uses both reference-image
                SigVQ features and source-image latents.
            negative_prompt (`str` or `list[str]`, *optional*):
                Text excluded from generation. The checkpoint's empty CFG prompt is used by default.
            height (`int`, defaults to `1024`):
                Output image height.
            width (`int`, defaults to `1024`):
                Output image width.
            num_inference_steps (`int`, defaults to `20`):
                Number of flow-matching denoising steps.
            guidance_scale (`float`, defaults to `4.5`):
                Classifier-free guidance scale. Guidance is disabled at values up to `1.0`.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images generated per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator or generator batch used to create the initial latents.
            latents (`torch.Tensor`, *optional*):
                Pre-generated patchified Flux2 latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed, projected positive prompt embeddings.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Valid-token mask for `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed, projected negative prompt embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Valid-token mask for `negative_prompt_embeds`.
            max_sequence_length (`int`, defaults to `2048`):
                Maximum text sequence length before the QueryFormer tokens are appended.
            output_type (`str`, defaults to `"pil"`):
                Output format. Choose `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return [`LLaDAImagePipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable`, *optional*):
                Function called after each denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensor names forwarded to `callback_on_step_end`.

        Returns:
            [`LLaDAImagePipelineOutput`] or `tuple`:
                Generated images or final patchified latents.
        """
        self.check_inputs(
            prompt,
            image,
            generation_mode,
            height,
            width,
            num_images_per_prompt,
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs,
            num_inference_steps,
        )

        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        elif isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        device = self._execution_device
        self._guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance,
                num_images_per_prompt,
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                max_sequence_length,
                device,
            )
        )
        effective_batch_size = batch_size * num_images_per_prompt

        source_latents = None
        semantic_features = None
        if generation_mode == "vq":
            vq_token_ids = self.generate_vq_tokens(prompt, height, width)
            vq_token_ids = vq_token_ids.repeat_interleave(num_images_per_prompt, dim=0)
            semantic_features = self.sigvq(token_ids=vq_token_ids.to(self._execution_device)).semantic_features
        elif generation_mode == "editing":
            source_latents, semantic_features = self._encode_source_image(
                image,
                height,
                width,
                batch_size,
                num_images_per_prompt,
            )

        latent_shape = (
            effective_batch_size,
            self.transformer.config.in_channels,
            height // self.latent_scale_factor,
            width // self.latent_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.float32)
            latents = latents.to(self.transformer.dtype).float()
        else:
            if latents.shape != latent_shape:
                raise ValueError(f"Expected `latents` to have shape {latent_shape}, got {tuple(latents.shape)}.")
            latents = latents.to(device=device, dtype=torch.float32)

        if self.scheduler.config.get("use_uniform_sigmas", False):
            # diffusers 0.39.0 does not natively support this scheduler option. Supplying the pre-shift grid
            # explicitly preserves the behavior of the patched scheduler used by LLaDA-Image-SGLang.
            sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)[:-1].tolist()
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        else:
            schedule_steps = num_inference_steps + 1
            schedule = torch.linspace(0.001, 1.0, schedule_steps, dtype=torch.float64)[:-1]
            schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
            sigmas = (1 - schedule).tolist()
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        cond_cap_feats = [
            embeds[mask].to(device=device, dtype=self.transformer.dtype)
            for embeds, mask in zip(prompt_embeds, prompt_attention_mask.bool())
        ]
        if do_classifier_free_guidance:
            uncond_cap_feats = [
                embeds[mask].to(device=device, dtype=self.transformer.dtype)
                for embeds, mask in zip(negative_prompt_embeds, negative_prompt_attention_mask.bool())
            ]
            cap_feats = cond_cap_feats + uncond_cap_feats
        else:
            cap_feats = cond_cap_feats

        glm_cap_feats = None
        source_latent_list = None
        if semantic_features is not None:
            cond_glm_cap_feats = [
                features.to(device=device, dtype=self.transformer.dtype) for features in semantic_features
            ]
            if source_latents is not None:
                source_latent_list = [
                    latent.unsqueeze(1).to(device=device, dtype=self.transformer.dtype) for latent in source_latents
                ]
            if do_classifier_free_guidance:
                empty_glm = semantic_features.new_zeros((0, semantic_features.shape[-1])).to(
                    device=device, dtype=self.transformer.dtype
                )
                glm_cap_feats = cond_glm_cap_feats + [empty_glm] * effective_batch_size
                if source_latent_list is not None:
                    source_latent_list = source_latent_list + source_latent_list
            else:
                glm_cap_feats = cond_glm_cap_feats

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                latent_list = [latent.unsqueeze(1).to(self.transformer.dtype) for latent in latent_model_input]
                model_timestep = (timestep / self.scheduler.config.num_train_timesteps).expand(
                    latent_model_input.shape[0]
                )

                model_output = self.transformer(
                    x=latent_list,
                    t=model_timestep.to(self.transformer.dtype),
                    cap_feats=cap_feats,
                    glm_cap_feats=glm_cap_feats,
                    source_latents=source_latent_list,
                ).sample
                model_output = -torch.stack(model_output, dim=0).squeeze(2).float()

                if do_classifier_free_guidance:
                    conditional_output, unconditional_output = model_output.chunk(2)
                    model_output = unconditional_output + self.guidance_scale * (
                        conditional_output - unconditional_output
                    )

                latents = self.scheduler.step(model_output, timestep, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if output_type == "latent":
            images = latents
        else:
            latents = latents.to(device=device, dtype=self.vae.dtype)
            latent_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
            latent_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                latents
            )
            latents = latents * latent_std + latent_mean
            latents = self._unpatchify_latents(latents)
            images = self.vae.decode(latents, return_dict=False)[0]
            images = self.image_processor.postprocess(images, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return LLaDAImagePipelineOutput(images=images)
