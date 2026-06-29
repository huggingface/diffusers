# Copyright 2026 SeFi-Image Authors and The HuggingFace Team. All rights reserved.
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

from typing import Callable

import torch
from transformers import Qwen2Tokenizer, Qwen3VLForConditionalGeneration

from ...models import AutoencoderKL, AutoencoderKLFlux2, SeFiTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..flux2.image_processor import Flux2ImageProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SeFiPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import SeFiPipeline

        >>> pipe = SeFiPipeline.from_pretrained("./sefi-1b-base-diffusers", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = pipe("A red apple on a wooden table.").images[0]
        >>> image.save("sefi.png")
        ```
"""


SUPPORTED_TURBO_STEPS = {4, 8, 10}


def _apply_timestep_shift_unit_interval(u_unit: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError(f"`timestep_shift_alpha` must be > 0, got {alpha}.")
    if alpha == 1.0:
        return u_unit
    denominator = 1.0 + (alpha - 1.0) * u_unit
    return (alpha * u_unit) / denominator


def _combine_guided_velocity(base_pred: torch.Tensor, cond_pred: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return base_pred + float(guidance_scale) * (cond_pred - base_pred)


class SeFiPipeline(DiffusionPipeline):
    r"""
    SeFi-Image text-to-image generation pipeline.

    Args:
        transformer ([`SeFiTransformer2DModel`]):
            Transformer that predicts semantic and texture latent velocities.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler whose training timesteps and sigmas are used for SeFi's dual-time update.
        vae ([`AutoencoderKL`] or [`AutoencoderKLFlux2`]):
            Texture VAE used to decode the final texture latent stream.
        text_encoder ([`~transformers.Qwen3VLForConditionalGeneration`]):
            Qwen3-VL text encoder. SeFi uses concatenated hidden states from selected text layers.
        tokenizer ([`~transformers.Qwen2Tokenizer`]):
            Tokenizer paired with the Qwen3-VL text encoder.
        semantic_channels (`int`, defaults to `16`):
            Number of semantic latent channels.
        texture_vae_name (`str`, defaults to `"flux2"`):
            Texture VAE normalization type. Supported values are `"sd1.5"`, `"flux1"`, and `"flux2"`.
        is_turbo (`bool`, defaults to `False`):
            Whether the checkpoint is a distilled Turbo model.
        default_guidance_scale (`float`, defaults to `4.0`):
            Default guidance scale used when `guidance_scale` is not provided.
        default_num_inference_steps (`int`, defaults to `50`):
            Default number of inference steps used when `num_inference_steps` is not provided.
        delta_t (`float`, defaults to `0.1`):
            Semantic stream lead over the texture stream.
        timestep_shift_alpha (`float`, defaults to `0.3`):
            Unit-interval timestep shift applied before the SeFi dual-time schedule.
        text_encoder_hidden_layers (`tuple[int, ...]`, defaults to `(9, 18, 27)`):
            Text encoder hidden-state indices concatenated as prompt embeddings.
        max_sequence_length (`int`, defaults to `1024`):
            Maximum prompt token length.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        transformer: SeFiTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL | AutoencoderKLFlux2,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        semantic_channels: int = 16,
        texture_vae_name: str = "flux2",
        is_turbo: bool = False,
        default_guidance_scale: float = 4.0,
        default_num_inference_steps: int = 50,
        delta_t: float = 0.1,
        timestep_shift_alpha: float = 0.3,
        text_encoder_hidden_layers: list[int] | tuple[int, ...] = (9, 18, 27),
        max_sequence_length: int = 1024,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        if isinstance(text_encoder_hidden_layers, str):
            text_encoder_hidden_layers = tuple(int(layer) for layer in text_encoder_hidden_layers.split(","))
        semantic_channels = 16 if semantic_channels is None else semantic_channels
        if texture_vae_name is None:
            texture_vae_name = "flux2" if vae is not None and hasattr(vae, "bn") else "sd1.5"
        default_guidance_scale = 4.0 if default_guidance_scale is None else default_guidance_scale
        default_num_inference_steps = 50 if default_num_inference_steps is None else default_num_inference_steps
        text_encoder_hidden_layers = (9, 18, 27) if text_encoder_hidden_layers is None else text_encoder_hidden_layers
        max_sequence_length = 1024 if max_sequence_length is None else max_sequence_length
        self.register_to_config(
            semantic_channels=semantic_channels,
            texture_vae_name=texture_vae_name,
            is_turbo=is_turbo,
            default_guidance_scale=default_guidance_scale,
            default_num_inference_steps=default_num_inference_steps,
            delta_t=delta_t,
            timestep_shift_alpha=timestep_shift_alpha,
            text_encoder_hidden_layers=tuple(text_encoder_hidden_layers),
            max_sequence_length=max_sequence_length,
        )

        self.semantic_channels = int(semantic_channels)
        self.texture_vae_name = str(texture_vae_name).lower()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128
        self._guidance_scale = None
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale is not None and self.guidance_scale > 1.0

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: torch.Tensor | None = None):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    @staticmethod
    def _unpack_latents_with_ids(
        x: torch.Tensor, x_ids: torch.Tensor, height: int | None = None, width: int | None = None
    ):
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height is not None and height <= 0:
            raise ValueError(f"`height` must be > 0, got {height}.")
        if width is not None and width <= 0:
            raise ValueError(f"`width` must be > 0, got {width}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if negative_prompt_embeds is not None and prompt_embeds is None:
            raise ValueError("`negative_prompt_embeds` requires `prompt_embeds`.")
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def _build_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _align_text_encoder_rotary_dtype(self, device: torch.device):
        text_encoder = self.text_encoder
        if text_encoder is None:
            return

        try:
            text_encoder_dtype = next(text_encoder.parameters()).dtype
        except StopIteration:
            return

        text_model = text_encoder.model if hasattr(text_encoder, "model") else text_encoder
        language_model = getattr(text_model, "language_model", None)
        rotary_emb = getattr(language_model, "rotary_emb", None)
        if rotary_emb is not None:
            # Qwen3-VL stores RoPE inverse frequencies as non-persistent buffers. `from_pretrained(torch_dtype=...)`
            # can leave them in fp32 even when text weights are bf16, while the reference SeFi wrapper casts the whole
            # text encoder module. Keep these buffers aligned before text encoding.
            rotary_emb.to(device=device, dtype=text_encoder_dtype)

    def _get_qwen3vl_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
        hidden_layers: tuple[int, ...],
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        chat_texts = [self._build_chat_text(single_prompt) for single_prompt in prompt]
        tokenized = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        self._align_text_encoder_rotary_dtype(device)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        max_idx = len(hidden_states) - 1
        for layer_idx in hidden_layers:
            if layer_idx > max_idx:
                raise ValueError(
                    f"Requested hidden layer {layer_idx}, but text encoder only provides up to {max_idx}."
                )

        stacked = torch.stack([hidden_states[idx] for idx in hidden_layers], dim=1)
        stacked = stacked.to(dtype=dtype, device=device)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int | None = None,
        text_encoder_hidden_layers: tuple[int, ...] | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or (self.transformer.dtype if self.transformer is not None else self.text_encoder.dtype)
        max_sequence_length = max_sequence_length or self.config.max_sequence_length
        text_encoder_hidden_layers = text_encoder_hidden_layers or tuple(self.config.text_encoder_hidden_layers)

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3vl_prompt_embeds(
                prompt=prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
                hidden_layers=tuple(text_encoder_hidden_layers),
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        text_ids = self._prepare_text_ids(prompt_embeds).to(device)
        return prompt_embeds, text_ids

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, self.transformer.config.in_channels, height // 2, width // 2)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if tuple(latents.shape) != tuple(shape):
                raise ValueError(f"Unexpected `latents` shape {tuple(latents.shape)}, expected {tuple(shape)}.")

        latent_ids = self._prepare_latent_ids(latents).to(device)
        return latents, latent_ids

    def _timesteps_and_sigmas(self, u_continuous: torch.Tensor, n_dim: int, dtype: torch.dtype):
        num_steps = int(self.scheduler.config.num_train_timesteps)
        indices = (u_continuous * (num_steps - 1)).long().clamp(0, num_steps - 1)
        timesteps = self.scheduler.timesteps[indices.cpu()].to(self._execution_device)
        sigmas = self.scheduler.sigmas[indices.cpu()].to(device=self._execution_device, dtype=dtype)
        while sigmas.ndim < n_dim:
            sigmas = sigmas.unsqueeze(-1)
        return timesteps, sigmas

    def decode_texture_latents(self, texture_latents: torch.Tensor, output_type: str = "pil"):
        if self.texture_vae_name == "flux2":
            if not hasattr(self.vae, "bn"):
                raise ValueError("`texture_vae_name='flux2'` requires a VAE with batch-norm statistics.")
            eps = float(getattr(self.vae.config, "batch_norm_eps", 1e-6))
            bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(texture_latents.device, texture_latents.dtype)
            bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1).to(texture_latents.device, texture_latents.dtype) + eps
            )
            texture_latents = texture_latents * bn_std + bn_mean
            raw_latents = self._unpatchify_latents(texture_latents)
        else:
            scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
            shift_factor = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
            raw_latents = self._unpatchify_latents(texture_latents)
            raw_latents = raw_latents / scaling_factor + shift_factor

        image = self.vae.decode(raw_latents.to(dtype=self.vae.dtype), return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int | None = None,
        text_encoder_hidden_layers: tuple[int, ...] | None = None,
    ) -> SeFiPipelineOutput | tuple:
        r"""
        Generates images from text prompts with SeFi-Image.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt or prompts to guide image generation.
            height (`int`, *optional*):
                Height in pixels of the generated image.
            width (`int`, *optional*):
                Width in pixels of the generated image.
            num_inference_steps (`int`, *optional*):
                Number of denoising steps. Base/RL checkpoints default to 50 and Turbo checkpoints default to 4.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance scale. Turbo checkpoints require `guidance_scale=1.0`.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated semantic and texture latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative prompt embeddings.
            output_type (`str`, defaults to `"pil"`):
                Output type of the generated image. Choose between `"pil"`, `"np"`, and `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`SeFiPipelineOutput`] instead of a tuple.
            attention_kwargs (`dict`, *optional*):
                Keyword arguments passed to attention processors.
            callback_on_step_end (`Callable`, *optional*):
                Function called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensor inputs passed to `callback_on_step_end`.
            max_sequence_length (`int`, *optional*):
                Maximum prompt sequence length.
            text_encoder_hidden_layers (`tuple[int, ...]`, *optional*):
                Text encoder hidden-state layers to concatenate.

        Examples:

        Returns:
            [`SeFiPipelineOutput`] or `tuple`: Generated images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = int(num_inference_steps or self.config.default_num_inference_steps)
        guidance_scale = float(guidance_scale if guidance_scale is not None else self.config.default_guidance_scale)

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if self.config.is_turbo:
            if num_inference_steps not in SUPPORTED_TURBO_STEPS:
                raise ValueError(f"SeFi Turbo models support {sorted(SUPPORTED_TURBO_STEPS)} steps.")
            if guidance_scale != 1.0:
                raise ValueError("SeFi Turbo models should run with `guidance_scale=1.0`.")

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            dtype=dtype,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_hidden_layers=text_encoder_hidden_layers,
        )

        negative_text_ids = None
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt = "" if batch_size == 1 else [""] * batch_size
            else:
                negative_prompt = None
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                dtype=dtype,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_hidden_layers=text_encoder_hidden_layers,
            )

        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        u_base_unit = torch.linspace(
            0.0,
            1.0,
            steps=num_inference_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        u_shifted_unit = _apply_timestep_shift_unit_interval(u_base_unit, self.config.timestep_shift_alpha)
        _, base_sigmas_schedule = self._timesteps_and_sigmas(u_shifted_unit, n_dim=1, dtype=torch.float32)
        u_sem_raw_schedule = u_shifted_unit * (1.0 + float(self.config.delta_t))

        self._num_timesteps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                if self.interrupt:
                    continue

                u_sem_raw_cur = torch.full((latents.shape[0],), float(u_sem_raw_schedule[i].item()), device=device)
                u_sem_raw_next = torch.full(
                    (latents.shape[0],), float(u_sem_raw_schedule[i + 1].item()), device=device
                )
                u_tex_cur = torch.clamp(u_sem_raw_cur - float(self.config.delta_t), min=0.0, max=1.0)
                u_sem_cur = torch.clamp(u_sem_raw_cur, max=1.0)
                u_tex_next = torch.clamp(u_sem_raw_next - float(self.config.delta_t), min=0.0, max=1.0)
                u_sem_next = torch.clamp(u_sem_raw_next, max=1.0)

                timesteps_sem_cur, sigmas_sem_cur = self._timesteps_and_sigmas(u_sem_cur, latents.ndim, latents.dtype)
                timesteps_tex_cur, sigmas_tex_cur = self._timesteps_and_sigmas(u_tex_cur, latents.ndim, latents.dtype)
                _, sigmas_sem_next = self._timesteps_and_sigmas(u_sem_next, latents.ndim, latents.dtype)
                _, sigmas_tex_next = self._timesteps_and_sigmas(u_tex_next, latents.ndim, latents.dtype)

                self._current_timestep = base_sigmas_schedule[i]
                packed_latents = self._pack_latents(latents)
                pred_cond = self.transformer(
                    hidden_states=packed_latents,
                    timestep_sem=timesteps_sem_cur / 1000,
                    timestep_tex=timesteps_tex_cur / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                pred_cond = pred_cond[:, : packed_latents.size(1)]
                pred_cond = self._unpack_latents_with_ids(pred_cond, latent_ids)

                if self.do_classifier_free_guidance:
                    pred_uncond = self.transformer(
                        hidden_states=packed_latents,
                        timestep_sem=timesteps_sem_cur / 1000,
                        timestep_tex=timesteps_tex_cur / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    pred_uncond = pred_uncond[:, : packed_latents.size(1)]
                    pred_uncond = self._unpack_latents_with_ids(pred_uncond, latent_ids)
                    velocity = _combine_guided_velocity(pred_uncond, pred_cond, guidance_scale)
                else:
                    velocity = pred_cond

                vel_sem = velocity[:, : self.semantic_channels]
                vel_tex = velocity[:, self.semantic_channels :]
                lat_sem = latents[:, : self.semantic_channels]
                lat_tex = latents[:, self.semantic_channels :]

                lat_sem = lat_sem + (sigmas_sem_next - sigmas_sem_cur) * vel_sem
                lat_tex = lat_tex + (sigmas_tex_next - sigmas_tex_cur) * vel_tex
                latents = torch.cat([lat_sem, lat_tex], dim=1)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, self._current_timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if XLA_AVAILABLE:
                    xm.mark_step()

                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            texture_latents = latents[:, self.semantic_channels :]
            image = self.decode_texture_latents(texture_latents, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SeFiPipelineOutput(images=image)
