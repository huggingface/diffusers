# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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

import math
from typing import Any, Callable

import torch
from transformers import AutoTokenizer, PreTrainedModel
from transformers.masking_utils import create_causal_mask

from ...image_processor import VaeImageProcessor
from ...loaders import Ideogram4LoraLoaderMixin
from ...models.autoencoders import AutoencoderKLFlux2
from ...models.transformers.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_outlines_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Ideogram4PipelineOutput
from .prompt_enhancer import (
    PROMPT_UPSAMPLE_TEMPERATURE,
    Ideogram4PromptEnhancerHead,
    build_caption_logits_processor,
    build_prompt_enhancer,
    generate_captions,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Hidden states of these Qwen3-VL decoder layers are concatenated to form the per-token
# text conditioning consumed by the Ideogram4 transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import Ideogram4Pipeline

        >>> pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-v4", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A photo of a cat holding a sign that says hello world"
        >>> # The defaults are the recommended settings for best quality.
        >>> image = pipe(prompt, height=2048, width=2048, generator=torch.Generator("cuda").manual_seed(0)).images[0]
        >>> image.save("ideogram4.png")
        ```
"""


def _logit_normal_sigmas(
    num_inference_steps: int,
    mu: float,
    std: float = 1.0,
    logsnr_min: float = -15.0,
    logsnr_max: float = 18.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    r"""
    Build a length-`num_inference_steps` sigma schedule using the Ideogram4 logit-normal flow-matching schedule.

    Sigmas are returned in `[0, 1]` in decreasing order (sigma close to 1 corresponds to pure noise, sigma close to 0
    to clean data), matching diffusers conventions.

    The Ideogram4 schedule applies `sigma(s) = 1 - logit_normal_cdf_inverse(1 - s)` to `s = linspace(0, 1, N + 1)` and
    keeps the first `N` entries; a terminal zero is appended downstream by the scheduler.
    """
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    # Apply the inverse CDF of a normal then push through the logistic to obtain a logit-normal CDF inverse.
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    # Convert from model time (0 = noise, 1 = data) to diffusers sigma (1 = noise, 0 = data) and reverse.
    sigmas = (1.0 - t).flip(0)
    # Drop the trailing 0; FlowMatchEulerDiscreteScheduler.set_timesteps appends one back internally.
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


def _resolution_aware_mu(
    height: int,
    width: int,
    base_mu: float,
    base_resolution: tuple[int, int] = (512, 512),
) -> float:
    """Shift the schedule mean as a function of image resolution."""
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


def _expand_tensor_to_effective_batch(
    tensor: torch.Tensor,
    batch_size: int,
    num_per_prompt: int,
    tensor_name: str | None = None,
) -> torch.Tensor:
    """Replicate `tensor` along dim 0 from `batch_size` (or 1) to `batch_size * num_per_prompt`."""
    target_batch_size = batch_size * num_per_prompt

    if tensor.shape[0] == target_batch_size:
        return tensor

    if tensor.shape[0] == 1:
        repeat_by = target_batch_size
    elif tensor.shape[0] == batch_size:
        repeat_by = num_per_prompt
    else:
        tensor_name = f"`{tensor_name}`" if tensor_name is not None else "Tensor"
        raise ValueError(
            f"{tensor_name} batch size must be 1, `batch_size` ({batch_size}), or "
            f"`batch_size * num_*_per_prompt` ({target_batch_size}), but got {tensor.shape[0]}."
        )

    return torch.repeat_interleave(tensor, repeats=repeat_by, dim=0, output_size=tensor.shape[0] * repeat_by)


class Ideogram4Pipeline(DiffusionPipeline, Ideogram4LoraLoaderMixin):
    r"""
    Text-to-image pipeline for Ideogram4.

    Ideogram4 is a flow-matching model trained with asymmetric classifier-free guidance: a `transformer` consumes
    text-conditioned features alongside the image latents, while a separate `unconditional_transformer` denoises with
    zeroed text features. The two velocity predictions are linearly blended each step.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler. The pipeline overrides the default sigma schedule with a resolution-aware
            logit-normal schedule.
        vae ([`AutoencoderKLFlux2`]):
            Variational auto-encoder used to decode latents back into images.
        text_encoder ([`PreTrainedModel`]):
            Multimodal text encoder. The pipeline consumes hidden states from a fixed set of intermediate decoder
            layers (see `QWEN3_VL_ACTIVATION_LAYERS`).
        tokenizer ([`AutoTokenizer`]):
            Tokenizer paired with `text_encoder`.
        transformer ([`Ideogram4Transformer2DModel`]):
            Conditional flow-matching transformer.
        unconditional_transformer ([`Ideogram4Transformer2DModel`]):
            Unconditional (asymmetric-CFG) flow-matching transformer.
    """

    model_cpu_offload_seq = "prompt_enhancer_head->text_encoder->transformer->unconditional_transformer->vae"
    _optional_components = ["prompt_enhancer_head"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: Ideogram4Transformer2DModel,
        unconditional_transformer: Ideogram4Transformer2DModel,
        prompt_enhancer_head: Ideogram4PromptEnhancerHead | None = None,
    ) -> None:
        super().__init__()

        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            unconditional_transformer=unconditional_transformer,
            prompt_enhancer_head=prompt_enhancer_head,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) is not None else 8
        )
        # Ideogram4 patchifies the VAE output by a factor of 2 before feeding into the transformer.
        self.patch_size = 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)

        # Built lazily on first upsample: the head-less encoder body + `prompt_enhancer_head`, combined.
        self._prompt_enhancer = None
        # Outlines logits processor for schema-constrained captions; built lazily on first upsample.
        self._caption_logits_processor = None

    def upsample_prompt(
        self,
        prompt: str | list[str],
        height: int = 2048,
        width: int = 2048,
        temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
        max_new_tokens: int = 1024,
        generator: torch.Generator | list[torch.Generator] | None = None,
        device: torch.device | None = None,
    ) -> list[str]:
        """Rewrite each prompt into Ideogram4's native structured JSON caption.

        Requires the optional `prompt_enhancer_head` component, which is grafted onto the shared `text_encoder` body to
        make it generative. Generation is schema-constrained when `outlines` is installed, otherwise it runs
        unconstrained. Pass `generator` (the same one accepted by `__call__`) to make sampling reproducible.
        """
        if self.prompt_enhancer_head is None:
            raise ValueError(
                "Prompt upsampling requires the `prompt_enhancer_head` component, which is not loaded. Load it and "
                "pass it in, e.g.:\n"
                "    from diffusers import Ideogram4PromptEnhancerHead\n"
                "    head = Ideogram4PromptEnhancerHead.from_pretrained('diffusers/qwen3-vl-8b-instruct-lm-head')\n"
                "    pipe = Ideogram4Pipeline.from_pretrained(model_id, prompt_enhancer_head=head)"
            )
        if self._prompt_enhancer is None:
            self._prompt_enhancer = build_prompt_enhancer(self.text_encoder, self.prompt_enhancer_head)
        if self._caption_logits_processor is None and is_outlines_available():
            self._caption_logits_processor = build_caption_logits_processor(self._prompt_enhancer, self.tokenizer)
        if self._caption_logits_processor is None:
            logger.warning_once(
                "`outlines` is not installed; prompt upsampling runs unconstrained and may not return schema-valid "
                "JSON. Install with `pip install outlines` for structured captions."
            )

        return generate_captions(
            self._prompt_enhancer,
            self.tokenizer,
            self._caption_logits_processor,
            prompt,
            height,
            width,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            generator=generator,
            device=device,
        )

    @staticmethod
    def _prepare_ids(
        text_lengths: list[int],
        grid_h: int,
        grid_w: int,
        max_text_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the packed `[left-pad][text][image]` layout from the per-prompt text lengths and the image grid.

        Returns `position_ids` (3-axis MRoPE), `segment_ids` (block-diagonal attention) and `indicator` (per-token
        text/image/pad role).
        """
        batch_size = len(text_lengths)
        num_image_tokens = grid_h * grid_w
        total_seq_len = max_text_tokens + num_image_tokens

        # Image position ids (t=0, h, w); offset keeps them disjoint from text positions.
        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, num_text in enumerate(text_lengths):
            offset = max_text_tokens - num_text

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

            segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

        return position_ids.to(device), segment_ids.to(device), indicator.to(device)

    @staticmethod
    def _get_text_encoder_hidden_states(
        text_encoder,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_2d: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run the text encoder's decoder layers, returning the hidden states tapped at each activation layer."""

        language_model = text_encoder.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    def encode_prompt(
        self,
        prompt: str | list[str],
        grid_h: int,
        grid_w: int,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the conditioning for the packed text+image sequence (one entry per prompt).

        Returns a flat tuple `(prompt_embeds, position_ids, segment_ids, indicator)`. The unconditional branch carries
        no text, so the pipeline builds its (zeroed) inputs directly rather than encoding a negative prompt.
        """
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompts)
        num_image_tokens = grid_h * grid_w

        # Tokenize each chat-formatted prompt and left-pad to `max_sequence_length`. Only the text region is fed to
        # the encoder: the packed image tokens come after the text and the encoder is causal, so they never affect it.
        token_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, max_sequence_length, dtype=torch.long)
        text_lengths = []
        for b, text_prompt in enumerate(prompts):
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            n = int(toks.shape[0])
            if n > max_sequence_length:
                raise ValueError(f"prompt has {n} tokens, exceeds max_sequence_length={max_sequence_length}")
            text_lengths.append(n)
            offset = max_sequence_length - n
            token_ids[b, offset:] = toks
            attention_mask[b, offset:] = 1
            text_position_ids[b, offset:] = torch.arange(n)

        # To support enable_model_cpu_offload, we need to move the text_encoder inputs to the text encoder's actual
        # device te_device. This is necessary because the `CpuOffload` model offload hook attaches to a component's
        # `forward` method, but we call text_encoder's submodules directly below, so the hook never fires to onload the
        # model to the execution device. Other offloading techniques (group, sequential) would work without te_device
        # because they hook submodules, not just the top-level component module. Note that in the
        # enable_model_cpu_offload case te_device will actually be the offload device (e.g. CPU).
        te_device = self.text_encoder.device
        token_ids = token_ids.to(te_device)
        attention_mask = attention_mask.to(te_device)
        text_position_ids = text_position_ids.to(te_device)

        # Concatenate the tapped activation-layer hidden states into per-token text features, zeroing padding.
        selected = self._get_text_encoder_hidden_states(
            self.text_encoder, token_ids, attention_mask, text_position_ids
        )
        text_features = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(batch_size, max_sequence_length, -1)
        text_features = (text_features * attention_mask.to(text_features.dtype).unsqueeze(-1)).to(torch.float32)
        text_features = text_features.to(device)

        position_ids, segment_ids, indicator = self._prepare_ids(
            text_lengths, grid_h, grid_w, max_sequence_length, device
        )

        # Pack the text features into the full sequence; image positions carry no text features.
        image_feature_padding = torch.zeros(
            batch_size, num_image_tokens, text_features.shape[-1], dtype=text_features.dtype, device=device
        )
        prompt_embeds = torch.cat([text_features, image_feature_padding], dim=1)

        return prompt_embeds, position_ids, segment_ids, indicator

    def prepare_latents(
        self,
        batch_size: int,
        num_image_tokens: int,
        latent_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (batch_size, num_image_tokens, latent_dim)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self) -> float | None:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def attention_kwargs(self) -> dict[str, Any] | None:
        return self._attention_kwargs

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    def check_inputs(
        self,
        prompt,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        guidance_schedule,
        callback_on_step_end_tensor_inputs=None,
    ):
        if prompt is None:
            raise ValueError("`prompt` must be provided.")
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` ({height}) and `width` ({width}) must both be divisible by {self.vae_scale_factor * self.patch_size} "
                f"(vae_scale_factor * patch_size)."
            )

        # Guidance is controlled by either a constant `guidance_scale` or a per-step `guidance_schedule`; exactly
        # one must be set (the `guidance_schedule` default makes the no-arg call use the recommended schedule).
        if guidance_scale is not None and guidance_schedule is not None:
            raise ValueError("Only one of `guidance_scale` and `guidance_schedule` may be set.")
        if guidance_scale is None and guidance_schedule is None:
            raise ValueError("One of `guidance_scale` and `guidance_schedule` must be set.")
        if guidance_schedule is not None and len(guidance_schedule) != num_inference_steps:
            raise ValueError(
                f"`guidance_schedule` must have length `num_inference_steps` ({num_inference_steps}), "
                f"got {len(guidance_schedule)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: int = 48,
        guidance_scale: float | None = None,
        guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45 + (3.0,) * 3,
        mu: float = 0.0,
        std: float = 1.5,
        prompt_upsampling: bool = False,
        prompt_upsampling_temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
        max_sequence_length: int = 2048,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[["Ideogram4Pipeline", int, int, dict[str, Any]], dict[str, Any]] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> Ideogram4PipelineOutput | tuple[Any]:
        r"""
        Run text-to-image generation.

        Args:
            prompt (`str` or `list[str]`):
                Prompt(s) to guide image generation.
            height (`int`, *optional*, defaults to 2048):
                Output image height in pixels; must be a multiple of `vae_scale_factor * patch_size`.
            width (`int`, *optional*, defaults to 2048):
                Output image width in pixels; must be a multiple of `vae_scale_factor * patch_size`.
            num_inference_steps (`int`, *optional*, defaults to 48):
                Number of flow-matching steps. The default is the recommended setting for best quality.
            guidance_scale (`float`, *optional*):
                Constant classifier-free guidance scale applied at every step. The conditional and unconditional
                velocity predictions are blended as `v = guidance_scale * v_pos + (1 - guidance_scale) * v_neg`.
                Mutually exclusive with `guidance_schedule` (setting both raises). Defaults to `None`.
            guidance_schedule (`list[float]` or `torch.Tensor`, *optional*):
                Per-step guidance scale schedule; must have length `num_inference_steps`. The first entry corresponds
                to the first step (largest noise level). Mutually exclusive with `guidance_scale`; exactly one must be
                set. Defaults to the recommended schedule (7.0 for the main steps, dropping to 3.0 for the final 3
                "polish" steps). To use a constant scale instead, pass `guidance_scale` and `guidance_schedule=None`.
            mu (`float`, *optional*, defaults to 0.0):
                Base mean of the logit-normal flow-matching schedule. The schedule mean is shifted by half the log of
                the resolution ratio relative to 512x512.
            std (`float`, *optional*, defaults to 1.5):
                Standard deviation of the logit-normal flow-matching schedule.
            prompt_upsampling (`bool`, *optional*, defaults to `False`):
                If `True`, rewrite `prompt` into Ideogram4's native structured JSON caption via
                [`~Ideogram4Pipeline.upsample_prompt`] before encoding. Requires the optional `prompt_enhancer_head`
                component; install `outlines` for schema-constrained captions. `generator` is reused to make the
                upsampling reproducible.
            prompt_upsampling_temperature (`float`, *optional*, defaults to 1.0):
                Sampling temperature for prompt upsampling when `prompt_upsampling=True`.
            max_sequence_length (`int`, *optional*, defaults to 2048):
                Maximum number of text tokens per prompt.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) used to make sampling deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noise of shape `(batch_size, num_image_tokens, latent_dim)`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                One of `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`~pipelines.ideogram4.Ideogram4PipelineOutput`].
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the attention processor of each transformer. A `"scale"` entry
                scales the loaded LoRA weights (e.g. `{"scale": 0.7}`) when the PEFT backend is active.
            callback_on_step_end (`Callable`, *optional*):
                Callback invoked at the end of every denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Names of tensors to expose to the callback via `callback_kwargs`.

        Examples:

        Returns:
            [`~pipelines.ideogram4.Ideogram4PipelineOutput`] or `tuple`.
        """
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 0. Optionally rewrite the prompt(s) into Ideogram4's native structured JSON caption.
        if prompt_upsampling:
            prompt = self.upsample_prompt(
                prompt,
                height=height,
                width=width,
                temperature=prompt_upsampling_temperature,
                max_new_tokens=max_sequence_length,
                generator=generator,
                device=device,
            )

        # 1. Image grid (drives both the packed layout and the latent shape).
        grid_h, grid_w = (
            height // (self.vae_scale_factor * self.patch_size),
            width // (self.vae_scale_factor * self.patch_size),
        )
        num_image_tokens = grid_h * grid_w

        # 2. Encode prompts into the packed conditioning (one entry per prompt).
        llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
            prompt=prompt,
            grid_h=grid_h,
            grid_w=grid_w,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 3. Replicate the conditioning for num_images_per_prompt.
        llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
        position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
        segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
        indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

        # 4. Unconditional (image-only) branch, derived from the conditioning: zeroed text features and the
        # image-region slices of the layout.
        neg_llm_features = torch.zeros(
            batch_size * num_images_per_prompt,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=device,
        )
        neg_position_ids = position_ids[:, max_sequence_length:]
        neg_segment_ids = segment_ids[:, max_sequence_length:]
        neg_indicator = indicator[:, max_sequence_length:]

        # 4. Set up the resolution-aware logit-normal schedule on the scheduler.
        schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
        sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 5. Resolve the per-step guidance schedule (a constant `guidance_scale` broadcasts to every step, otherwise
        # use the provided `guidance_schedule`, validated by `check_inputs`) and the tensor of per-step weights `gw`.
        if guidance_scale is not None:
            guidance_schedule = [float(guidance_scale)] * num_inference_steps
        gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

        # 6. Prepare latents in the packed (B, num_image_tokens, latent_dim) layout.
        latent_dim = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 7. Padding for the text region of the conditional packed sequence (image latents are appended after it).
        max_text_tokens = max_sequence_length
        text_z_padding = torch.zeros(
            batch_size * num_images_per_prompt,
            max_text_tokens,
            latent_dim,
            dtype=torch.float32,
            device=device,
        )

        # The transformers run in their loaded compute dtype; cast the (otherwise float32) text features to match.
        # `latents` stay float32 for scheduler precision and are cast per-step at the transformer call below.
        llm_features = llm_features.to(self.transformer.dtype)
        neg_llm_features = neg_llm_features.to(self.unconditional_transformer.dtype)

        # 8. Denoising loop. The scheduler stores `num_train_timesteps`-scaled timesteps; convert back to model time.
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Map sigma-domain timestep to model time `t` in [0, 1] (0 = noise, 1 = clean data).
                t_model = 1.0 - (t.float() / num_train_timesteps)
                t_model = t_model.expand(batch_size * num_images_per_prompt).to(self.transformer.dtype)

                # Conditional pass operates on the full packed sequence.
                pos_z = torch.cat([text_z_padding, latents], dim=1).to(self.transformer.dtype)
                pos_out = self.transformer(
                    hidden_states=pos_z,
                    timestep=t_model,
                    encoder_hidden_states=llm_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                # Velocity (and guidance) is computed in float32 for scheduler precision; the transformers
                # return their compute dtype, so cast the predicted velocities up here.
                pos_v = pos_out[:, max_text_tokens:].to(torch.float32)

                # Unconditional pass uses image-only positions with zeroed text features.
                neg_v = self.unconditional_transformer(
                    hidden_states=latents.to(self.unconditional_transformer.dtype),
                    timestep=t_model,
                    encoder_hidden_states=neg_llm_features,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0].to(torch.float32)

                # Expose the current step's guidance weight via `self.guidance_scale` so callbacks can read it.
                self._guidance_scale = guidance_schedule[i]
                gw_i = gw[i]
                v = gw_i * pos_v + (1.0 - gw_i) * neg_v

                latents = self.scheduler.step(-v, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        # 9. Decode: unpatch the latents, denormalize with the VAE batch-norm stats, and decode through the VAE.
        if output_type == "latent":
            image = latents
        else:
            z = latents
            # VAE bn stores per-channel statistics on the packed-channel latent space (ae_channels * patch ** 2).
            bn_mean = self.vae.bn.running_mean.view(1, 1, -1).to(device=z.device, dtype=z.dtype)
            bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, 1, -1)
            bn_std = bn_std.to(device=z.device, dtype=z.dtype)
            z = z * bn_std + bn_mean

            patch = self.patch_size
            ae_channels = z.shape[-1] // (patch * patch)
            z = z.view(batch_size * num_images_per_prompt, grid_h, grid_w, patch, patch, ae_channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(batch_size * num_images_per_prompt, ae_channels, grid_h * patch, grid_w * patch)

            decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(decoded.float(), output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return Ideogram4PipelineOutput(images=image)
