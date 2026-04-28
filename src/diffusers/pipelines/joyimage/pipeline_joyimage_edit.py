import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2Tokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKLWan, JoyImageEditTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import BaseOutput, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import JoyImageEditImageProcessor
from .pipeline_output import JoyImageEditPipelineOutput


EXAMPLE_DOC_STRING = """"""


def _get_text_encoder_ckpt(
    text_encoder: Qwen3VLForConditionalGeneration,
    fallback: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> str:
    """
    Retrieve the checkpoint identifier from the text encoder.

    Args:
        text_encoder: The text encoder model instance.
        fallback: Default checkpoint name if none can be resolved.

    Returns:
        A non-empty string identifying the checkpoint.
    """
    candidates = [
        getattr(text_encoder, "name_or_path", None),
        getattr(getattr(text_encoder, "config", None), "_name_or_path", None),
    ]
    for c in candidates:
        if isinstance(c, str) and len(c) > 0:
            return c
    return fallback


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Configure the scheduler and return its timestep sequence.

    Exactly one of ``timesteps``, ``sigmas``, or ``num_inference_steps`` should be provided to control the denoising
    schedule.

    Args:
        scheduler: The diffusion scheduler.
        num_inference_steps: Number of denoising steps (used when neither
            ``timesteps`` nor ``sigmas`` is given).
        device: Target device for the timestep tensor.
        timesteps: Custom discrete timesteps.
        sigmas: Custom sigma values (alternative to ``timesteps``).
        **kwargs: Additional keyword arguments forwarded to ``set_timesteps``.

    Returns:
        Tuple of (timesteps tensor, num_inference_steps int).

    Raises:
        ValueError: If both ``timesteps`` and ``sigmas`` are provided, or if the
            scheduler does not support the requested schedule parameterisation.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        if "timesteps" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


@dataclass
class _LegacyPipelineOutput(BaseOutput):
    """Legacy output dataclass retained for backward compatibility."""

    videos: Union[torch.Tensor, np.ndarray]


class JoyImageEditPipeline(DiffusionPipeline):
    """
    Diffusion pipeline for image editing using the JoyImage architecture.

    The pipeline encodes text and image conditioning via a Qwen3-VL text encoder, denoises latents with a 3-D
    transformer, and decodes the result with a WAN VAE.

    Model offloading order: text_encoder -> transformer -> vae.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLWan,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: JoyImageEditTransformer3DModel,
        processor: Qwen3VLProcessor,
        text_token_max_length: int = 2048,
        text_encoder_ckpt: Optional[str] = None,
    ):
        """
        Initialise the pipeline and register all sub-modules.

        Args:
            scheduler: Noise scheduler for the denoising process.
            vae: Variational autoencoder used for encoding / decoding latents.
            text_encoder: Qwen3-VL multimodal language model for prompt encoding.
            tokenizer: Tokenizer paired with the text encoder.
            transformer: 3-D transformer denoising network.
            processor: Qwen3-VL processor for multi-image prompt preparation.
            text_token_max_length: Maximum number of text tokens for the encoder.
            text_encoder_ckpt: Path to text encoder checkpoint. Inferred from
                ``text_encoder`` when not provided.
        """
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            processor=processor,
        )

        self.text_token_max_length = text_token_max_length

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.vae_image_processor = JoyImageEditImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
        )

        if text_encoder_ckpt is None:
            text_encoder_ckpt = _get_text_encoder_ckpt(self.text_encoder)
        self.qwen_processor = processor if processor is not None else AutoProcessor.from_pretrained(text_encoder_ckpt)

        # Prompt templates used when encoding text with / without image tokens.
        self.prompt_template_encode = {
            "image": (
                "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
                "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            ),
            "multiple_images": (
                "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
                "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                "{}<|im_start|>assistant\n"
            ),
        }
        # Number of system-prompt tokens to drop from the beginning of hidden states.
        self.prompt_template_encode_start_idx = {
            "image": 34,
            "multiple_images": 34,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Extract valid (non-padded) hidden states for each sequence in the batch.

        Args:
            hidden_states: Shape (B, T, D).
        mask: Binary attention mask of shape (B, T).

        Returns:
            Tuple of tensors, one per batch element, each of shape (valid_T, D).
        """
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using the Qwen tokenizer (text-only path).

        Args:
            prompt: A single prompt string or a list of prompt strings.
            template_type: Key into ``prompt_template_encode`` / ``prompt_template_encode_start_idx``.
            device: Target device.
            dtype: Target floating-point dtype.

        Returns:
            Tuple of (prompt_embeds, encoder_attention_mask) where both tensors have shape (B, max_seq_len, D) and (B,
            max_seq_len) respectively, zero-padded to the same length.
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]

        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.text_token_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]

        # Drop system-prompt prefix tokens and re-pack into a padded batch.
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]

        max_seq_len = min(
            self.text_token_max_length,
            max(u.size(0) for u in split_hidden_states),
            max(u.size(0) for u in attn_mask_list),
        )
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt_multiple_images(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        images: Optional[torch.Tensor] = None,
        template_type: Optional[str] = "multiple_images",
        max_sequence_length: Optional[int] = None,
        drop_vit_feature: Optional[float] = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts that contain inline image tokens via the Qwen processor.

        ``<image>\\n`` placeholders in each prompt string are replaced by the Qwen vision special tokens before being
        fed to the multimodal encoder.

        Args:
            prompt: Prompt string(s), optionally containing ``<image>\\n`` tokens.
            device: Target device.
            images: Pixel tensors corresponding to the inline image tokens.
            template_type: Must be ``"multiple_images"``.
            max_sequence_length: If set, truncate the output to this length
                (keeping the last ``max_sequence_length`` tokens).
            drop_vit_feature: When True, drop all tokens up to and including the
                last vision-end token so that only the text portion is returned.

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask).
        """
        assert template_type == "multiple_images"
        device = device or self._execution_device
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # If no image tokens are present, discard the image tensors.
        if not any("<image>\n" in p for p in prompt):
            images = None

        prompt = [p.replace("<image>\n", "<|vision_start|><|image_pad|><|vision_end|>") for p in prompt]
        prompt = [template.format(p) for p in prompt]

        if (
            images is not None
            and isinstance(images, list)
            and len(images) < len(prompt)
            and len(prompt) % len(images) == 0
        ):
            images = images * (len(prompt) // len(images))

        inputs = self.qwen_processor(
            text=prompt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        encoder_hidden_states = self.text_encoder(**inputs, output_hidden_states=True)
        last_hidden_states = encoder_hidden_states.hidden_states[-1]

        if drop_vit_feature:
            # Find the last vision-end token and drop everything before it.
            input_ids = inputs["input_ids"]
            vlm_image_end_idx = torch.where(input_ids[0] == 151653)[0][-1]
            drop_idx = vlm_image_end_idx + 1

        prompt_embeds = last_hidden_states[:, drop_idx:]
        prompt_embeds_mask = inputs["attention_mask"][:, drop_idx:]

        if max_sequence_length is not None and prompt_embeds.shape[1] > max_sequence_length:
            prompt_embeds = prompt_embeds[:, -max_sequence_length:, :]
            prompt_embeds_mask = prompt_embeds_mask[:, -max_sequence_length:]

        return prompt_embeds, prompt_embeds_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        images: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        template_type: str = "image",
        drop_vit_feature: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a text prompt (and optional inline images) into embeddings.

        When ``images`` is provided the multi-image encoding path is used; otherwise the text-only Qwen tokenizer path
        is used. Pre-computed ``prompt_embeds`` bypass encoding entirely.

        Args:
            prompt: Prompt string or list of prompt strings.
            images: Optional image tensors for multi-image conditioning.
            device: Target device.
            num_images_per_prompt: Number of outputs to generate per prompt.
            prompt_embeds: Pre-computed prompt embeddings.
            prompt_embeds_mask: Attention mask for pre-computed embeddings.
            max_sequence_length: Maximum output sequence length.
            template_type: Prompt template key (``"image"`` or ``"multiple_images"``).
            drop_vit_feature: Drop vision tokens in the multi-image path.

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask).
        """
        if images is not None:
            return self.encode_prompt_multiple_images(
                prompt=prompt,
                images=images,
                device=device,
                max_sequence_length=max_sequence_length,
                drop_vit_feature=drop_vit_feature,
            )

        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, template_type, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """
        Validate pipeline inputs before the forward pass.

        Raises:
            ValueError: On any invalid combination of arguments.
        """
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError("`callback_on_step_end_tensor_inputs` has invalid keys.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError("`prompt` has to be of type `str` or `list`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("If `prompt_embeds` are provided, `prompt_embeds_mask` is required.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` is required.")

    def normalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Normalise latents using per-channel statistics from the VAE config.

        Uses (latent - mean) / std when the VAE exposes ``latents_mean`` and ``latents_std``; otherwise falls back to
        scaling by ``scaling_factor``.

        Args:
            latent: Raw latent tensor from ``vae.encode``.

        Returns:
            Normalised latent tensor.
        """
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        else:
            latent = latent * self.vae.config.scaling_factor
        return latent

    def denormalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Invert :meth:`normalize_latents` to recover the original latent scale.

        Args:
            latent: Normalised latent tensor.

        Returns:
            Latent tensor in the scale expected by ``vae.decode``.
        """
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latent = latent * latents_std + latents_mean
        else:
            latent = latent / self.vae.config.scaling_factor
        return latent

    def prepare_latents(
        self,
        batch_size: int,
        num_items: int,
        num_channels_latents: int,
        height: int,
        width: int,
        video_length: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
        reference_images: Optional[List[Image.Image]] = None,
        enable_denormalization: bool = True,
    ) -> torch.Tensor:
        """
        Prepare the initial noisy latent tensor for the denoising loop.

        When ``reference_images`` is provided the first (num_items - 1) slots are filled with VAE-encoded reference
        image latents; the last slot is random noise. When ``latents`` is provided it is moved to ``device`` without
        modification. Otherwise pure random noise is returned.

        Args:
            batch_size: Number of samples in the batch.
            num_items: Number of image slots (reference + target).
            num_channels_latents: Latent channel dimension from the transformer config.
            height: Spatial height in pixels.
            width: Spatial width in pixels.
            video_length: Number of frames (1 for image inference).
            dtype: Floating-point dtype for the latent tensor.
            device: Target device.
            generator: RNG generator(s) for reproducible sampling.
            latents: Optional pre-allocated latent tensor.
            reference_images: Optional list of PIL images to encode as conditioning.
            enable_denormalization: Whether to normalise encoded reference latents.

        Returns:
            Latent tensor of shape (B, num_items, C, T, H', W').

        Raises:
            ValueError: If ``generator`` is a list whose length differs from ``batch_size``.
        """
        shape = (
            batch_size,
            num_items,
            num_channels_latents,
            (video_length - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generator list length must match batch size.")

        if latents is None:
            if reference_images is not None:
                if batch_size > len(reference_images) and batch_size % len(reference_images) == 0:
                    reference_images = reference_images * (batch_size // len(reference_images))
                elif batch_size > len(reference_images):
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {len(reference_images)} to {batch_size} text prompts."
                    )
                # Encode reference images and concatenate with a noise slot.
                ref_img = [torch.from_numpy(np.array(x.convert("RGB"))) for x in reference_images]
                ref_img = torch.stack(ref_img).to(device=device, dtype=dtype)
                ref_img = ref_img / 127.5 - 1.0
                ref_img = ref_img.permute(0, 3, 1, 2).unsqueeze(2)
                ref_vae = self.vae.encode(ref_img).latent_dist.sample()
                if enable_denormalization:
                    ref_vae = self.normalize_latents(ref_vae)
                ref_vae = ref_vae.view(shape[0], num_items - 1, *ref_vae.shape[1:])
                noise = randn_tensor((shape[0], 1, *shape[2:]), generator=generator, device=device, dtype=dtype)
                latents = torch.cat([ref_vae, noise], dim=1)
            else:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents

    # ------------------------------------------------------------------
    # Pipeline properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self) -> float:
        """Classifier-free guidance scale used in the current forward pass."""
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        """True when guidance_scale > 1, enabling classifier-free guidance."""
        return self._guidance_scale > 1

    @property
    def num_timesteps(self) -> int:
        """Total number of denoising timesteps in the current forward pass."""
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        """When True, the denoising loop is interrupted at the next step."""
        return self._interrupt

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput | None = None,
        prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 4096,
        drop_vit_feature: bool = False,
        enable_denormalization: bool = True,
        **kwargs,
    ):
        r"""
        Generate an edited image conditioned on a reference image and a text prompt.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide generation.
            height (`int`):
                Height of the generated output in pixels.
            width (`int`):
                Width of the generated output in pixels.
            image (`PipelineImageInput`, *optional*):
                Reference image used for conditioning. When provided the pipeline operates in image-editing mode with
                ``num_items=2``.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of denoising steps. More steps generally improve quality at the cost of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps for the denoising process. When provided, ``num_inference_steps`` is inferred from the
                list length.
            sigmas (`List[float]`, *optional*):
                Custom sigmas for the denoising process. Mutually exclusive with ``timesteps``.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) used to suppress undesired content.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of generated samples per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                RNG generator(s) for deterministic sampling.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents. Sampled from a Gaussian distribution when not provided.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed prompt embeddings. When provided ``prompt`` can be omitted.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for ``prompt_embeds``.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed negative prompt embeddings.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for ``negative_prompt_embeds``.
            output_type (`str`, *optional*, defaults to ``"pil"``):
                Output format. Pass ``"latent"`` to return raw latents.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a :class:`JoyImageEditPipelineOutput` or a plain tensor.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                Callback invoked at the end of each denoising step with signature ``(self, step: int, timestep: int,
                callback_kwargs: Dict)``.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*, defaults to ``["latents"]``):
                Tensor keys included in ``callback_kwargs`` for ``callback_on_step_end``.
            enable_tiling (`bool`, *optional*, defaults to `False`):
                Enable tiled VAE decoding to reduce peak memory usage.
            max_sequence_length (`int`, *optional*, defaults to 4096):
                Maximum sequence length for prompt encoding.
            drop_vit_feature (`bool`, *optional*, defaults to `False`):
                Drop vision tokens in the multi-image encoding path.
            enable_denormalization (`bool`, *optional*, defaults to `True`):
                Denormalise latents before VAE decoding.
            **kwargs:
                Additional keyword arguments for forward compatibility.

        Examples:

        Returns:
            [`~pipelines.joyimage.JoyImageEditPipelineOutput`] or `torch.Tensor`:
                If ``return_dict`` is ``True``, returns a pipeline output object containing the generated image(s).
                Otherwise returns the image tensor directly.
        """
        # Resize the input image to the nearest bucket resolution.
        # Or resize the specified height and width to the nearest bucket resolution.
        height, width = self.vae_image_processor.get_default_height_width(image, height, width)
        processed_image = self.vae_image_processor.resize_center_crop(image, (height, width))

        self.check_inputs(
            prompt,
            height,
            width,
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

        device = self._execution_device

        # num_items: 1 for unconditional generation, 2 for reference-image editing.
        num_items = 1 if image is None else 2

        # Encode the conditioning prompt (and reference image when present).
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            images=processed_image,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            template_type="image",
            drop_vit_feature=drop_vit_feature,
        )

        if self.do_classifier_free_guidance:
            # Build default negative prompts when none are provided.
            if negative_prompt is None and negative_prompt_embeds is None:
                if num_items <= 1:
                    negative_prompt = ["<|im_start|>user\n<|im_end|>\n"] * batch_size
                else:
                    negative_prompt = ["<|im_start|>user\n<image>\n<|im_end|>\n"] * batch_size

            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                images=processed_image,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                template_type="image",
            )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_items,
            num_channels_latents,
            height,
            width,
            1,  # video_length = 1 for image inference
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            reference_images=processed_image if isinstance(processed_image, list) else [processed_image],
            enable_denormalization=enable_denormalization,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Cache reference latents to restore them at each denoising step.
        if num_items > 1:
            ref_latents = latents[:, : (num_items - 1)].clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Restore reference latents so they are never overwritten by the scheduler.
                if num_items > 1:
                    latents[:, : (num_items - 1)] = ref_latents.clone()

                latent_model_input = latents
                t_expand = t.repeat(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_expand,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_expand,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        return_dict=False,
                    )[0]

                    comb_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)
                    # Rescale to match the conditional prediction norm (guidance rescaling).
                    cond_norm = torch.norm(noise_pred, dim=2, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=2, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm.clamp_min(1e-6))

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        if output_type != "latent":
            latents = latents.flatten(0, 1)
            if enable_denormalization:
                latents = self.denormalize_latents(latents)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = image.unflatten(0, (batch_size, -1))
        else:
            image = latents

        # Extract the target slot (last item) from each batch element.
        # (B, num_items, C, T, H, W) -> permute -> (B, num_items, T, C, H, W) -> [:, -1] -> (B, T, C, H, W)
        image = image.float().permute(0, 1, 3, 2, 4, 5)[:, -1].squeeze(1)

        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return JoyImageEditPipelineOutput(images=image)
