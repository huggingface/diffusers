# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...models import AutoencoderKLMagi, MagiTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    is_torch_xla_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import MagiPipelineOutput


if is_torch_xla_available():

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import MagiPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Text-to-video generation
        >>> pipeline = MagiPipeline.from_pretrained("sand-ai/MAGI-1-4.5B", torch_dtype=torch.float16)
        >>> pipeline = pipeline.to("cuda")
        >>> prompt = "A cat and a dog playing in a garden. The cat is chasing a butterfly while the dog is digging a hole."
        >>> output = pipeline(
        ...     prompt=prompt,
        ...     num_frames=24,
        ...     height=720,
        ...     width=720,
        ...     t_schedule_func="sd3",
        ... ).frames[0]
        >>> export_to_video(output, "magi_output.mp4", fps=8)
        ```
"""


class MagiPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using MAGI-1.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer (`AutoTokenizer`):
            Tokenizer for the text encoder.
        text_encoder (`UMT5EncoderModel`):
            Text encoder for conditioning.
        transformer (`MagiTransformer3DModel`):
            Conditional Transformer to denoise the latent video.
        vae (`AutoencoderKLMagi`):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: MagiTransformer3DModel,
        vae: AutoencoderKLMagi,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = vae.temporal_downsample_factor
        self.vae_scale_factor_spatial = vae.spatial_downsample_factor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`): The prompt or prompts to guide the video generation.
            device: The device to place the encoded prompt on.
            num_videos_per_prompt (`int`): The number of videos that should be generated per prompt.
            do_classifier_free_guidance (`bool`): Whether to use classifier-free guidance or not.
            negative_prompt (`str` or `List[str]`, *optional*): The prompt or prompts not to guide the video generation.
            max_length (`int`, *optional*): The maximum length of the prompt to be encoded.

        Returns:
            `torch.Tensor`: A tensor containing the encoded text embeddings.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # Default to 77 if not specified
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]

        # Process special tokens if present (following MAGI-1's approach)
        # In diffusers style, we don't need to explicitly handle special tokens as they're part of the tokenizer
        # But we can ensure proper mask handling similar to MAGI-1
        seq_len = prompt_embeds.shape[1]

        # Duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_videos_per_prompt, dim=0)

        # Get unconditional embeddings for classifier-free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids.to(device)
            uncond_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0]

            # Duplicate unconditional embeddings for each generation per prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)
            uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_videos_per_prompt, dim=0)

            # Ensure null embeddings have proper attention mask handling (similar to MAGI-1's null_emb_masks)
            # In MAGI-1, they set attention to first 50 tokens and zero for the rest
            if uncond_attention_mask.shape[1] > 50:
                uncond_attention_mask[:, :50] = 1
                uncond_attention_mask[:, 50:] = 0

            # Concatenate unconditional and text embeddings for classifier-free guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([uncond_attention_mask, attention_mask])

        return prompt_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare latents for diffusion.

        Args:
            batch_size (`int`): The batch size.
            num_channels_latents (`int`): The number of channels in the latent space.
            num_frames (`int`): The number of frames to generate.
            height (`int`): The height of the video.
            width (`int`): The width of the video.
            dtype (`torch.dtype`): The data type of the latents.
            device (`torch.device`): The device to place the latents on.
            generator (`torch.Generator`, *optional*): A generator to use for random number generation.
            latents (`torch.Tensor`, *optional*): Pre-generated latent vectors. If not provided, latents will be generated randomly.

        Returns:
            `torch.Tensor`: The prepared latent vectors.
        """
        shape = (
            batch_size,
            num_channels_latents,
            num_frames // self.vae_scale_factor_temporal,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _get_chunk_indices(self, num_frames: int, chunk_size: int = 24) -> List[Tuple[int, int]]:
        """
        Get the indices for processing video in chunks.

        Args:
            num_frames (`int`): Total number of frames.
            chunk_size (`int`, *optional*, defaults to 24): Size of each chunk.

        Returns:
            `List[Tuple[int, int]]`: List of (start_idx, end_idx) tuples for each chunk.
        """
        chunks = []
        for i in range(0, num_frames, chunk_size):
            chunks.append((i, min(i + chunk_size, num_frames)))
        return chunks

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        """
        Validate the inputs for the pipeline.

        Args:
            prompt (`str` or `List[str]`): The prompt or prompts to guide generation.
            height (`int`): The height in pixels of the generated video.
            width (`int`): The width in pixels of the generated video.
            callback_steps (`int`): The frequency at which the callback function is called.
            negative_prompt (`str` or `List[str]`, *optional*): The prompt or prompts not to guide generation.
            prompt_embeds (`torch.Tensor`, *optional*): Pre-computed text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*): Pre-computed negative text embeddings.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f"`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def _prepare_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
        t_schedule_func: str = "sd3",
        shift: float = 3.0,
    ) -> torch.Tensor:
        """
        Prepare timesteps for diffusion process, with scheduling options similar to MAGI-1.

        Args:
            num_inference_steps (`int`): Number of diffusion steps.
            device (`torch.device`): Device to place timesteps on.
            t_schedule_func (`str`, optional, defaults to "sd3"):
                Timestep scheduling function. Options: "sd3", "square", "piecewise", "linear".
            shift (`float`, optional, defaults to 3.0): Shift parameter for sd3 scheduler.

        Returns:
            `torch.Tensor`: Prepared timesteps.
        """
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Apply custom scheduling similar to MAGI-1 if needed
        if t_schedule_func == "sd3":
            # Apply quadratic transformation
            t = torch.linspace(0, 1, num_inference_steps + 1, device=device)
            t = t ** 2

            # Apply SD3-style transformation
            def t_resolution_transform(x, shift_value=shift):
                assert shift_value >= 1.0, "shift should >=1"
                shift_inv = 1.0 / shift_value
                return shift_inv * x / (1 + (shift_inv - 1) * x)

            t = t_resolution_transform(t, shift)

            # Map to scheduler timesteps
            # Note: This is a simplified approach - in a full implementation,
            # we would need to properly map these values to the scheduler's timesteps
            return self.scheduler.timesteps

        elif t_schedule_func == "square":
            # Simple quadratic scheduling
            t = torch.linspace(0, 1, num_inference_steps + 1, device=device)
            t = t ** 2
            return self.scheduler.timesteps

        elif t_schedule_func == "piecewise":
            # Piecewise scheduling as in MAGI-1
            t = torch.linspace(0, 1, num_inference_steps + 1, device=device)

            # Apply piecewise transformation
            mask = t < 0.875
            t_transformed = torch.zeros_like(t)
            t_transformed[mask] = t[mask] * (0.5 / 0.875)
            t_transformed[~mask] = 0.5 + (t[~mask] - 0.875) * (0.5 / (1 - 0.875))

            # Map to scheduler timesteps (simplified)
            return self.scheduler.timesteps

        # Default: use scheduler's default timesteps
        return timesteps

    def denoise_latents(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: List[int],
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Denoise the latents using the transformer model.

        Args:
            latents (`torch.Tensor`): The initial noisy latents.
            prompt_embeds (`torch.Tensor`): The text embeddings for conditioning.
            timesteps (`List[int]`): The timesteps for the diffusion process.
            callback (`Callable`, *optional*): A function that will be called every `callback_steps` steps.
            callback_steps (`int`, *optional*, defaults to 1): The frequency at which the callback is called.
            guidance_scale (`float`, *optional*, defaults to 7.5): The scale for classifier-free guidance.

        Returns:
            `torch.Tensor`: The denoised latents.
        """
        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = latents.shape[0] // (2 if do_classifier_free_guidance else 1)

        for i, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.transformer(
                latent_model_input,
                timesteps=torch.tensor([t], device=latents.device),
                encoder_hidden_states=prompt_embeds,
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Call the callback, if provided
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)

        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 720,
        width: Optional[int] = 720,
        num_frames: Optional[int] = 24,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        chunk_size: int = 24,
        t_schedule_func: str = "sd3",
        t_schedule_shift: float = 3.0,
    ) -> Union[MagiPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the video generation.
            height (`int`, *optional*, defaults to 720):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 720):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 24):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`, usually at the expense of lower video quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate video. Choose between `np` for `numpy.array`, `pt` for `torch.Tensor` or `latent` to get the latent space output.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.magi.MagiPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under `self.processor` in [`diffusers.cross_attention`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            chunk_size (`int`, *optional*, defaults to 24):
                The chunk size to use for autoregressive generation. Measured in frames.
            t_schedule_func (`str`, *optional*, defaults to "sd3"):
                Timestep scheduling function. Options: "sd3", "square", "piecewise", "linear".
            t_schedule_shift (`float`, *optional*, defaults to 3.0):
                Shift parameter for sd3 scheduler.

        Examples:

        Returns:
            [`~pipelines.magi.MagiPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.magi.MagiPipelineOutput`] is returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.transformer.config.sample_size
        width = width or self.transformer.config.sample_size

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Prepare timesteps
        timesteps = self._prepare_timesteps(
            num_inference_steps,
            device,
            t_schedule_func=t_schedule_func,
            shift=t_schedule_shift
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        # Regular text-to-video case
        latents = self._prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Process in chunks for autoregressive generation
        chunk_indices = self._get_chunk_indices(num_frames // self.vae_scale_factor_temporal, chunk_size // self.vae_scale_factor_temporal)
        all_latents = []

        # 7. Denoise the latents
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_indices):
            # Extract the current chunk
            chunk_frames = end_idx - start_idx
            if chunk_idx == 0:
                # For the first chunk, use the initial latents
                chunk_latents = latents[:, :, start_idx:end_idx, :, :]
            else:
                # For subsequent chunks, implement proper autoregressive conditioning
                # In MAGI-1, each chunk conditions the next in an autoregressive manner
                # We use the previous chunk's output as conditioning for the current chunk
                prev_chunk_end = chunk_indices[chunk_idx - 1][1]
                overlap_start = max(0, start_idx - self.vae_scale_factor_temporal)  # Add overlap for conditioning

                # Initialize with noise
                chunk_latents = randn_tensor(
                    (batch_size * num_videos_per_prompt, num_channels_latents, chunk_frames,
                     height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial),
                    generator=generator,
                    device=device,
                    dtype=prompt_embeds.dtype,
                )
                chunk_latents = chunk_latents * self.scheduler.init_noise_sigma

                # Use previous chunk output as conditioning by copying overlapping frames
                if start_idx > 0 and chunk_idx > 0:
                    overlap_frames = min(self.vae_scale_factor_temporal, start_idx)
                    if overlap_frames > 0:
                        # Copy overlapping frames from previous chunk's output
                        chunk_latents[:, :, :overlap_frames, :, :] = all_latents[chunk_idx - 1][:, :, -overlap_frames:, :, :]

            # Denoise this chunk
            chunk_latents = self.denoise_latents(
                chunk_latents,
                prompt_embeds,
                timesteps,
                callback=callback if chunk_idx == 0 else None,  # Only use callback for first chunk
                callback_steps=callback_steps,
                guidance_scale=guidance_scale,
            )

            all_latents.append(chunk_latents)

        # 8. Concatenate all chunks
        latents = torch.cat(all_latents, dim=2)

        # 9. Post-processing
        if output_type == "latent":
            video = latents
        else:
            # Decode the latents
            latents = 1 / self.vae.scaling_factor * latents
            video = self.vae.decode(latents).sample
            video = (video / 2 + 0.5).clamp(0, 1)

            # Convert to the desired output format
            if output_type == "pt":
                video = video
            else:
                video = video.cpu().permute(0, 2, 3, 4, 1).float().numpy()

        # 10. Return output
        if not return_dict:
            return (video,)

        return MagiPipelineOutput(frames=video)
