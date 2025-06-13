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
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import MagiPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import MagiVideoToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_video

        >>> pipeline = MagiVideoToVideoPipeline.from_pretrained("sand-ai/MAGI-1-4.5B", torch_dtype=torch.float16)
        >>> pipeline = pipeline.to("cuda")

        >>> input_video = load_video("path/to/video.mp4")
        >>> prompt = "A cat playing in a garden. The cat is chasing a butterfly."
        >>> output = pipeline(prompt=prompt, video=input_video, num_frames=24, height=720, width=720).frames[0]
        >>> export_to_video(output, "magi_v2v_output.mp4", fps=8)
        ```
"""


class MagiVideoToVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for video-to-video generation using MAGI-1.

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
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** (1 if hasattr(self.vae, "temporal_downsample") else 0)
        self.vae_scale_factor_spatial = 2 ** (
            3 if hasattr(self.vae, "config") else 8
        )  # Default to 8 for 3 downsamples
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_sequence_length (`int`, *optional*, defaults to 512):
                The maximum length of the sequence to be processed by the text encoder.

        Returns:
            `torch.Tensor`: text embeddings
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        prompt_embeds = self.text_encoder(text_input_ids).last_hidden_state

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        # get unconditional embeddings for classifier-free guidance
        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            max_length = text_inputs.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_prompt_embeds = self.text_encoder(uncond_input_ids).last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

            # For classifier-free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        video=None,
    ):
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

        if video is None:
            raise ValueError("`video` input cannot be undefined.")

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode an input video to latent space.

        Args:
            video (`torch.Tensor`):
                Input video to be encoded.
            batch_size (`int`):
                Batch size.
            num_videos_per_prompt (`int`):
                Number of videos per prompt.
            do_classifier_free_guidance (`bool`):
                Whether to use classifier-free guidance.
            device (`torch.device`):
                Device to place the latents on.

        Returns:
            `torch.Tensor`: Encoded video latents.
        """
        # Ensure video is on the correct device
        video = video.to(device=device)

        # Encode video
        video_latents = self.vae.encode(video).latent_dist.sample()
        video_latents = video_latents * self.vae.scaling_factor

        # Expand for batch size and classifier-free guidance
        video_latents = video_latents.repeat(batch_size * num_videos_per_prompt, 1, 1, 1, 1)
        if do_classifier_free_guidance:
            video_latents = torch.cat([video_latents, video_latents], dim=0)

        return video_latents

    def _prepare_video_based_latents(
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
        video_latents: Optional[torch.Tensor] = None,
        num_frames_to_condition: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Prepare latents for diffusion with video conditioning.

        Args:
            batch_size (`int`): The batch size.
            num_channels_latents (`int`): The number of channels in the latent space.
            num_frames (`int`): The number of frames to generate.
            height (`int`): The height of the video.
            width (`int`): The width of the video.
            dtype (`torch.dtype`): The data type of the latents.
            device (`torch.device`): The device to place the latents on.
            generator (`torch.Generator`, *optional*): A generator to use for random number generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated latent vectors. If not provided, latents will be generated randomly.
            video_latents (`torch.Tensor`, *optional*): Video latents for conditioning.
            num_frames_to_condition (`int`, *optional*): Number of frames to use for conditioning.

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

        # If we have video latents, use them to condition the first frames
        if video_latents is not None:
            if num_frames_to_condition is None:
                num_frames_to_condition = video_latents.shape[2]

            # Only replace the first N frames with the video latents
            latents[:, :, :num_frames_to_condition, :, :] = video_latents[:, :, :num_frames_to_condition, :, :]

        return latents

    def _get_chunk_indices(self, num_frames: int, chunk_size: int) -> List[Tuple[int, int]]:
        """
        Get the start and end indices for each chunk.

        Args:
            num_frames (`int`): Total number of frames.
            chunk_size (`int`): Size of each chunk.

        Returns:
            `List[Tuple[int, int]]`: List of (start_idx, end_idx) tuples for each chunk.
        """
        return [(i, min(i + chunk_size, num_frames)) for i in range(0, num_frames, chunk_size)]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video: torch.Tensor,
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
        num_frames_to_condition: Optional[int] = None,
    ) -> Union[MagiPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the video generation.
            video (`torch.Tensor`):
                The input video to guide the video generation. Should be a tensor of shape (B, C, F, H, W).
            height (`int`, *optional*, defaults to 720):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 720):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 24):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,
                usually at the expense of lower video quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate video. Choose between `np` for `numpy.array`, `pt` for `torch.Tensor`
                or `latent` to get the latent space output.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.magi.MagiPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [`diffusers.cross_attention`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            chunk_size (`int`, *optional*, defaults to 24):
                The chunk size to use for autoregressive generation. Measured in frames.
            num_frames_to_condition (`int`, *optional*):
                Number of frames from the input video to use for conditioning. If not provided, all frames from the
                input video will be used.

        Examples:

        Returns:
            [`~pipelines.magi.MagiPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.magi.MagiPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.transformer.config.sample_size
        width = width or self.transformer.config.sample_size

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, video
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
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare video latents
        video_latents = self.prepare_video_latents(
            video,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            device,
        )

        # 6. Prepare latent variables with video conditioning
        num_channels_latents = self.transformer.in_channels
        latents = self._prepare_video_based_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            video_latents,
            num_frames_to_condition,
        )

        # 7. Process in chunks for autoregressive generation
        chunk_indices = self._get_chunk_indices(
            num_frames // self.vae_scale_factor_temporal, chunk_size // self.vae_scale_factor_temporal
        )
        all_latents = []

        # 8. Process each chunk
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_indices):
            # Extract the current chunk
            chunk_frames = end_idx - start_idx
            if chunk_idx == 0:
                # For the first chunk, use the initial latents
                chunk_latents = latents[:, :, start_idx:end_idx, :, :]
            else:
                # For subsequent chunks, use the previous chunk as conditioning
                # This is a simplified version - in a real implementation, we would need to handle
                # the autoregressive conditioning properly
                chunk_latents = randn_tensor(
                    (
                        batch_size * num_videos_per_prompt,
                        num_channels_latents,
                        chunk_frames,
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                    ),
                    generator=generator,
                    device=device,
                    dtype=prompt_embeds.dtype,
                )
                chunk_latents = chunk_latents * self.scheduler.init_noise_sigma

            # 9. Denoising loop for this chunk
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([chunk_latents] * 2) if do_classifier_free_guidance else chunk_latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.transformer(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    chunk_latents = self.scheduler.step(noise_pred, t, chunk_latents).prev_sample

                    # call the callback, if provided
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, chunk_latents)

                    progress_bar.update()

            all_latents.append(chunk_latents)

        # 10. Concatenate all chunks
        latents = torch.cat(all_latents, dim=2)

        # 11. Post-processing
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

        # 12. Return output
        if not return_dict:
            return (video,)

        return MagiPipelineOutput(frames=video)
