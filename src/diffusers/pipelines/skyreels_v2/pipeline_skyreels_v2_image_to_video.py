# Copyright 2024 The SkyReels-V2 Authors and The HuggingFace Team. All rights reserved.
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

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...image_processor import VideoProcessor
from ...models import AutoencoderKLWan, WanTransformer3DModel
from ...schedulers import FlowUniPCMultistepScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_skyreels_v2_text_to_video import SkyReelsV2PipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import PIL.Image
        >>> from diffusers import SkyReelsV2ImageToVideoPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> # Load the pipeline
        >>> pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
        ...     "HF_placeholder/SkyReels-V2-I2V-14B-540P", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Load the conditioning image
        >>> image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"  # Example image
        >>> image = load_image(image_url)

        >>> prompt = "A cat running across the grass"
        >>> video_frames = pipe(prompt=prompt, image=image, num_frames=97).frames
        >>> export_to_video(video_frames, "skyreels_v2_i2v.mp4")
        ```
"""


class SkyReelsV2ImageToVideoPipeline(DiffusionPipeline):
    """
    Pipeline for image-to-video generation using SkyReels-V2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a specific device, etc.).

    The pipeline is based on the Wan 2.1 architecture (WanTransformer3DModel, AutoencoderKLWan). It uses a
    `CLIPVisionModelWithProjection` to encode the conditioning image. It expects checkpoints saved in the standard
    diffusers format, typically including subfolders: `vae`, `text_encoder`, `tokenizer`, `image_encoder`,
    `transformer`, `scheduler`.

    Args:
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) model capable of encoding and decoding videos in latent space.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder (e.g., CLIP).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            Tokenizer corresponding to the `text_encoder`.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen image encoder (e.g., CLIP Vision Model) to encode the conditioning image.
        image_processor ([`~transformers.CLIPImageProcessor`]):
            Image processor corresponding to the `image_encoder`.
        transformer ([`WanTransformer3DModel`]):
            The core diffusion transformer model that denoises latents based on text and image conditioning.
        scheduler ([`FlowUniPCMultistepScheduler`]):
            A scheduler compatible with the Flow Matching framework used by SkyReels-V2.
        video_processor ([`VideoProcessor`]):
            Processor for converting VAE output latents to standard video formats (np, tensor, pil).
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "image_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        image_encoder: CLIPVisionModelWithProjection,
        image_processor: CLIPImageProcessor,
        transformer: WanTransformer3DModel,
        scheduler: FlowUniPCMultistepScheduler,
        video_processor: VideoProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
            scheduler=scheduler,
            video_processor=video_processor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # VaeImageProcessor is not needed here as CLIPImageProcessor handles image preprocessing.

    def _encode_image(
        self,
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Encodes the input image using the image encoder.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `List[PIL.Image.Image]`):
                Image or batch of images to encode.
            device (`torch.device`): Target device.
            num_videos_per_prompt (`int`): Number of videos per prompt (for repeating embeddings).
            do_classifier_free_guidance (`bool`): Whether to generate negative embeddings.
            dtype (`torch.dtype`): Target data type for embeddings.

        Returns:
            `torch.Tensor`: Encoded image embeddings.
        """
        if isinstance(image, PIL.Image.Image):
            image = [image]  # Processor expects a list

        # Preprocess image(s)
        image_pixels = self.image_processor(image, return_tensors="pt").pixel_values
        image_pixels = image_pixels.to(device=device, dtype=dtype)

        # Get image embeddings
        image_embeds = self.image_encoder(image_pixels).image_embeds  # [batch_size, seq_len, embed_dim]

        # Duplicate image embeddings for each generation per prompt
        image_embeds = image_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        # Get negative embeddings for CFG
        if do_classifier_free_guidance:
            negative_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_image_embeds, image_embeds])

        return image_embeds

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        lora_scale: Optional[float] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            device: (`torch.device`):
                The torch device to place the resulting embeddings on.
            num_videos_per_prompt (`int`):
                The number of videos that should be generated per prompt.
            do_classifier_free_guidance (`bool`):
                Whether to use classifier-free guidance or not.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                provide `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than 1).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            max_sequence_length (`int`, *optional*):
                Maximum sequence length for input text when generating embeddings. If not provided, defaults to 77.
            lora_scale (`float`, *optional*):
                Scale for LoRA-based text embeddings.
        """
        # Set LoRA scale
        lora_scale = lora_scale or self.lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if max_sequence_length is None:
            max_sequence_length = self.tokenizer.model_max_length

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = ""
                if isinstance(negative_prompt, str) and negative_prompt == "":
                    negative_prompt = [negative_prompt] * batch_size
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                if isinstance(negative_prompt, list) and batch_size != len(negative_prompt):
                    raise ValueError("Negative prompt batch size mismatch")
                uncond_input = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device), attention_mask=uncond_input.attention_mask.to(device)
                )[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            bs_embed, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode the generated latent sample using the VAE to produce video frames.

        Args:
            latents (`torch.Tensor`):
                Generated latent samples of shape (batch, channels, latent_frames, height, width).

        Returns:
            `torch.Tensor`: Decoded video frames of shape (batch, frames, channels, height, width) as a float tensor in
            range [0, 1].
        """
        # AutoencoderKLWan expects B, C, F, H, W latents directly
        video = self.vae.decode(latents).sample
        video = video.permute(0, 2, 1, 3, 4)  # B, F, C, H, W
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare latent variables from noise for the diffusion process.

        Args:
            batch_size (`int`):
                Number of samples to generate.
            num_channels_latents (`int`):
                Number of channels in the latent space.
            num_frames (`int`):
                Number of video frames to generate.
            height (`int`):
                Height of the generated video in pixels.
            width (`int`):
                Width of the generated video in pixels.
            dtype (`torch.dtype`):
                Data type of the latent variables.
            device (`torch.device`):
                Device to generate the latents on.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a random
                noisy latent is generated.

        Returns:
            `torch.Tensor`: Prepared initial latent variables.
        """
        vae_scale_factor = self.vae_scale_factor
        shape_spatial = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        shape_spatial = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if hasattr(self.vae.config, "temperal_downsample") and self.vae.config.temperal_downsample is not None:
            num_true_temporal_downsamples = sum(1 for td in self.vae.config.temperal_downsample if td)
            temporal_downsample_factor = 2**num_true_temporal_downsamples
        else:
            temporal_downsample_factor = 4
            logger.warning(
                "VAE config does not have 'temperal_downsample'. Using default temporal_downsample_factor=4."
            )

        num_latent_frames = (num_frames - 1) // temporal_downsample_factor + 1
        shape = (shape_spatial[0], shape_spatial[1], num_latent_frames, shape_spatial[2], shape_spatial[3])

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]]] = None,
        num_frames: int = 97,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        custom_shift: Optional[float] = 8.0,
    ) -> Union[SkyReelsV2PipelineOutput, Tuple]:
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                The image to use as the starting point for the video generation.
            num_frames (`int`, *optional*, defaults to 97):
                The number of video frames to generate.
            height (`int`, *optional*, defaults to None):
                The height in pixels of the generated video frames. If not provided, height is automatically determined
                from the model configuration.
            width (`int`, *optional*, defaults to None):
                The width in pixels of the generated video frames. If not provided, width is automatically determined
                from the model configuration.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 6.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            max_sequence_length (`int`, *optional*):
                Maximum sequence length for input text when generating embeddings. If not provided, defaults to 77.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `tensor` and `np` for `torch.Tensor` or
                `numpy.array` output respectively.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            custom_shift (`float`, *optional*):
                Custom shifting factor to use in the flow matching framework.

        Examples:
            ```py
            >>> import torch
            >>> import PIL.Image
            >>> from diffusers import SkyReelsV2ImageToVideoPipeline

            >>> pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
            ...     "SkyworkAI/SkyReels-V2-DiffusionForcing-4.0B", torch_dtype=torch.float16
            ... )
            >>> pipe = pipe.to("cuda")

            >>> image = PIL.Image.open("input_image.jpg").convert("RGB")
            >>> prompt = "A beautiful view of mountains"
            >>> video_frames = pipe(prompt, image=image, num_frames=16).frames[0]
            ```

        Returns:
            [`~pipelines.SkyReelsV2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.SkyReelsV2PipelineOutput`] is returned, otherwise a tuple is
                returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to transformer dimensions
        height = height or self.transformer.config.patch_size[1] * 112  # Default from SkyReels-V2: 224
        width = width or self.transformer.config.patch_size[2] * 112  # Default from SkyReels-V2: 224

        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            height,
            num_frames,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is None:
            raise ValueError("For image-to-video generation, an input image is required.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif image is not None:
            if isinstance(image, PIL.Image.Image):
                batch_size = 1
            elif isinstance(image, list) and all(isinstance(i, PIL.Image.Image) for i in image):
                batch_size = len(image)
            elif isinstance(image, torch.Tensor):
                batch_size = image.shape[0]
            else:
                # Fallback or error if image type is not recognized for batch size inference
                raise ValueError("Cannot determine batch size from the provided image type.")
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Either `prompt`, `image`, or `prompt_embeds` must be provided.")

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        # 4. Encode input image
        if image is None:
            # This case should ideally be caught by check_inputs or initial ValueError
            raise ValueError("`image` is a required argument for SkyReelsV2ImageToVideoPipeline.")

        image_embeds = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=custom_shift)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,  # Use prompt_embeds.dtype, image_embeds could be different
            device,
            generator,
            latents=latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                model_pred = self.transformer(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,  # Pass image_embeds here
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                if do_classifier_free_guidance:
                    model_pred_uncond, model_pred_text = model_pred.chunk(2)
                    model_pred = model_pred_uncond + guidance_scale * (model_pred_text - model_pred_uncond)

                latents = self.scheduler.step(model_pred, t, latents).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        video_tensor = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video_tensor, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return SkyReelsV2PipelineOutput(frames=video)
