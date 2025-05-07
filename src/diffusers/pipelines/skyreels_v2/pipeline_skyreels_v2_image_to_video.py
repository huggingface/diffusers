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

import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer

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

        >>> pipe = SkyReelsV2ImageToVideoPipeline.from_pretrained(
        ...     "SkyworkAI/SkyReels-V2-DiffusionForcing-4.0B", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> image = PIL.Image.open("input_image.jpg").convert("RGB")
        >>> prompt = "A beautiful view of mountains"
        >>> video_frames = pipe(prompt, image=image, num_frames=16).frames[0]
        ```
"""


class SkyReelsV2ImageToVideoPipeline(DiffusionPipeline):
    """
    Pipeline for image-to-video generation using SkyReels-V2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a specific device, etc.).

    Args:
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        transformer ([`WanTransformer3DModel`]):
            A SkyReels-V2 transformer model for diffusion.
        scheduler ([`FlowUniPCMultistepScheduler`]):
            A scheduler to be used in combination with the transformer to denoise the encoded video latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        transformer: WanTransformer3DModel,
        scheduler: FlowUniPCMultistepScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

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
    ):
        r"""
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
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizer parameters
        if max_sequence_length is None:
            max_sequence_length = self.tokenizer.model_max_length

        # Get prompt text embeddings
        if prompt_embeds is None:
            # Text encoder expects tokens to be of shape (batch_size, context_length)
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

        # Duplicate prompt embeddings for each generation per prompt
        if prompt_embeds.shape[0] < batch_size * num_videos_per_prompt:
            prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        # Get negative prompt embeddings
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"Batch size of `negative_prompt` should be {batch_size}, but is {len(negative_prompt)}"
                )

            negative_text_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_input_ids = negative_text_inputs.input_ids
            negative_attention_mask = negative_text_inputs.attention_mask

            negative_prompt_embeds = self.text_encoder(
                negative_input_ids.to(device),
                attention_mask=negative_attention_mask.to(device),
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        # Duplicate negative prompt embeddings for each generation per prompt
        if negative_prompt_embeds.shape[0] < batch_size * num_videos_per_prompt:
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        # For classifier-free guidance, combine embeddings
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode the generated latent sample using the VAE to produce video frames.

        Args:
            latents (`torch.Tensor`): Generated latent samples from the diffusion process.

        Returns:
            `torch.Tensor`: Decoded video frames.
        """
        video_length = latents.shape[2]

        latents = 1 / self.vae.config.scaling_factor * latents  # scale latents

        # Reshape latents from [batch, channels, frames, height, width] to [batch*frames, channels, height, width]
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            latents.shape[0] * latents.shape[2], latents.shape[1], latents.shape[3], latents.shape[4]
        )

        # Decode all frames
        video = self.vae.decode(latents).sample

        # Reshape back to [batch, frames, channels, height, width]
        video = video.reshape(-1, video_length, video.shape[1], video.shape[2], video.shape[3])

        # Rescale video from [-1, 1] to [0, 1]
        video = (video / 2 + 0.5).clamp(0, 1)

        # Rescale to pixel values
        video = (video * 255).to(torch.uint8)

        # Permute channels to [batch, frames, height, width, channels]
        return video.permute(0, 1, 3, 4, 2)

    def encode_image(self, image: Union[torch.Tensor, PIL.Image.Image]) -> torch.Tensor:
        """
        Encode the input image to latent space using VAE.

        Args:
            image (`torch.Tensor` or `PIL.Image.Image`): Input image to encode.

        Returns:
            `torch.Tensor`: Latent representation of the input image.
        """
        if isinstance(image, PIL.Image.Image):
            # Convert PIL image to tensor
            image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, torch.Tensor) and image.ndim == 3:
            # Add batch dimension for single image tensor
            image = image.unsqueeze(0)
        elif isinstance(image, torch.Tensor) and image.ndim == 4:
            # Ensure input is in -1 to 1 range
            if image.min() >= 0 and image.max() <= 1:
                image = 2.0 * image - 1.0
        else:
            raise ValueError(f"Invalid image input type: {type(image)}")

        image = image.to(device=self._execution_device, dtype=self.vae.dtype)

        # Encode the image using VAE
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents

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
        image_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare latent variables from noise for the diffusion process, optionally incorporating image conditioning.

        Args:
            batch_size (`int`): Number of samples to generate.
            num_channels_latents (`int`): Number of channels in the latent space.
            num_frames (`int`): Number of video frames to generate.
            height (`int`): Height of the generated images in pixels.
            width (`int`): Width of the generated images in pixels.
            dtype (`torch.dtype`): Data type of the latent variables.
            device (`torch.device`): Device to generate the latents on.
            generator (`torch.Generator` or List[`torch.Generator`], *optional*): One or a list of generators.
            latents (`torch.Tensor`, *optional*): Pre-generated noisy latent variables.
            image_latents (`torch.Tensor`, *optional*): Latent representation of the conditioning image.

        Returns:
            `torch.Tensor`: Prepared initial latent variables.
        """
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Must provide a list of generators of length {batch_size}, but list of length {len(generator)} was provided."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # Scale initial noise by the standard deviation
        latents = latents * self.scheduler.init_noise_sigma

        # If we have image conditioning, incorporate it into the first frame
        if image_latents is not None:
            # Expand image latents to match the number of frames by repeating along frame dimension
            # This helps provide a stronger image signal throughout the video
            image_latents = image_latents.unsqueeze(2)
            first_frame_latents = image_latents.expand(-1, -1, 1, -1, -1)

            # Create a stronger conditioning for the first frame
            # This helps ensure the video starts with the input image
            alpha = 0.8  # Higher alpha means stronger image conditioning
            latents[:, :, 0:1] = alpha * first_frame_latents + (1 - alpha) * latents[:, :, 0:1]

        return latents

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        num_frames: int = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        custom_shift: Optional[float] = None,
    ) -> Union[SkyReelsV2PipelineOutput, Tuple]:
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor` or `PIL.Image.Image`, *optional*):
                The image to use as the starting point for the video generation.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames to generate.
            height (`int`, *optional*, defaults to None):
                The height in pixels of the generated video frames. If not provided, height is automatically determined
                from the model configuration.
            width (`int`, *optional*, defaults to None):
                The width in pixels of the generated video frames. If not provided, width is automatically determined
                from the model configuration.
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
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
            output_type (`str`, *optional*, defaults to `"tensor"`):
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
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # 3. Determine whether to apply classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Encode input prompt
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

        # 5. Encode the input image
        image_latents = self.encode_image(image)

        # Duplicate image latents for each batch and prompt
        if isinstance(image, PIL.Image.Image) or (isinstance(image, torch.Tensor) and image.ndim < 4):
            # For a single image to be duplicated
            image_latents = image_latents.repeat(batch_size * num_videos_per_prompt, 1, 1, 1)

        # 6. Prepare timesteps
        timestep_shift = None if custom_shift is None else {"shift": custom_shift}
        self.scheduler.set_timesteps(num_inference_steps, device=device, **timestep_shift if timestep_shift else {})
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image_latents,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # Scale model input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents with the scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 9. Post-processing: decode latents
        video = self.decode_latents(latents)

        # 10. Convert output format
        if output_type == "np":
            video = video.cpu().numpy()
        elif output_type == "tensor":
            video = video.cpu()

        # 11. Offload all models
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (video,)

        return SkyReelsV2PipelineOutput(frames=video)
