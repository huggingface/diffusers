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
        >>> from diffusers import SkyReelsV2DiffusionForcingPipeline

        >>> # Load the pipeline
        >>> pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        ...     "SkyworkAI/SkyReels-V2-DiffusionForcing-4.0B", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Prepare conditioning frames (list of PIL Images or a tensor of shape [frames, height, width, channels])
        >>> frames = [PIL.Image.open(f"frame_{i}.jpg").convert("RGB") for i in range(5)]
        >>> # Create mask: 1 for conditioning frames, 0 for frames to generate
        >>> mask = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        >>> prompt = "A person walking in the park"
        >>> video = pipe(prompt, conditioning_frames=frames, conditioning_frame_mask=mask, num_frames=16).frames[0]
        ```
"""


class SkyReelsV2DiffusionForcingPipeline(DiffusionPipeline):
    """
    Pipeline for video generation with diffusion forcing (conditioning on specific frames) using SkyReels-V2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a specific device, etc.).

    Args:
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        transformer ([`SkyReelsV2TransformerModel`]):
            A SkyReels-V2 transformer model for diffusion with diffusion forcing capability.
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

    def encode_frames(self, frames: Union[List[PIL.Image.Image], torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Encode the conditioning frames to latent space using VAE.

        Args:
            frames (`List[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`):
                List of frames or tensor/array containing frames to encode.

        Returns:
            `torch.Tensor`: Latent representation of the input frames.
        """
        device = self._execution_device
        dtype = self.vae.dtype

        if isinstance(frames, list):
            # Convert list of PIL Images to tensor [frames, channels, height, width]
            processed_frames = []
            for frame in frames:
                if isinstance(frame, PIL.Image.Image):
                    frame = np.array(frame).astype(np.float32) / 127.5 - 1.0
                    frame = torch.from_numpy(frame).permute(2, 0, 1)
                processed_frames.append(frame)
            frames_tensor = torch.stack(processed_frames)

        elif isinstance(frames, np.ndarray):
            # Convert numpy array to tensor
            if frames.ndim == 4:  # [frames, height, width, channels]
                frames = frames.astype(np.float32) / 127.5 - 1.0
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [frames, channels, height, width]
            else:
                raise ValueError(
                    f"Unexpected numpy array shape: {frames.shape}, expected [frames, height, width, channels]"
                )

        elif isinstance(frames, torch.Tensor):
            if frames.ndim == 4:
                if frames.shape[1] == 3:  # [frames, channels, height, width]
                    frames_tensor = frames
                elif frames.shape[3] == 3:  # [frames, height, width, channels]
                    frames_tensor = frames.permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected tensor shape: {frames.shape}, cannot determine channel dimension")
            else:
                raise ValueError(f"Unexpected tensor shape: {frames.shape}, expected 4D tensor")

            # Ensure pixel values are in range [-1, 1]
            if frames_tensor.min() >= 0 and frames_tensor.max() <= 1:
                frames_tensor = 2.0 * frames_tensor - 1.0
            elif frames_tensor.min() >= 0 and frames_tensor.max() <= 255:
                frames_tensor = frames_tensor / 127.5 - 1.0
        else:
            raise ValueError(f"Unsupported frame input type: {type(frames)}")

        # Move to device and correct dtype
        frames_tensor = frames_tensor.to(device=device, dtype=dtype)

        # Process in batches if there are many frames, to avoid OOM
        batch_size = 8  # reasonable batch size for VAE encoding
        latents = []

        for i in range(0, frames_tensor.shape[0], batch_size):
            batch = frames_tensor[i : i + batch_size]
            with torch.no_grad():
                batch_latents = self.vae.encode(batch).latent_dist.sample()
                batch_latents = batch_latents * self.vae.config.scaling_factor
                latents.append(batch_latents)

        # Concatenate all batches
        latents = torch.cat(latents, dim=0)

        return latents

    def prepare_latents_with_forcing(
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
        conditioning_latents: Optional[torch.Tensor] = None,
        conditioning_frame_mask: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare latent variables for diffusion forcing.

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
            conditioning_latents (`torch.Tensor`, *optional*): Latent representations of conditioning frames.
            conditioning_frame_mask (`List[int]`, *optional*):
                Binary mask indicating which frames are conditioning frames.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: Prepared initial latent variables and forcing frame mask.
        """
        # Check if we have all required inputs for diffusion forcing
        if conditioning_frame_mask is None:
            raise ValueError("conditioning_frame_mask is required for diffusion forcing")

        if conditioning_latents is None:
            raise ValueError("conditioning_latents are required for diffusion forcing")

        # Ensure mask has the right length
        if len(conditioning_frame_mask) != num_frames:
            raise ValueError(
                f"conditioning_frame_mask length ({len(conditioning_frame_mask)}) must match num_frames ({num_frames})"
            )

        # Count conditioning frames in the mask
        num_cond_frames = sum(conditioning_frame_mask)

        # Check if conditioning_latents has correct number of frames
        if conditioning_latents.shape[0] != num_cond_frames:
            raise ValueError(
                f"Number of conditioning frames ({conditioning_latents.shape[0]}) must match "
                f"number of 1s in conditioning_frame_mask ({num_cond_frames})"
            )

        # Shape for full video latents
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

        # Generate or use provided latents
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # Scale initial noise by the standard deviation
        latents = latents * self.scheduler.init_noise_sigma

        # Create forcing mask tensor [batch, 1, frames, 1, 1]
        forcing_mask = torch.tensor(conditioning_frame_mask, device=device, dtype=dtype)
        forcing_mask = forcing_mask.view(1, 1, num_frames, 1, 1).expand(batch_size, 1, -1, 1, 1)

        # Insert conditioning latents at the correct positions based on mask
        cond_idx = 0
        for frame_idx, is_cond in enumerate(conditioning_frame_mask):
            if is_cond:
                # Replace the random noise with the encoded conditioning frame
                latents[:, :, frame_idx : frame_idx + 1] = (
                    conditioning_latents[cond_idx : cond_idx + 1].unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
                )
                cond_idx += 1

        return latents, forcing_mask

    def check_conditioning_inputs(
        self,
        conditioning_frames: Union[List[PIL.Image.Image], torch.Tensor, np.ndarray],
        conditioning_frame_mask: List[int],
        num_frames: int,
    ):
        """Check validity of conditioning inputs."""
        # Validate mask length
        if len(conditioning_frame_mask) != num_frames:
            raise ValueError(
                f"conditioning_frame_mask length ({len(conditioning_frame_mask)}) must match num_frames ({num_frames})"
            )

        # Validate mask values
        if not all(x in [0, 1] for x in conditioning_frame_mask):
            raise ValueError("conditioning_frame_mask must only contain 0s and 1s")

        # Count conditioning frames
        num_conditioning_frames = sum(conditioning_frame_mask)

        # Validate number of conditioning frames
        if isinstance(conditioning_frames, list):
            if len(conditioning_frames) != num_conditioning_frames:
                raise ValueError(
                    f"Number of conditioning frames ({len(conditioning_frames)}) must match "
                    f"number of 1s in conditioning_frame_mask ({num_conditioning_frames})"
                )
        elif isinstance(conditioning_frames, (torch.Tensor, np.ndarray)):
            if conditioning_frames.shape[0] != num_conditioning_frames:
                raise ValueError(
                    f"Number of conditioning frames ({conditioning_frames.shape[0]}) must match "
                    f"number of 1s in conditioning_frame_mask ({num_conditioning_frames})"
                )

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        conditioning_frames: Optional[Union[List[PIL.Image.Image], torch.Tensor, np.ndarray]] = None,
        conditioning_frame_mask: Optional[List[int]] = None,
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
        The call function to the pipeline for generation with diffusion forcing.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            conditioning_frames (`List[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`, *optional*):
                Frames to use as conditioning points during video generation. Should be provided as a list of PIL
                images, or as a tensor/array of shape [num_cond_frames, height, width, channels] or [num_cond_frames,
                channels, height, width].
            conditioning_frame_mask (`List[int]`, *optional*):
                Binary mask indicating which frames are conditioning frames (1) and which are to be generated (0). Must
                have the same length as num_frames and the same number of 1s as the number of conditioning_frames.
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
                Whether or not to return a [`~pipelines.SkyReelsV2PipelineOutput`] instead of a plain tuple.
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
            >>> from diffusers import SkyReelsV2DiffusionForcingPipeline

            >>> # Load the pipeline
            >>> pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
            ...     "SkyworkAI/SkyReels-V2-DiffusionForcing-4.0B", torch_dtype=torch.float16
            ... )
            >>> pipe = pipe.to("cuda")

            >>> # Prepare conditioning frames (list of PIL Images or a tensor of shape [frames, height, width, channels])
            >>> frames = [PIL.Image.open(f"frame_{i}.jpg").convert("RGB") for i in range(5)]
            >>> # Create mask: 1 for conditioning frames, 0 for frames to generate
            >>> mask = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            >>> prompt = "A person walking in the park"
            >>> video = pipe(prompt, conditioning_frames=frames, conditioning_frame_mask=mask, num_frames=16).frames[0]
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

        # Check diffusion forcing inputs
        if conditioning_frames is None or conditioning_frame_mask is None:
            raise ValueError("For diffusion forcing, conditioning_frames and conditioning_frame_mask must be provided")

        self.check_conditioning_inputs(conditioning_frames, conditioning_frame_mask, num_frames)

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

        # 5. Encode conditioning frames
        conditioning_latents = self.encode_frames(conditioning_frames)

        # 6. Prepare timesteps
        timestep_shift = None if custom_shift is None else {"shift": custom_shift}
        self.scheduler.set_timesteps(num_inference_steps, device=device, **timestep_shift if timestep_shift else {})
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables with forcing
        num_channels_latents = self.vae.config.latent_channels
        latents, forcing_mask = self.prepare_latents_with_forcing(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            conditioning_latents,
            conditioning_frame_mask,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # Scale model input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict the noise residual using Diffusion Forcing
                # Use standard forward pass; forcing logic is applied outside the model
                noise_pred = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents with the scheduler step
                latents_input = latents
                latents_updated = self.scheduler.step(noise_pred, t, latents_input).prev_sample

                # Apply forcing: use original latents for conditioning frames, updated latents for frames to generate
                # forcing_mask is 1 for conditioning frames, 0 for frames to generate
                latents = torch.where(forcing_mask, latents_input, latents_updated)

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
