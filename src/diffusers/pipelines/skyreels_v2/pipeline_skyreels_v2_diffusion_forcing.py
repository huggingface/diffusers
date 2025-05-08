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


EXAMPLE_DOC_STRING = """\
    Examples:
        ```py
        >>> import torch
        >>> import PIL.Image
        >>> from diffusers import SkyReelsV2DiffusionForcingPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> # Load the pipeline
        >>> pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        ...     "HF_placeholder/SkyReels-V2-DF-1.3B-540P", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Prepare conditioning frames (list of PIL Images)
        >>> # Example: Condition on frames 0, 24, 48, 72 for a 97-frame video
        >>> frame_0 = load_image("./frame_0.png")  # Placeholder paths
        >>> frame_24 = load_image("./frame_24.png")
        >>> frame_48 = load_image("./frame_48.png")
        >>> frame_72 = load_image("./frame_72.png")
        >>> conditioning_frames = [frame_0, frame_24, frame_48, frame_72]

        >>> # Create mask: 1 for conditioning frames, 0 for frames to generate
        >>> num_frames = 97  # Match the default
        >>> conditioning_frame_mask = [0] * num_frames
        >>> # Example conditioning indices for a 97-frame video
        >>> conditioning_indices = [0, 24, 48, 72]
        >>> for idx in conditioning_indices:
        ...     if idx < num_frames:  # Check bounds
        ...         conditioning_frame_mask[idx] = 1

        >>> prompt = "A person walking in the park"
        >>> video = pipe(
        ...     prompt=prompt,
        ...     conditioning_frames=conditioning_frames,
        ...     conditioning_frame_mask=conditioning_frame_mask,
        ...     num_frames=num_frames,
        ...     height=544,
        ...     width=960,
        ...     num_inference_steps=30,
        ...     guidance_scale=6.0,
        ...     custom_shift=8.0,
        ... ).frames
        >>> export_to_video(video, "skyreels_v2_df.mp4")
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
        transformer ([`WanTransformer3DModel`]):
            A SkyReels-V2 transformer model for diffusion with diffusion forcing capability.
        scheduler ([`FlowUniPCMultistepScheduler`]):
            A scheduler to be used in combination with the transformer to denoise the encoded video latents.
        video_processor ([`VideoProcessor`]):
            Processor for post-processing generated videos (e.g., tensor to numpy array).
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
        video_processor: VideoProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            video_processor=video_processor,
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

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # AutoencoderKLWan expects B, C, F, H, W latents directly
        video = self.vae.decode(latents).sample
        video = video.permute(0, 2, 1, 3, 4)
        video = (video / 2 + 0.5).clamp(0, 1)
        return video

    def encode_frames(self, frames: Union[List[PIL.Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Encodes conditioning frames into VAE latent space.

        Args:
            frames (`List[PIL.Image.Image]` or `torch.Tensor`):
                The conditioning frames (batch, frames, channels, height, width) or list of PIL images. Assumes frames
                are already preprocessed (e.g., correct size, range [-1, 1] if tensor).

        Returns:
            `torch.Tensor`: Latent representations of the frames (batch, channels, latent_frames, height, width).
        """
        if isinstance(frames, list):
            # Assume list of PIL Images, needs preprocessing similar to VAE requirements
            # Note: This uses a basic preprocessing, might need alignment with VaeImageProcessor
            frames_np = np.stack([np.array(frame.convert("RGB")) for frame in frames])
            frames_tensor = torch.from_numpy(frames_np).float() / 127.5 - 1.0  # Range [-1, 1]
            frames_tensor = frames_tensor.permute(
                0, 3, 1, 2
            )  # -> (batch*frames, channels, H, W) if flattened? No, needs batch dim.
            # Let's assume the input list is for a SINGLE batch item's frames.
            # Needs shape (batch=1, frames, channels, H, W) -> permute to (batch=1, channels, frames, H, W)
            frames_tensor = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        elif isinstance(frames, torch.Tensor):
            # Assume input tensor is already preprocessed and has shape (batch, frames, channels, H, W) or similar
            # Ensure range [-1, 1]
            if frames.min() >= 0.0 and frames.max() <= 1.0:
                frames = 2.0 * frames - 1.0
            # Permute to (batch, channels, frames, H, W)
            if frames.ndim == 5 and frames.shape[2] == 3:  # Check if channels is dim 2
                frames_tensor = frames.permute(0, 2, 1, 3, 4)
            elif frames.ndim == 5 and frames.shape[1] == 3:  # Check if channels is dim 1
                frames_tensor = frames  # Already in correct channel order
            else:
                raise ValueError("Input tensor shape not recognized. Expected (B, F, C, H, W) or (B, C, F, H, W).")
        else:
            raise TypeError("`conditioning_frames` must be a list of PIL Images or a torch Tensor.")

        frames_tensor = frames_tensor.to(device=self.device, dtype=self.vae.dtype)

        # Encode frames using VAE
        # Note: VAE encode expects (batch, channels, frames, height, width)? Check AutoencoderKLWan docs/code
        # AutoencoderKLWan._encode takes (B, C, F, H, W)
        conditioning_latents = self.vae.encode(frames_tensor).latent_dist.sample()
        conditioning_latents = conditioning_latents * self.vae.config.scaling_factor

        # Expected output shape: (batch, channels, latent_frames, latent_height, latent_width)
        return conditioning_latents

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
        conditioning_latents_sparse: Optional[torch.Tensor] = None,
        conditioning_frame_mask: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[bool]]:
        r"""
        Prepare latent variables, incorporating conditioning frames based on the mask.

        Args:
            batch_size (`int`): Number of samples to generate.
            num_channels_latents (`int`): Number of channels in the latent space.
            num_frames (`int`): Total number of video frames to generate.
            height (`int`): Height of the generated video in pixels.
            width (`int`): Width of the generated video in pixels.
            dtype (`torch.dtype`): Data type of the latent variables.
            device (`torch.device`): Device to generate the latents on.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*): Generator(s) for noise.
            latents (`torch.Tensor`, *optional*): Pre-generated noisy latents.
            conditioning_latents_sparse (`torch.Tensor`, *optional*): Latent representations of conditioning frames.
                 Shape: (batch, channels, num_cond_latent_frames, latent_H, latent_W).
            conditioning_frame_mask (`List[int]`, *optional*):
                Mask indicating which output frames are conditioned (1) or generated (0). Length must match
                `num_frames`.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor, List[bool]]`:
                - Prepared initial latent variables (noise).
                - Mask tensor in latent space indicating regions to generate (True) vs conditioned (False).
                - Boolean list representing the mask at the latent frame level (True=Conditioned).
        """
        # Calculate latent spatial shape
        shape_spatial = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        # Calculate temporal downsampling factor
        if hasattr(self.vae.config, "temperal_downsample") and self.vae.config.temperal_downsample is not None:
            num_true_temporal_downsamples = sum(1 for td in self.vae.config.temperal_downsample if td)
            temporal_downsample_factor = 2**num_true_temporal_downsamples
        else:
            temporal_downsample_factor = 4
            logger.warning(
                "VAE config does not have 'temperal_downsample'. Using default temporal_downsample_factor=4."
            )

        # Calculate number of latent frames required for the full output sequence
        num_latent_frames = (num_frames - 1) // temporal_downsample_factor + 1
        shape = (shape_spatial[0], shape_spatial[1], num_latent_frames, shape_spatial[2], shape_spatial[3])

        # Create initial noise latents
        if latents is None:
            initial_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            initial_latents = latents.to(device)

        # Create latent mask
        latent_mask_list_bool = [False] * num_latent_frames  # Default: All False (generate)
        if conditioning_latents_sparse is not None and conditioning_frame_mask is not None:
            if len(conditioning_frame_mask) != num_frames:
                raise ValueError("Length of conditioning_frame_mask must equal num_frames.")

            # Correct mapping from frame mask to latent frame mask
            num_conditioned_latents_expected = 0
            for j in range(num_latent_frames):
                start_frame_idx = j * temporal_downsample_factor
                end_frame_idx = min(start_frame_idx + temporal_downsample_factor, num_frames)
                # Check if any original frame corresponding to this latent frame is a conditioning frame (mask=1)
                is_latent_conditioned = any(
                    conditioning_frame_mask[k] == 1 for k in range(start_frame_idx, end_frame_idx)
                )
                latent_mask_list_bool[j] = (
                    is_latent_conditioned  # True if this latent frame corresponds to a conditioned frame
                )
                if is_latent_conditioned:
                    num_conditioned_latents_expected += 1

            # Validate the number of conditioning latents provided vs expected
            if conditioning_latents_sparse.shape[2] != num_conditioned_latents_expected:
                logger.warning(
                    f"Number of provided conditioning latents (frame dim: {conditioning_latents_sparse.shape[2]}) does not match "
                    f"the number of latent frames marked for conditioning ({num_conditioned_latents_expected}) based on the mask and stride. "
                    f"Ensure encode_frames provides latents only for the necessary frames."
                )
                # This indicates a potential mismatch that could cause errors later.

            # Create the tensor mask for computations
            # latent_mask_list_bool is True for conditioned frames
            # We need a mask where True means GENERATE (inpaint area)
            latent_mask_tensor_cond = torch.tensor(
                latent_mask_list_bool, device=device, dtype=torch.bool
            )  # True=Conditioned
            latent_mask = ~latent_mask_tensor_cond.reshape(1, 1, num_latent_frames, 1, 1).expand_as(
                initial_latents
            )  # True = Generate
        else:
            # No conditioning, generate everything. Mask is all True (generate).
            latent_mask = torch.ones_like(initial_latents, dtype=torch.bool)
            latent_mask_list_bool = [False] * num_latent_frames  # No frames are conditioned

        # Scale the initial noise by the standard deviation required by the scheduler
        initial_latents = initial_latents * self.scheduler.init_noise_sigma

        # Return initial noise, mask (True=Generate), and boolean list (True=Conditioned)
        return initial_latents, latent_mask, latent_mask_list_bool

    def check_conditioning_inputs(
        self,
        conditioning_frames: Optional[Union[List[PIL.Image.Image], torch.Tensor]],
        conditioning_frame_mask: Optional[List[int]],
        num_frames: int,
    ):
        if conditioning_frames is None and conditioning_frame_mask is not None:
            raise ValueError("`conditioning_frame_mask` provided without `conditioning_frames`.")
        if conditioning_frames is not None and conditioning_frame_mask is None:
            raise ValueError("`conditioning_frames` provided without `conditioning_frame_mask`.")

        if conditioning_frames is not None:
            if not isinstance(conditioning_frame_mask, list) or not all(
                isinstance(i, int) for i in conditioning_frame_mask
            ):
                raise TypeError("`conditioning_frame_mask` must be a list of integers (0 or 1).")
            if len(conditioning_frame_mask) != num_frames:
                raise ValueError(
                    f"`conditioning_frame_mask` length ({len(conditioning_frame_mask)}) must equal `num_frames` ({num_frames})."
                )
            if not all(m in [0, 1] for m in conditioning_frame_mask):
                raise ValueError("`conditioning_frame_mask` must only contain 0s and 1s.")

            num_masked_frames = sum(conditioning_frame_mask)

            if isinstance(conditioning_frames, list):
                if not all(isinstance(f, PIL.Image.Image) for f in conditioning_frames):
                    raise TypeError("If `conditioning_frames` is a list, it must contain only PIL Images.")
                if len(conditioning_frames) != num_masked_frames:
                    raise ValueError(
                        f"Number of `conditioning_frames` ({len(conditioning_frames)}) must equal the number of 1s in `conditioning_frame_mask` ({num_masked_frames})."
                    )
            elif isinstance(conditioning_frames, torch.Tensor):
                # Assuming tensor shape is (num_masked_frames, C, H, W) or (B, num_masked_frames, C, H, W) etc.
                # A simple check on the frame dimension assuming it's the first or second dim after batch
                if not (
                    conditioning_frames.shape[0] == num_masked_frames
                    or (conditioning_frames.ndim > 1 and conditioning_frames.shape[1] == num_masked_frames)
                ):
                    # This check is basic and might need refinement based on expected tensor layout
                    logger.warning(
                        f"Number of frames in `conditioning_frames` tensor ({conditioning_frames.shape}) does not seem to match the number of 1s in `conditioning_frame_mask` ({num_masked_frames}). Ensure tensor shape is correct."
                    )
            else:
                raise TypeError("`conditioning_frames` must be a List[PIL.Image.Image] or torch.Tensor.")

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        conditioning_frames: Optional[Union[List[PIL.Image.Image], torch.Tensor]] = None,
        conditioning_frame_mask: Optional[List[int]] = None,
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
        max_sequence_length: Optional[int] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        custom_shift: Optional[float] = 8.0,
    ) -> Union[SkyReelsV2PipelineOutput, Tuple]:
        r"""
        Generate video frames conditioned on text prompts and optionally on specific input frames (diffusion forcing).

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide video generation. If not defined, prompt_embeds must be.
            conditioning_frames (`List[PIL.Image.Image]` or `torch.Tensor`, *optional*):
                Frames to condition on. Must be provided if `conditioning_frame_mask` is provided. If a list, should
                contain PIL Images. If a Tensor, assumes shape compatible with VAE input after batching.
            conditioning_frame_mask (`List[int]`, *optional*):
                A list of 0s and 1s with length `num_frames`. 1 indicates a conditioning frame, 0 indicates a frame to
                generate.
            num_frames (`int`, *optional*, defaults to 97):
                The total number of frames to generate in the video sequence.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated video.
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 6.0):
                Guidance scale for classifier-free guidance. Enabled when > 1.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompts for CFG.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                PyTorch Generator object(s) for deterministic generation.
            latents (`torch.Tensor`, *optional*):
                Pre-generated initial latents (noise). If provided, shape should match expected latent shape.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            max_sequence_length (`int`, *optional*):
                Maximum sequence length for tokenizer. Defaults to model max length (e.g., 77).
            output_type (`str`, *optional*, defaults to `"np"`):
                Output format: `"tensor"` (torch.Tensor) or `"np"` (list of np.ndarray).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return `SkyReelsV2PipelineOutput` or a tuple.
            callback (`Callable`, *optional*):
                Callback function called every `callback_steps` steps.
            callback_steps (`int`, *optional*, defaults to 1):
                Frequency of callback calls.
            cross_attention_kwargs (`dict`, *optional*):
                Keyword arguments passed to the attention processor.
            custom_shift (`float`, *optional*):
                Shift parameter for the `FlowUniPCMultistepScheduler`.

        Returns:
            [`~pipelines.skyreels_v2.pipeline_skyreels_v2_text_to_video.SkyReelsV2PipelineOutput`] or `tuple`.
        """
        # 0. Default height and width
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        self.check_conditioning_inputs(conditioning_frames, conditioning_frame_mask, num_frames)
        has_conditioning = conditioning_frames is not None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        elif isinstance(conditioning_frames, list) or isinstance(conditioning_frames, torch.Tensor):
            batch_size = 1  # Assuming single batch item from frames for now
        else:
            raise ValueError("Cannot determine batch size.")
        if has_conditioning and batch_size > 1:
            logger.warning("Batch size > 1 not fully tested with diffusion forcing.")
            batch_size = 1

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            max_sequence_length,
        )
        prompt_dtype = prompt_embeds.dtype

        # 4. Encode conditioning frames if provided
        conditioning_latents_sparse = None
        if has_conditioning:
            conditioning_latents_sparse = self.encode_frames(conditioning_frames)
            conditioning_latents_sparse = conditioning_latents_sparse.to(device=device, dtype=prompt_dtype)
            # Repeat for num_videos_per_prompt
            if conditioning_latents_sparse.shape[0] != batch_size * num_videos_per_prompt:
                conditioning_latents_sparse = conditioning_latents_sparse.repeat_interleave(
                    num_videos_per_prompt, dim=0
                )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=custom_shift)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables and mask
        num_channels_latents = self.vae.config.latent_channels
        # Pass conditioning_latents_sparse to prepare_latents only for validation checks if needed
        latents, latent_mask, latent_mask_list_bool = self.prepare_latents_with_forcing(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_dtype,
            device,
            generator,
            latents=latents,
            conditioning_latents_sparse=conditioning_latents_sparse,
            conditioning_frame_mask=conditioning_frame_mask,
        )
        # latents = initial noise; latent_mask = True means generate; latent_mask_list_bool = True means conditioned

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare the known conditioned part (noised)
                noised_conditioning_latents_full = None
                if has_conditioning:
                    # Create a full-shaped tensor for the noised conditioning latents
                    full_conditioning_latents = torch.zeros_like(latents)
                    sparse_idx_counter = 0
                    for latent_idx, is_conditioned in enumerate(latent_mask_list_bool):
                        if is_conditioned:  # True means it *was* a conditioning frame
                            if sparse_idx_counter < conditioning_latents_sparse.shape[2]:
                                full_conditioning_latents[:, :, latent_idx, :, :] = conditioning_latents_sparse[
                                    :, :, sparse_idx_counter, :, :
                                ]
                                sparse_idx_counter += 1
                            # else: warning already issued in prepare_latents

                    noise = randn_tensor(
                        full_conditioning_latents.shape, generator=generator, device=device, dtype=prompt_dtype
                    )
                    # Noise the 'clean' conditioning latents appropriate for this timestep t
                    noised_conditioning_latents_full = self.scheduler.add_noise(full_conditioning_latents, noise, t)

                    # Combine current latents with noised conditioning latents using the mask
                    # latent_mask is True for generated regions, False for conditioned regions
                    model_input = torch.where(latent_mask, latents, noised_conditioning_latents_full)
                else:
                    model_input = latents

                # Expand for CFG
                latent_model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise
                # Note: Transformer sees the combined input (noise in generated areas, noised known in conditioned areas)
                model_pred = self.transformer(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # CFG
                if do_classifier_free_guidance:
                    model_pred_uncond, model_pred_text = model_pred.chunk(2)
                    model_pred = model_pred_uncond + guidance_scale * (model_pred_text - model_pred_uncond)

                # Scheduler step (operates on the full latents)
                step_output = self.scheduler.step(model_pred, t, latents)
                current_latents = step_output.prev_sample

                # Re-apply known conditioning information using the mask
                # Ensures the conditioned areas stay consistent with their noised versions
                if has_conditioning:
                    # Use the same noised_conditioning_latents_full calculated for timestep t
                    latents = torch.where(latent_mask, current_latents, noised_conditioning_latents_full)
                else:
                    latents = current_latents

                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        video_tensor = self._decode_latents(latents)
        # video_tensor shape should be (batch, frames, channels, height, width) float [0,1]

        # Use VideoProcessor for standard output formats
        video = self.video_processor.postprocess_video(video_tensor, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return SkyReelsV2PipelineOutput(frames=video)
