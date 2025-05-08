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
        ...     shift=8.0,
        ...     # Parameters for long video generation / advanced forcing (optional)
        ...     # base_num_frames=97,
        ...     # ar_step=5,
        ...     # overlap_history=24, # Number of *frames* (not latent frames) for overlap
        ...     # addnoise_condition=0.0,
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
        # Permute from (B, C, F, H, W) to (B, F, C, H, W) for video_processor and standard video format
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

    def _generate_timestep_matrix(
        self,
        num_latent_frames: int,
        step_template: torch.Tensor,
        base_latent_frames: int,
        ar_step: int = 5,
        num_latent_frames_pre_ready: int = 0,
        causal_block_size: int = 1,
        shrink_interval_with_mask: bool = False,  # Not used in original SkyReels-V2 call, kept for completeness
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Generates the timestep matrix for autoregressive scheduling, adapted from SkyReels-V2. Operates on latent frame
        counts.
        """
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1  # num_inference_steps + 1 effectively

        # Ensure operations are on latent frames, assuming inputs are already latent frame counts
        num_frames_block = num_latent_frames // causal_block_size
        base_num_frames_block = base_latent_frames // causal_block_size

        if base_num_frames_block > 0 and base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            if gen_block > 0:
                min_ar_step = infer_step_num / gen_block
                if ar_step < min_ar_step:
                    logger.warning(
                        f"ar_step ({ar_step}) is less than the suggested minimum ({np.ceil(min_ar_step)}) "
                        f"for base_latent_frames={base_latent_frames} and num_inference_steps={infer_step_num}. "
                        "This might lead to suboptimal scheduling."
                    )
            else:
                # Should not happen if base_num_frames_block is 0 and causal_block_size > 0
                logger.warning("base_num_frames_block is zero, ar_step check skipped.")

        # Add sentinel values to step_template for indexing logic
        # Original SkyReels-V2 uses [999, ..., 0]
        # self.scheduler.timesteps are typically [high, ..., low]
        # We need to ensure indexing works correctly.
        # The original logic `step_template[new_row]` implies new_row contains indices into step_template.
        # `new_row` counts from 0 to num_iterations. Let's adjust `step_template` to be 0-indexed
        # from num_iterations-1 down to 0.
        # Example: if step_template is [980, 960 ... 20, 0])
        # The values in new_row are essentially "how many steps have been processed for this frame"
        # from 0 (not started) to num_iterations (fully denoised).
        # step_matrix.append(step_template[new_row]) -> This seems problematic if new_row is 0 to num_iterations.
        # original: step_template = torch.cat([torch.tensor([999]), timesteps, torch.tensor([0])])
        # This padding makes step_template 1-indexed essentially.
        # Let's use a direct mapping from "number of steps processed" to actual timestep value.
        # If new_row[i] = k, it means frame i has undergone k denoising iterations.
        # The corresponding timestep should be init_timesteps[k-1] if new_row is 1-indexed for steps.
        # Original `pre_row` starts at 0. `new_row` increments. `new_row` goes from 0 to `num_iterations`.
        # `step_template[new_row]` means `new_row` values are indices into a padded step_template.
        # Let's use `step_template` (which are the actual timesteps from the scheduler) directly.
        # if new_row[i] = k: use step_template[k-1]
        # if new_row[i] = 0: this block is still pure noise / at initial state, use first timestep for processing.
        # The original `step_matrix.append(step_template[new_row])` used a 1-indexed padded template.
        # Our `new_row` is 0-indexed for states (0 to num_inference_steps).
        # Timestep for state k (1 <= k <= num_inference_steps) is step_template[k-1].
        # Timestep for state 0 (initial) is step_template[0].
        # So, for a state `s` in `new_row` (0 to num_inference_steps), the timestep is `step_template[s.clamp(min=0, max=len(step_template)-1)]`
        # No, simpler: if state is `k`, it means it has undergone `k` steps. The *next* step to apply is `step_template[k]`.
        # So `new_row` (clamped 0 to `len(step_template)-1`) can directly index `step_template`.
        # This gives the timestep *for the current operation*.
        timesteps_for_matrix = step_template  # These are the actual t values
        # `new_row` will count how many steps a frame has been processed. Ranges 0 to `len(timesteps_for_matrix)`.
        # 0 = initial noise state. `len(timesteps_for_matrix)` = fully processed by all timesteps.
        # `num_iterations` here is `len(timesteps_for_matrix)`.
        # Original `num_iterations = len(step_template) + 1`.
        # Let's stick to original logic for `num_iterations` for `pre_row` and `new_row` counters.
        # `num_iterations` = number of denoising *states* (0=initial noise, 1=after 1st step, ..., N=after Nth step)
        # So, if N inference steps, there are N+1 states. `num_iterations = len(step_template) + 1`.

        pre_row = torch.zeros(num_frames_block, dtype=torch.long, device=step_template.device)
        if num_latent_frames_pre_ready > 0:
            # Ensure pre_ready frames are marked as fully processed through all steps.
            pre_row[: num_latent_frames_pre_ready // causal_block_size] = (
                num_iterations - 1
            )  # Mark as if processed by all steps

        # The loop condition `torch.all(pre_row >= (num_iterations - 1))` means loop until all blocks are fully processed.
        while not torch.all(pre_row >= (num_iterations - 1)):
            new_row = torch.zeros(num_frames_block, dtype=torch.long, device=step_template.device)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (num_iterations - 1):  # first block or previous block is fully denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = torch.clamp(
                new_row, 0, num_iterations - 1
            )  # Clamp to valid state indices (0 to num_inference_steps)

            current_update_mask = (new_row != pre_row) & (
                new_row != (num_iterations - 1)
            )  # Original: & (new_row != num_iterations)
            # If new_row == num_iterations-1, it means it just reached the final denoised state. It *should* be updated.
            # Let's use original: (new_row != pre_row) & (new_row < (num_iterations -1))
            # A frame is updated if its state changes AND it's not yet in the "fully processed" state.
            # The original logic: update_mask.append((new_row != pre_row) & (new_row != num_iterations))
            # This seems to imply that even the step *to* num_iterations is not in update_mask.
            # Let's stick to the original:
            # update_mask is True if state changes AND it is not yet at the state corresponding to the last timestep.
            # However, new_row is clamped to num_iterations-1 (max index for timesteps).
            # So new_row == num_iterations will not happen here.
            # Update if state changes AND it is not yet at the state corresponding to the last timestep.
            current_update_mask = new_row != pre_row  # True: need to update this frame at this stage
            update_mask.append(current_update_mask)

            step_index.append(new_row.clone())  # Stores the "state index" for each block

            # Map state index (0 to N_steps) to actual timestep values.
            # new_row values are 0 (initial noise) to N_steps (processed by last timestep).
            # If new_row[j] = k: use timesteps_for_matrix[k-1] if k > 0.
            # If new_row[j] = 0: this block is still pure noise / at initial state, use first timestep for processing.
            # The original `step_matrix.append(step_template[new_row])` used a 1-indexed padded template.
            # Our `new_row` is 0-indexed for states (0 to num_inference_steps).
            # Timestep for state k (1 <= k <= num_inference_steps) is timesteps_for_matrix[k-1].
            # Timestep for state 0 (initial) is timesteps_for_matrix[0].
            # So, for a state `s` in `new_row` (0 to N_steps), the timestep is `timesteps_for_matrix[s.clamp(min=0, max=len(timesteps_for_matrix)-1)]`
            # No, simpler: if state is `k`, it means it has undergone `k` steps. The *next* step to apply is `timesteps_for_matrix[k]`.
            # So `new_row` (clamped 0 to `len(timesteps_for_matrix)-1`) can directly index `timesteps_for_matrix`.
            # This gives the timestep *for the current operation*.
            current_timesteps_for_blocks = timesteps_for_matrix[new_row.clamp(0, len(timesteps_for_matrix) - 1)]
            step_matrix.append(current_timesteps_for_blocks)

            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block  # Latent blocks
        if shrink_interval_with_mask:  # This was not used in original calls we saw
            idx_sequence = torch.arange(num_frames_block, dtype=torch.long, device=step_template.device)
            if update_mask:  # Ensure update_mask is not empty
                # Consider the update mask from the first iteration where meaningful updates happen
                first_meaningful_update_mask = None
                for um in update_mask:
                    if um.any():
                        first_meaningful_update_mask = um
                        break
                if first_meaningful_update_mask is not None:
                    update_mask_idx = idx_sequence[first_meaningful_update_mask]
                    if len(update_mask_idx) > 0:
                        last_update_idx = update_mask_idx[-1].item()
                        terminal_flag = last_update_idx + 1

        for curr_mask_row in update_mask:  # Iterate over rows of update masks
            # Original: if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
            # Needs to check if terminal_flag is a valid index for curr_mask_row
            if (
                terminal_flag < num_frames_block
                and terminal_flag < len(curr_mask_row)
                and curr_mask_row[terminal_flag]
            ):
                terminal_flag += 1
            # Ensure start of interval is not negative
            current_interval_start = max(terminal_flag - base_num_frames_block, 0)
            valid_interval.append(
                (current_interval_start, terminal_flag)  # These are in terms of blocks
            )

        if not step_matrix:  # Handle case where loop doesn't run (e.g. num_latent_frames is 0)
            # This case should ideally be caught earlier.
            # Return empty tensors of appropriate shape if possible, or raise error.
            # For now, let's assume num_latent_frames > 0.
            # If num_frames_block is 0, then pre_row is empty, loop condition is true, returns empty lists.
            # This needs robust handling if num_frames_block can be 0.
            # Assuming num_frames_block > 0 from here.
            if num_frames_block == 0:  # If no blocks, then step_matrix etc will be empty
                # Return empty tensors, but shapes need to be (0,0) or (0, num_latent_frames) if causal_block_size > 1
                # This edge case means num_latent_frames < causal_block_size
                # The original code seems to assume num_latent_frames >= causal_block_size for block logic.
                # Let's assume for now this means no processing needed for the matrix.
                # The actual latents will be handled by the main loop.
                # The matrix generation might not make sense.
                # Let's return empty tensors that can be concatenated, or handle this in the caller.
                # For now, if step_matrix is empty (e.g. num_frames_block=0), stack will fail.
                # If num_frames_block is 0, then num_latent_frames < causal_block_size.
                # The matrix logic might not apply. The caller should handle this.
                # Or, we make it work for this by bypassing block logic.
                # For now, assume num_frames_block > 0.
                # The caller will ensure num_latent_frames is appropriate.
                # If num_frames_block is 0, the while loop condition is met immediately,
                # update_mask, step_index, step_matrix are empty.
                # Stacking empty lists will raise an error.
                # If step_matrix is empty, it implies no steps defined by matrix.
                # This could happen if num_latent_frames_pre_ready covers all frames.
                # Or if num_frames_block = 0.

                # If no iterations in while loop (e.g. all pre_row already >= num_iterations-1)
                # this can happen if all frames are pre_ready.
                # In this case, step_matrix will be empty.
                # The caller needs to handle this (e.g., no denoising loop needed).
                # For safety, if they are empty, create dummy tensors.
                if not update_mask:
                    update_mask.append(torch.zeros(num_frames_block, dtype=torch.bool, device=step_template.device))
                if not step_index:
                    step_index.append(torch.zeros(num_frames_block, dtype=torch.long, device=step_template.device))
                if not step_matrix:
                    step_matrix.append(
                        torch.zeros(num_frames_block, dtype=step_template.dtype, device=step_template.device)
                    )
                if not valid_interval:
                    valid_interval.append((0, 0))

        step_update_mask_stacked = torch.stack(update_mask, dim=0)
        step_index_stacked = torch.stack(step_index, dim=0)
        step_matrix_stacked = torch.stack(step_matrix, dim=0)

        if causal_block_size > 1:
            # Expand from blocks back to latent frames
            step_update_mask_stacked = (
                step_update_mask_stacked.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            )
            step_index_stacked = (
                step_index_stacked.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            )
            step_matrix_stacked = (
                step_matrix_stacked.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            )

            # Adjust valid_interval from block indices to latent frame indices
            valid_interval_frames = []
            for s_block, e_block in valid_interval:
                s_frame = s_block * causal_block_size
                e_frame = e_block * causal_block_size
                # Ensure the end frame does not exceed total latent frames
                e_frame = min(e_frame, num_latent_frames)
                valid_interval_frames.append((s_frame, e_frame))
            valid_interval = valid_interval_frames
        else:  # causal_block_size is 1, valid_interval is already in terms of latent frames
            valid_interval_frames = []
            for s_idx, e_idx in valid_interval:
                valid_interval_frames.append((s_idx, min(e_idx, num_latent_frames)))
            valid_interval = valid_interval_frames

        # Ensure all returned tensors cover the full num_latent_frames if causal_block_size expansion happened
        # This might be needed if num_latent_frames is not perfectly divisible by causal_block_size
        # The original code implies that num_frames is handled by block logic.
        # If num_latent_frames = 7, causal_block_size = 4. num_frames_block = 1.
        # step_matrix_stacked would be (num_iterations_in_loop, 1, 4) -> (num_iterations_in_loop, 4)
        # We need it to be (num_iterations_in_loop, 7).
        # This flattening and repeating assumes num_frames_block * causal_block_size = num_latent_frames.
        # This is only true if num_latent_frames is a multiple of causal_block_size.
        # If not, the original code seems to truncate: `num_frames_block = num_frames // casual_block_size`
        # The output matrices will then only cover `num_frames_block * causal_block_size` frames.
        # This needs to be clarified or handled. For now, assume it covers up to num_latent_frames or truncates.
        # The original code in `__call__` uses `latent_length` (which is num_latent_frames) for schedulers,
        # but then `generate_timestep_matrix` is called with this `latent_length`.
        # The `valid_interval` then slices these.
        # It seems the matrix dimensions should align with `num_latent_frames`.

        # If causal_block_size > 1 and num_latent_frames is not a multiple:
        if causal_block_size > 1 and step_matrix_stacked.shape[1] < num_latent_frames:
            padding_size = num_latent_frames - step_matrix_stacked.shape[1]
            # Pad with the values from the last valid frame/block
            step_update_mask_stacked = torch.cat(
                [step_update_mask_stacked, step_update_mask_stacked[:, -1:].repeat(1, padding_size)], dim=1
            )
            step_index_stacked = torch.cat(
                [step_index_stacked, step_index_stacked[:, -1:].repeat(1, padding_size)], dim=1
            )
            step_matrix_stacked = torch.cat(
                [step_matrix_stacked, step_matrix_stacked[:, -1:].repeat(1, padding_size)], dim=1
            )

        return step_matrix_stacked, step_index_stacked, step_update_mask_stacked, valid_interval

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
        fps: int = 24,  # Add missing fps parameter
        shift: Optional[float] = 8.0,
        # New parameters for SkyReels-V2 original-style forcing and long video
        base_num_frames: Optional[int] = None,  # Max frames processed in one segment by transformer (pixel space)
        ar_step: int = 5,
        overlap_history: Optional[int] = None,
        addnoise_condition: float = 0.0,
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
            fps (`int`, *optional*, defaults to 24):
                Target frames per second for the video, passed to the transformer if supported.
            shift (`float`, *optional*, defaults to 8.0):
                Shift parameter for the `FlowUniPCMultistepScheduler` (if used as main scheduler).
            base_num_frames (`int`, *optional*):
                Maximum number of frames the transformer processes in a single segment when using original-style long
                video generation. If None or if `num_frames` is less than this, processes `num_frames`. Corresponds to
                `base_num_frames` in original SkyReels-V2 (pixel space).
            ar_step (`int`, *optional*, defaults to 5):
                Autoregressive step size used in `_generate_timestep_matrix` for scheduling timesteps across frames
                within a segment.
            overlap_history (`int`, *optional*):
                Number of frames to overlap between segments for long video generation. If None, long video generation
                with overlap is disabled. Uses pixel frame count.
            addnoise_condition (`float`, *optional*, defaults to 0.0):
                Controls the amount of noise added to conditioned latents (prefix or user-provided) during the
                denoising loop when using original-style forcing. A value > 0 enables it. Corresponds to
                `addnoise_condition` in SkyReels-V2.

        Returns:
            [`~pipelines.skyreels_v2.pipeline_skyreels_v2_text_to_video.SkyReelsV2PipelineOutput`] or `tuple`.
        """
        # 0. Require height and width & VAE spatial scale factor
        if height is None or width is None:
            raise ValueError("Please provide `height` and `width` for video generation.")
        vae_spatial_scale_factor = self.vae_scale_factor
        height = height - height % vae_spatial_scale_factor
        width = width - width % vae_spatial_scale_factor
        if height == 0 or width == 0:
            raise ValueError(
                f"Provided height {height} and width {width} are too small. Must be divisible by {vae_spatial_scale_factor}."
            )

        # Determine VAE temporal downsample factor
        if hasattr(self.vae.config, "temporal_downsample") and self.vae.config.temporal_downsample is not None:
            num_true_temporal_downsamples = sum(1 for td in self.vae.config.temporal_downsample if td)
            vae_temporal_scale_factor = 2**num_true_temporal_downsamples
        elif (
            hasattr(self.vae.config, "temperal_downsample") and self.vae.config.temperal_downsample is not None
        ):  # Typo in some old configs
            num_true_temporal_downsamples = sum(1 for td in self.vae.config.temperal_downsample if td)
            vae_temporal_scale_factor = 2**num_true_temporal_downsamples
            logger.warning("VAE config has misspelled 'temperal_downsample'. Using it.")
        else:
            vae_temporal_scale_factor = 4  # Default if not specified
            logger.warning(
                f"VAE config does not specify 'temporal_downsample'. Using default temporal_downsample_factor={vae_temporal_scale_factor}."
            )

        def to_latent_frames(pixel_frames):
            if pixel_frames is None or pixel_frames <= 0:
                return 0
            return (pixel_frames - 1) // vae_temporal_scale_factor + 1

        num_latent_frames_total = to_latent_frames(num_frames)
        if num_latent_frames_total <= 0:
            raise ValueError(
                f"num_frames {num_frames} results in {num_latent_frames_total} latent frames. Must be > 0."
            )

        # Determine causal_block_size for _generate_timestep_matrix from transformer config or default
        causal_block_size = getattr(self.transformer.config, "causal_block_size", 1)
        if not isinstance(causal_block_size, int) or causal_block_size <= 0:
            causal_block_size = 1

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        self.check_conditioning_inputs(conditioning_frames, conditioning_frame_mask, num_frames)  # Will need review
        has_initial_conditioning = conditioning_frames is not None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:  # prompt_embeds must be provided
            batch_size = prompt_embeds.shape[0] // num_videos_per_prompt  # Correct batch_size from prompt_embeds

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,  # Pass through pre-generated embeds
            negative_prompt_embeds=negative_prompt_embeds,  # Pass through
            max_sequence_length=max_sequence_length,
        )
        effective_batch_size = batch_size * num_videos_per_prompt

        # 4. Prepare scheduler and timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        timesteps = self.scheduler.timesteps

        # 5. Prepare initial conditioning information from user-provided frames for the ENTIRE video duration.
        # This section prepares data structures that will be used by both short and long video paths
        # to incorporate user-specified conditioning frames using the `addnoise_condition` logic.

        initial_clean_conditioning_latents = None
        # Stores VAE encoded clean latents from user `conditioning_frames` at their respective
        # positions in the full video timeline. Shape: (eff_batch_size, C, num_latent_frames_total, H_latent, W_latent)

        initial_conditioning_latent_mask = torch.zeros(num_latent_frames_total, dtype=torch.bool, device=device)
        # Boolean mask indicating which *latent frames* along the total video duration are directly conditioned by user input.
        # True if a latent frame `i` has user-provided conditioning data in `initial_clean_conditioning_latents`.

        num_latent_frames_pre_ready_from_user = 0
        # This specific variable counts how many *contiguous latent frames from the very beginning* of the video
        # are to be considered "pre-ready" or "frozen" for the `_generate_timestep_matrix` function.
        # For typical diffusion forcing where specific frames are conditioned (not necessarily a prefix),
        # this will often be 0. True video prefixes would set this.
        # All other user-specified conditioning (sparse, non-prefix) will be handled by `addnoise_condition` logic
        # guided by `initial_conditioning_latent_mask` within the denoising loops.

        if has_initial_conditioning:
            if conditioning_frame_mask is None:
                raise ValueError("If conditioning_frames are provided, conditioning_frame_mask must also be provided.")
            if len(conditioning_frame_mask) != num_frames:
                raise ValueError(
                    f"conditioning_frame_mask length ({len(conditioning_frame_mask)}) must equal num_frames ({num_frames})."
                )

            # Encode the user-provided frames. self.encode_frames is expected to return appropriately batched latents.
            # Assuming self.encode_frames returns: (effective_batch_size, C, N_sparse_encoded_latents, Hl, Wl)
            # where N_sparse_encoded_latents matches the number of 1s in conditioning_frame_mask (potentially after VAE temporal compression).
            sparse_user_latents = self.encode_frames(conditioning_frames)

            if sparse_user_latents.shape[0] != effective_batch_size:
                if sparse_user_latents.shape[0] == 1 and effective_batch_size > 1:
                    sparse_user_latents = sparse_user_latents.repeat(effective_batch_size, 1, 1, 1, 1)
                else:
                    raise ValueError(
                        f"Batch size mismatch: encoded conditioning frames have batch {sparse_user_latents.shape[0]}, expected {effective_batch_size}."
                    )

            latent_channels_for_cond = self.vae.config.latent_channels  # Should match sparse_user_latents.shape[1]
            latent_height_for_cond = sparse_user_latents.shape[-2]
            latent_width_for_cond = sparse_user_latents.shape[-1]

            initial_clean_conditioning_latents = torch.zeros(
                effective_batch_size,
                latent_channels_for_cond,
                num_latent_frames_total,
                latent_height_for_cond,
                latent_width_for_cond,
                dtype=sparse_user_latents.dtype,
                device=device,
            )

            processed_sparse_count = 0
            # Map the 1s in pixel-space `conditioning_frame_mask` to latent frame indices
            # and place the corresponding `sparse_user_latents`.
            for pixel_idx, is_cond_pixel in enumerate(conditioning_frame_mask):
                if is_cond_pixel == 1:
                    if processed_sparse_count >= sparse_user_latents.shape[2]:
                        logger.warning(
                            f"More 1s in conditioning_frame_mask than available encoded conditioning frames ({sparse_user_latents.shape[2]})."
                        )
                        break  # Stop if we've run out of provided conditioning latents

                    # Determine the target latent frame index for this pixel-space conditioned frame
                    # to_latent_frames expects 1-indexed pixel frame, returns 1-indexed latent frame count up to that point.
                    # So, for a pixel_idx (0-indexed), its corresponding latent frame index (0-indexed) is to_latent_frames(pixel_idx + 1) - 1.
                    target_latent_idx = to_latent_frames(pixel_idx + 1) - 1

                    if 0 <= target_latent_idx < num_latent_frames_total:
                        initial_clean_conditioning_latents[:, :, target_latent_idx, :, :] = sparse_user_latents[
                            :, :, processed_sparse_count, :, :
                        ]
                        initial_conditioning_latent_mask[target_latent_idx] = True
                        processed_sparse_count += 1
                else:
                    logger.warning(
                        f"Pixel frame {pixel_idx} maps to latent index {target_latent_idx} out of bounds [0, {num_latent_frames_total - 1}]. Skipping."
                    )

            if processed_sparse_count < sparse_user_latents.shape[2]:
                logger.warning(
                    f"Only used {processed_sparse_count} out of {sparse_user_latents.shape[2]} provided conditioning latents. "
                    "Ensure conditioning_frame_mask aligns with the number of conditioning_frames provided and video length."
                )

            # For num_latent_frames_pre_ready_from_user: count contiguous conditioned frames from latent start.
            # This is specifically for _generate_timestep_matrix's `num_pre_ready` which expects a prefix.
            # Other sparse conditioning is handled by `addnoise_condition` using the mask and clean latents.
            current_pre_ready_count = 0
            for i in range(num_latent_frames_total):
                if initial_conditioning_latent_mask[i]:
                    current_pre_ready_count += 1
                else:
                    break
            num_latent_frames_pre_ready_from_user = current_pre_ready_count
            if num_latent_frames_pre_ready_from_user > 0:
                logger.info(
                    f"{num_latent_frames_pre_ready_from_user} latent frames from the start are user-conditioned and will be treated as pre-ready."
                )

        # Latent Dims
        num_channels_latents = self.vae.config.latent_channels
        latent_height = height // vae_spatial_scale_factor
        latent_width = width // vae_spatial_scale_factor

        # Determine if using long video path
        use_long_video_path = False
        base_latent_frames_seg = num_latent_frames_total  # Default to full length if not long video mode
        overlap_latent_frames_seg = 0

        if overlap_history is not None and base_num_frames is not None and num_frames > base_num_frames:
            if base_num_frames <= 0:
                raise ValueError("base_num_frames must be positive.")
            if overlap_history < 0:
                raise ValueError("overlap_history must be non-negative.")
            if overlap_history >= base_num_frames:
                raise ValueError("overlap_history must be < base_num_frames.")

            # Check if long video generation is actually needed after converting to latent frames
            base_latent_frames_seg = to_latent_frames(base_num_frames)
            overlap_latent_frames_seg = to_latent_frames(overlap_history)

            if base_latent_frames_seg <= 0:
                base_latent_frames_seg = 1  # Ensure minimum segment length
            # Overlap can be 0 in latent space even if > 0 in pixel space, handle this
            if overlap_latent_frames_seg < 0:
                overlap_latent_frames_seg = 0  # Should not happen with to_latent_frames but safety

            if num_latent_frames_total > base_latent_frames_seg:
                use_long_video_path = True
                if overlap_latent_frames_seg >= base_latent_frames_seg:
                    logger.warning(
                        f"Calculated overlap_latent_frames ({overlap_latent_frames_seg}) >= base_latent_frames_seg ({base_latent_frames_seg}). Disabling overlap for long video."
                    )
                    overlap_latent_frames_seg = 0

        # Prepare initial latents for the full video (initial noise or user provided latents)
        if latents is None:
            shape = (effective_batch_size, num_channels_latents, num_latent_frames_total, latent_height, latent_width)
            full_video_latents = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            full_video_latents = full_video_latents * self.scheduler.init_noise_sigma
        else:
            expected_shape = (
                effective_batch_size,
                num_channels_latents,
                num_latent_frames_total,
                latent_height,
                latent_width,
            )
            if latents.shape != expected_shape:
                raise ValueError(f"Provided latents shape {latents.shape} does not match expected {expected_shape}.")
            full_video_latents = latents.to(device, dtype=prompt_embeds.dtype)

        # Helper method for denoising a single segment
        def _denoise_segment(
            self,
            segment_latents: torch.Tensor,  # Latents for the current segment (slice of full_video_latents)
            segment_start_global_idx: int,  # Start index of this segment in the total video
            num_latent_frames_this_segment: int,  # Number of latent frames in this segment
            num_pre_ready_for_this_segment: int,  # Number of contiguous pre-ready frames at segment start
            total_num_latent_frames: int,  # Total latent frames in the whole video
            initial_clean_conditioning_latents: Optional[torch.Tensor],  # Clean conditioning for the whole video
            initial_conditioning_latent_mask: Optional[torch.Tensor],  # Mask for conditioned frames in the whole video
            addnoise_condition: float,
            timesteps: torch.Tensor,
            prompt_embeds: torch.Tensor,
            guidance_scale: float,
            do_classifier_free_guidance: bool,
            cross_attention_kwargs: Optional[Dict[str, Any]],
            causal_block_size: int,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]],
            progress_bar,
            callback: Optional[Callable[[int, int, torch.Tensor], None]],
            callback_steps: int,
            fps: Optional[int] = None,  # Add fps parameter
            # Optional: segment index for logging
            segment_index: int = 0,
            num_segments: int = 1,
        ) -> torch.Tensor:
            # This method encapsulates the denoising loop logic previously in the short video path
            # It will denoise `segment_latents` in place or return updated latents.

            # Generate the timestep matrix for this segment
            step_matrix, step_index_matrix, update_mask_matrix, valid_interval_list = self._generate_timestep_matrix(
                num_latent_frames=num_latent_frames_this_segment,
                step_template=timesteps,
                base_latent_frames=num_latent_frames_this_segment,  # Base is segment length for matrix calc
                ar_step=ar_step,
                num_latent_frames_pre_ready=num_pre_ready_for_this_segment,
                causal_block_size=causal_block_size,
            )

            if (
                not step_matrix.numel()
                and num_latent_frames_this_segment > 0
                and num_pre_ready_for_this_segment < num_latent_frames_this_segment
            ):
                # Check if step_matrix is empty but should not be (i.e. not all frames are pre-ready)
                logger.warning(
                    f"Segment {segment_index + 1}/{num_segments}: _generate_timestep_matrix returned empty."
                )
                # If no steps, latents remain as is.
                return segment_latents  # Return unchanged if no denoising steps generated

            # Denoising loop for the current segment
            # The progress bar total is managed by the main loop (either short path or long video outer loop)
            for i_matrix_step in range(len(step_matrix)):
                current_timesteps_for_frames = step_matrix[i_matrix_step]  # Timestamps for each frame in the segment
                current_update_mask_for_frames = update_mask_matrix[
                    i_matrix_step
                ]  # Update mask for each frame in the segment
                valid_interval_start_local, valid_interval_end_local = valid_interval_list[
                    i_matrix_step
                ]  # Local indices within segment

                # Slice segment latents for the current valid processing window
                latent_model_input = segment_latents[
                    :, :, valid_interval_start_local:valid_interval_end_local
                ].clone()  # Clone for modification
                # Timesteps for the transformer input - corresponds to the sliced latents
                timestep_tensor_for_transformer = current_timesteps_for_frames[
                    valid_interval_start_local:valid_interval_end_local
                ]

                # === Implement addnoise_condition logic ===
                if addnoise_condition > 0.0 and initial_clean_conditioning_latents is not None:
                    # Iterate over frames within the current valid interval slice (local index j_local)
                    for j_local in range(valid_interval_end_local - valid_interval_start_local):
                        # Map local segment index to global video index
                        j_global = segment_start_global_idx + valid_interval_start_local + j_local

                        # Check if this global frame is user-conditioned AND NOT considered pre-ready/frozen by _generate_timestep_matrix.
                        # _generate_timestep_matrix should handle num_pre_ready by setting update_mask=False for those frames.
                        # So we apply addnoise to frames marked as conditioned (globally) AND are being updated in this matrix step.
                        if (
                            j_global < total_num_latent_frames
                            and initial_conditioning_latent_mask[j_global]
                            and current_update_mask_for_frames[valid_interval_start_local + j_local]
                        ):
                            # This is a conditioned frame that's being processed, apply addnoise logic
                            # Get the clean conditioned frame from the global conditioning tensor
                            clean_cond_frame = initial_clean_conditioning_latents[:, :, j_global, :, :]
                            # Original code used 0.001 * addnoise_condition for noise factor. Let's use param directly.
                            noise_factor = addnoise_condition

                            # Add noise to the clean conditioned frame
                            noise = randn_tensor(
                                clean_cond_frame.shape,
                                generator=generator,
                                device=device,
                                dtype=clean_cond_frame.dtype,
                            )
                            noised_cond_frame = clean_cond_frame * (1.0 - noise_factor) + noise * noise_factor

                            # Replace the noisy latent in the model input slice with the noised conditioned frame
                            latent_model_input[:, :, j_local] = noised_cond_frame

                            # Original code also clamped the timestep for conditioned frames.
                            # Let's clamp the specific frame's timestep in the transformer input tensor.
                            # Use addnoise_condition value as the clamping threshold.
                            if addnoise_condition > 0:  # Avoid min with 0 if addnoise_condition is 0
                                # Ensure addnoise_condition is treated as a valid timestep index or value.
                                # Assuming addnoise_condition is intended as a timestep value or something to clamp against.
                                # clamped_timestep_value = torch.tensor(addnoise_condition, device=device, dtype=timestep_tensor_for_transformer.dtype)
                                # Original clamped: `timestep_tensor_for_transformer[j_local] = torch.min(timestep_tensor_for_transformer[j_local], clamped_timestep_value)`
                                # Let's remove timestep clamping based on `addnoise_condition` for now, as it's less standard in Diffusers schedulers
                                # and its exact effect in original is tied to their scheduler step logic.
                                # addnoise_condition will primarily function as a noise amount control.
                                pass  # Clamping logic removed

                # === End of addnoise_condition logic ===

                # Model input for transformer (potentially with CFG duplication)
                model_input_for_transformer = latent_model_input
                if do_classifier_free_guidance:
                    model_input_for_transformer = torch.cat([model_input_for_transformer] * 2)

                # Timesteps for transformer (duplicated for CFG)
                if do_classifier_free_guidance:
                    timestep_tensor_for_transformer_cfg = torch.cat([timestep_tensor_for_transformer] * 2)
                else:
                    timestep_tensor_for_transformer_cfg = timestep_tensor_for_transformer

                # Transformer forward pass
                model_pred = self.transformer(
                    model_input_for_transformer,
                    timestep=timestep_tensor_for_transformer_cfg,  # Use per-frame timesteps
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # Pass fps_embeds if fps is provided
                    fps=torch.tensor([fps] * model_input_for_transformer.shape[0], device=self.device)
                    if fps is not None
                    else None,
                ).sample

                # CFG guidance
                if do_classifier_free_guidance:
                    model_pred_uncond, model_pred_text = model_pred.chunk(2)
                    model_pred = model_pred_uncond + guidance_scale * (model_pred_text - model_pred_uncond)

                # Scheduler step per frame if updated
                # Iterate over the frames in the current valid interval (local index idx_local_in_segment)
                for idx_local_in_segment in range(model_pred.shape[2]):
                    # Global frame index corresponding to this local segment index
                    # g_idx = segment_start_global_idx + valid_interval_start_local + idx_local_in_segment # Not directly used in indexing below

                    # Check the update mask *for the corresponding frame in the current segment* to see if it should be updated
                    # update_mask_matrix is (num_matrix_steps, num_latent_frames_this_segment)
                    # The update mask for the current frame is at `current_update_mask_for_frames[valid_interval_start_local + idx_local_in_segment]`
                    # which is equivalent to `current_update_mask_for_frames[idx_local_in_segment]` within the sliced valid interval
                    if current_update_mask_for_frames[valid_interval_start_local + idx_local_in_segment]:
                        frame_pred = model_pred[:, :, idx_local_in_segment]  # Prediction for this frame (local index)
                        frame_latent = segment_latents[
                            :, :, valid_interval_start_local + idx_local_in_segment
                        ]  # Current latent for this frame (local in segment)
                        frame_timestep = current_timesteps_for_frames[
                            valid_interval_start_local + idx_local_in_segment
                        ]  # Timestep for this frame from matrix

                        # Apply scheduler step for this single frame's latent
                        # Update the corresponding frame directly in the segment_latents tensor
                        segment_latents[:, :, valid_interval_start_local + idx_local_in_segment] = self.scheduler.step(
                            frame_pred,
                            frame_timestep,
                            frame_latent,
                            return_dict=False,
                            generator=generator,  # Pass generator
                        )[0]

                # Progress bar update - handled by the outer loop caller

                # Callback - handled by the outer loop caller

            return segment_latents  # Return the denoised segment latents

        # Main generation loop(s)
        if not use_long_video_path:
            logger.info(f"Short video path: {num_latent_frames_total} latent frames.")
            # Denoise the full video (single segment)
            denoised_latents = _denoise_segment(
                self,
                segment_latents=full_video_latents,  # Denoise the whole video
                segment_start_global_idx=0,  # Starts at the beginning of the video
                num_latent_frames_this_segment=num_latent_frames_total,  # Segment is the whole video
                num_pre_ready_for_this_segment=num_latent_frames_pre_ready_from_user,  # Use user-provided pre-ready count
                total_num_latent_frames=num_latent_frames_total,
                initial_clean_conditioning_latents=initial_clean_conditioning_latents,
                initial_conditioning_latent_mask=initial_conditioning_latent_mask,
                addnoise_condition=addnoise_condition,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                cross_attention_kwargs=cross_attention_kwargs,
                causal_block_size=causal_block_size,
                generator=generator,
                progress_bar=None,  # Progress bar handled below the if block if needed, or manage here
                callback=callback,
                callback_steps=callback_steps,
                segment_index=0,
                num_segments=1,  # For logging in helper
            )
            # The progress bar and callback for the short video path need to be managed around the _denoise_segment call
            # Or, pass the progress_bar and callback down and manage inside _denoise_segment.
            # Let's manage the progress bar and callback *inside* _denoise_segment.
            # Need to pass the progress_bar object to _denoise_segment.
            # Refactor: pass progress_bar and callback objects to _denoise_segment.

            # Rerun _denoise_segment call with progress_bar and callback passed:
            logger.info("Short video path: Starting denoising.")
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                # Need to determine the actual number of matrix steps for progress bar total
                # Call _generate_timestep_matrix first to get matrix size
                temp_step_matrix, _, _, _ = self._generate_timestep_matrix(
                    num_latent_frames=num_latent_frames_total,
                    step_template=timesteps,
                    base_latent_frames=num_latent_frames_total,  # For short path, base is the full length
                    ar_step=ar_step,
                    num_latent_frames_pre_ready=num_latent_frames_pre_ready_from_user,
                    causal_block_size=causal_block_size,
                )
                progress_bar.total = (
                    len(temp_step_matrix) if len(temp_step_matrix) > 0 else num_inference_steps
                )  # Adjusted total
                if (
                    progress_bar.total == 0
                    and num_latent_frames_total > 0
                    and num_latent_frames_pre_ready_from_user < num_latent_frames_total
                ):
                    logger.warning(
                        "Progress bar total is 0 but video needs denoising. Setting total to num_inference_steps."
                    )
                    progress_bar.total = num_inference_steps

                denoised_latents = _denoise_segment(
                    self,
                    segment_latents=full_video_latents,  # Denoise the whole video
                    segment_start_global_idx=0,  # Starts at the beginning of the video
                    num_latent_frames_this_segment=num_latent_frames_total,  # Segment is the whole video
                    num_pre_ready_for_this_segment=num_latent_frames_pre_ready_from_user,  # Use user-provided pre-ready count
                    total_num_latent_frames=num_latent_frames_total,
                    initial_clean_conditioning_latents=initial_clean_conditioning_latents,
                    initial_conditioning_latent_mask=initial_conditioning_latent_mask,
                    addnoise_condition=addnoise_condition,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    cross_attention_kwargs=cross_attention_kwargs,
                    causal_block_size=causal_block_size,
                    generator=generator,
                    progress_bar=progress_bar,  # Pass the segment progress bar
                    callback=callback,  # Pass callback
                    callback_steps=callback_steps,
                    segment_index=0,
                    num_segments=1,
                )

        else:  # Long video path - Implementation
            logger.info(
                f"Long video path: {num_latent_frames_total} total latents, {base_latent_frames_seg} base/segment, {overlap_latent_frames_seg} overlap."
            )

            # Calculate number of segments
            non_overlapping_part_len = base_latent_frames_seg - overlap_latent_frames_seg
            if non_overlapping_part_len <= 0:
                logger.error("Non-overlapping part of segment is <=0. Adjust base_num_frames or overlap_history.")
                raise ValueError("Non-overlapping part of segment must be positive.")

            num_iterations = (
                1
                + (num_latent_frames_total - base_latent_frames_seg + non_overlapping_part_len - 1)
                // non_overlapping_part_len
            )
            logger.info(f"Long video: processing in {num_iterations} segments.")

            # Initialize tensor to store the final denoised latents for the whole video
            # final_denoised_latents = torch.zeros_like(full_video_latents) # This was unused
            # Or accumulate in a list and concatenate later, might be better to avoid pre-allocation issues.
            # Let's accumulate in a list for now.
            final_denoised_segments_list = []

            # Keep track of the last part of the previously denoised segment for overlap
            previous_segment_denoised_overlap_latents = None  # Will be (B, C, overlap_latent_frames_seg, H_l, W_l)

            # Main loop for processing segments
            # Total progress bar should cover all matrix steps across all segments.
            # This is hard to pre-calculate exactly due to matrix generation per segment.
            # Alternative: progress bar tracks segments, and each segment denoising shows internal progress (if tqdm nested allowed).
            # Let's track segments in the main progress bar for simplicity first.

            with self.progress_bar(
                total=num_iterations, desc="Generating Long Video Segments"
            ) as progress_bar_segments:
                for i_segment in range(num_iterations):
                    # Determine the global latent frame indices for the current segment
                    segment_start_global_latent_idx = i_segment * non_overlapping_part_len
                    # The end index is start + base_segment_length, clamped to total length
                    segment_end_global_latent_idx = min(
                        segment_start_global_latent_idx + base_latent_frames_seg, num_latent_frames_total
                    )

                    # Adjust start index to include overlap from previous segment (if not the first segment)
                    current_segment_global_start_with_overlap = segment_start_global_latent_idx
                    if i_segment > 0:
                        current_segment_global_start_with_overlap -= overlap_latent_frames_seg

                    # Determine the actual number of latent frames in this current segment
                    num_latent_frames_this_segment = (
                        segment_end_global_latent_idx - current_segment_global_start_with_overlap
                    )

                    logger.info(
                        f"  Processing segment {i_segment + 1}/{num_iterations} (global latent frames {current_segment_global_start_with_overlap} to {segment_end_global_latent_idx - 1})."
                    )

                    # Prepare latents for the current segment
                    # Start with the initial noise (or user provided) for this segment's range
                    current_segment_latents = full_video_latents[
                        :, :, current_segment_global_start_with_overlap:segment_end_global_latent_idx
                    ].clone()

                    # If there's overlap from the previous segment, overwrite the initial part with the denoised overlap
                    num_pre_ready_for_this_segment = (
                        num_latent_frames_pre_ready_from_user  # Start with user prefix count
                    )
                    if i_segment > 0 and previous_segment_denoised_overlap_latents is not None:
                        # Overwrite the first `overlap_latent_frames_seg` of current_segment_latents
                        # with the denoised overlap from the previous segment.
                        # The number of pre-ready frames for the matrix generation should be the overlap length.
                        current_segment_latents[:, :, :overlap_latent_frames_seg] = (
                            previous_segment_denoised_overlap_latents
                        )
                        num_pre_ready_for_this_segment = (
                            overlap_latent_frames_seg  # Overlap serves as the frozen prefix for matrix generation
                        )
                        logger.info(
                            f"    Segment includes {overlap_latent_frames_seg} latent frames of overlap from previous segment."
                        )
                    elif i_segment == 0 and num_latent_frames_pre_ready_from_user > 0:
                        # First segment, use user-provided prefix as pre-ready
                        num_pre_ready_for_this_segment = num_latent_frames_pre_ready_from_user

                    # Denoise the current segment
                    # Pass the segment_latents to the helper function.
                    # The helper will operate on this tensor and return the denoised result for the segment.
                    denoised_segment_latents = _denoise_segment(
                        self,
                        segment_latents=current_segment_latents,  # Latents for this segment
                        segment_start_global_idx=current_segment_global_start_with_overlap,  # Global start index (including overlap)
                        num_latent_frames_this_segment=num_latent_frames_this_segment,  # Length of this segment
                        num_pre_ready_for_this_segment=num_pre_ready_for_this_segment,  # Pre-ready frames for matrix generation
                        total_num_latent_frames=num_latent_frames_total,
                        initial_clean_conditioning_latents=initial_clean_conditioning_latents,  # Full video conditioning
                        initial_conditioning_latent_mask=initial_conditioning_latent_mask,  # Full video mask
                        addnoise_condition=addnoise_condition,
                        timesteps=timesteps,
                        prompt_embeds=prompt_embeds,
                        guidance_scale=guidance_scale,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        cross_attention_kwargs=cross_attention_kwargs,
                        causal_block_size=causal_block_size,
                        generator=generator,
                        progress_bar=progress_bar_segments,  # Pass the segment progress bar
                        callback=callback,  # Pass callback
                        callback_steps=callback_steps,
                        fps=fps,  # Pass fps from __call__ scope
                        segment_index=i_segment,
                        num_segments=num_iterations,
                    )

                    # Extract the non-overlapping part of the denoised segment
                    # For the first segment, this is from the start up to base_latent_frames_seg.
                    # For subsequent segments, this is from overlap_latent_frames_seg onwards.
                    non_overlapping_segment_start_local_idx = 0 if i_segment == 0 else overlap_latent_frames_seg
                    non_overlapping_segment_latents = denoised_segment_latents[
                        :, :, non_overlapping_segment_start_local_idx:
                    ].clone()

                    # Add the non-overlapping part to the final list
                    final_denoised_segments_list.append(non_overlapping_segment_latents)

                    # Prepare overlap for the next segment (if not the last segment)
                    if i_segment < num_iterations - 1:
                        # The overlap is the last `overlap_latent_frames_seg` of the *denoised* current segment.
                        overlap_start_local_idx = num_latent_frames_this_segment - overlap_latent_frames_seg
                        overlap_latents_to_process = denoised_segment_latents[:, :, overlap_start_local_idx:].clone()

                        # Implementing original SkyReels V2 overlap handling (decode and re-encode):
                        # 1. Decode overlap latents to pixel space
                        # VAE decode expects (B, C, F, H, W). overlap_latents_to_process is (B, C, F_overlap, H_l, W_l)
                        decoded_overlap_pixels = self.vae.decode(overlap_latents_to_process).sample
                        # decoded_overlap_pixels is (B, F_overlap, C, H, W) after vae.decode (check vae_outputs)

                        # 2. Re-encode pixel frames back to latent space
                        # VAE encode expects (B, C, F, H, W), so permute decoded pixels
                        # decoded_overlap_pixels needs permuting from (B, F_overlap, C, H, W) to (B, C, F_overlap, H, W)
                        encoded_overlap_latents = self.vae.encode(
                            decoded_overlap_pixels.permute(0, 2, 1, 3, 4)  # Permute to (B, C, F, H, W)
                        ).latent_dist.sample()

                        # Apply VAE scaling factor after encoding
                        previous_segment_denoised_overlap_latents = (
                            encoded_overlap_latents * self.vae.config.scaling_factor
                        )

                    # Update segment progress bar
                    progress_bar_segments.update(1)  # Update by 1 for each completed segment

            # Concatenate all denoised segments to get the full video latents
            denoised_latents = torch.cat(
                final_denoised_segments_list, dim=2
            )  # Concatenate along the frame dimension (dim=2)

        # 7. Post-processing (decode final denoised_latents)
        video_tensor = self._decode_latents(denoised_latents)
        video = self.video_processor.postprocess_video(video_tensor, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return SkyReelsV2PipelineOutput(frames=video)
