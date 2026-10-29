# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ..modular_pipeline import SequentialPipelineBlocks
from .before_denoise import HyNomasPrepareStep
from .decoders import HyNomasDecodeStep
from .denoise import HyNomasSegmentLoop
from .encoders import HyNomasTextEncoderStep


# auto_docstring
class HyNomasAutoBlocks(SequentialPipelineBlocks):
    """
    Encode segment prompts, generate world latents with camera control, and decode video using PixelUMM.

      Components:
          text_encoder (`Qwen2_5_VLTextModel`) tokenizer (`AutoTokenizer`) transformer (`HyNomasTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) decoder (`PixelUMMDecoder`) video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str | list`):
              One prompt or an ordered list of prompts describing successive segments of a single video.
          max_sequence_length (`int`, *optional*, defaults to 1280):
              Number of text tokens retained per segment after removing the Qwen chat prefix; shorter sequences are
              padded.
          height (`int`, *optional*, defaults to 704):
              Output height in pixels; must be a positive multiple of 32 and match the existing session.
          width (`int`, *optional*, defaults to 1280):
              Output width in pixels; must be a positive multiple of 32 and match the existing session.
          num_frames (`int`, *optional*, defaults to 97):
              Exact number of new pixel frames to generate in this call, excluding all prior session history.
          segment_latent_lengths (`list`, *optional*):
              Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).
          camera_extrinsics (`Tensor`, *optional*):
              Normalized camera-to-world matrices in the training coordinate system, shaped (1, num_frames, 4, 4), for
              the new rollout only.
          fps (`float`, *optional*, defaults to 24.0):
              Frames per second used to convert structured action timestamps into latent-frame intervals.
          generator (`Generator`, *optional*):
              Generator for rollout noise and cache-write noise; omitted values restore the session generator or start
              from seed 42.
          latents (`Tensor`, *optional*):
              Optional initial noise for the new rollout, shaped (1, channels, ceil(num_frames / 4), height / 32, width
              / 32).
          noise_capacity (`int`, *optional*):
              Total latent-frame capacity reserved when creating a fixed session noise timeline, including future
              continuation segments.
          session (`HyNomasGenerationState`, *optional*):
              Earlier segment history to continue or branch; this call forks the state before appending new cache
              entries and latents.
          num_inference_steps (`int`, *optional*, defaults to 8):
              Number of flow-matching Euler denoising steps for each new segment.
          cache_noise_sigma (`float`, *optional*, defaults to 0.206897):
              Noise level in [0, 1] applied to a completed segment before committing its transformer keys and values to
              history.
          output_type (`str`, *optional*, defaults to np):
              Output representation: np for NumPy arrays, pt for tensors, pil for frame images, or latent to skip
              decoding.
          decoder_generator (`Generator`, *optional*):
              Independent PixelUMM noise generator; only its initial seed is used, without advancing its state. If
              omitted, the decoder configuration seed is used (42 by default).

      Outputs:
          prompts (`list`):
              Ordered prompts for successive segments of one video, rather than a batch of independent videos.
          prompt_embeds (`list`):
              Qwen hidden states for each segment, shaped (1, max_sequence_length, hidden_size), with the chat prefix
              removed.
          prompt_attention_masks (`list`):
              Per-segment text masks shaped (1, max_sequence_length), with one for valid tokens and zero for padding.
          action_text_offsets (`list`):
              Per-segment action-token boundaries in the cropped text sequence, or None for prompts without structured
              actions.
          cache_action_starts (`list`):
              Per-segment text boundary after which tokens are hidden from future segments; None keeps all valid text
              visible.
          session (`HyNomasGenerationState`):
              Forked generation state containing prior segment history and the optional fixed noise timeline.
          initial_latents (`Tensor`):
              Initial noise for the new rollout, shaped (1, channels, new_latent_frames, height / 32, width / 32).
          segment_latent_lengths (`list`):
              Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).
          segment_cameras (`list`):
              Normalized camera-to-world poses for each segment, shaped (1, latent_frames * 4, 4, 4), or None without
              camera conditioning.
          action_frame_windows (`list`):
              Per-segment (start, end, action_index) intervals in latent frames; -1 denotes an action gap, and None
              disables action-specific windows.
          generator (`Generator`):
              Generator initialized or restored for this rollout and its cache-write noise.
          segment_pixel_lengths (`list`):
              Exact visible pixel-frame count of each new segment after removing decoder padding from the final
              segment.
          segment_latents (`Tensor`):
              Denoised latents for the current segment, shaped (1, channels, segment_latent_frames, latent_height,
              latent_width).
          latents (`Tensor`):
              Concatenated denoised latents for the complete world history, including prior calls and every new
              segment.
          videos (`Tensor | ndarray | list`):
              Complete accumulated video including prior history: pt has shape (1, frames, 3, height, width), np has
              shape (1, frames, height, width, 3), and pil is a nested list of frames. The latent output preserves the
              input latent tensor.
    """

    model_name = "hy-nomas"
    block_classes = [HyNomasTextEncoderStep, HyNomasPrepareStep, HyNomasSegmentLoop, HyNomasDecodeStep]
    block_names = ["text_encoder", "prepare", "denoise", "decode"]

    @property
    def description(self):
        return "Encode segment prompts, generate world latents with camera control, and decode video using PixelUMM."
