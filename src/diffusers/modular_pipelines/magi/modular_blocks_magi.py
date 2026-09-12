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
from ..modular_pipeline_utils import OutputParam
from .before_denoise import MagiPrepareConditionedLatentsStep, MagiPrepareLatentsStep
from .decoders import MagiPrefixVaeDecoderStep, MagiVaeDecoderStep
from .denoise import MagiDenoiseStep, MagiPrefixDenoiseStep
from .encoders import MagiImageVaeEncoderStep, MagiTextEncoderStep, MagiVideoVaeEncoderStep


# auto_docstring
class MagiTextToVideoBlocks(SequentialPipelineBlocks):
    """
    Generate videos with a MAGI base model, using official HQ and duration conditioning.

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`AutoTokenizer`) transformer (`MagiTransformer3DModel`) vae
          (`AutoencoderKLMagi`) text_conditioning (`MagiTextConditioningModel`) scheduler (`MagiEulerScheduler`) guider
          (`MagiClassifierFreeGuidance`) video_processor (`VideoProcessor`)

      Configs:
          latent_scaling_factor (default: 0.18215)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 800):
              Padded T5 caption length.
          clean_caption (`bool`, *optional*, defaults to True):
              Apply the official two-pass text cleaning.
          height (`int`, *optional*, defaults to 720):
              Video height in pixels.
          width (`int`, *optional*, defaults to 720):
              Video width in pixels.
          num_frames (`int`, *optional*, defaults to 96):
              Requested video frames; generation rounds up to full chunks.
          chunk_width (`int`, *optional*, defaults to 6):
              Latent frames per chunk.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Optional initial FP32 noise for all generated chunks.
          prefix_latents (`Tensor`, *optional*):
              Not supported by this text-to-video preparation block.
          num_inference_steps (`int`, *optional*, defaults to 64):
              Number of Euler updates per generated chunk.
          window_size (`int`, *optional*, defaults to 4):
              Maximum number of simultaneously denoised chunks.
          noise2clean_kvrange (`tuple`, *optional*, defaults to (5, 4, 3, 2)):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`, *optional*, defaults to 1):
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`, *optional*, defaults to 0.9999):
              Model evaluation time for clean-prefix cache extraction.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          output_type (`str`, *optional*, defaults to np):
              Output format: pt, np, pil, or latent.

      Outputs:
          text_embeds (`Tensor`):
              Per-prompt FP32 T5 features before special-token insertion.
          text_attention_mask (`Tensor`):
              Per-prompt boolean T5 keep-mask.
          latents (`Tensor`):
              Denoised latents.
          prompt_embeds (`Tensor`):
              HQ/duration conditioned chunk text features.
          prompt_attention_mask (`Tensor`):
              Conditional text keep-mask.
          negative_prompt_embeds (`Tensor`):
              Learned null text features.
          negative_prompt_attention_mask (`Tensor`):
              Null text keep-mask.
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          clean_kv_cache (`tuple`):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
          chunk_start (`int`):
              First active chunk index.
          chunk_end (`int`):
              Exclusive end index of the active chunk window.
          refresh_cache (`bool`):
              Whether this iteration prepends a finalized chunk to refresh its clean KV.
          chunk_step_indices (`list`):
              Denoising step indices for active chunks, ordered from oldest to newest.
          chunk_times (`Tensor`):
              Current per-chunk model times shaped (batch, active_chunks).
          next_chunk_times (`Tensor`):
              Next Euler endpoints shaped (batch, active_chunks).
          model_times (`Tensor`):
              Model times including an optional clean-refresh chunk.
          latent_model_input (`Tensor`):
              Current latent window including an optional clean-refresh chunk.
          window_prompt_embeds (`Tensor`):
              Conditional features for the current window, with null features for a clean-refresh chunk.
          window_prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the current window text features.
          kv_ranges (`tuple`):
              Exclusive token attention ranges indexing the cached prefix plus the current window.
          velocity (`Tensor`):
              Three-way guided FP32 velocities for active chunks only.
          videos (`list`):
              The generated videos.
    """

    model_name = "magi"
    block_classes = [MagiTextEncoderStep, MagiPrepareLatentsStep, MagiDenoiseStep, MagiVaeDecoderStep]
    block_names = ["text_encoder", "prepare_latents", "denoise", "decode"]

    @property
    def outputs(self):
        return [OutputParam.template("latents") if param.name == "latents" else param for param in super().outputs]

    @property
    def description(self):
        return "Generate videos with a MAGI base model, using official HQ and duration conditioning."


# auto_docstring
class MagiImageToVideoBlocks(MagiTextToVideoBlocks):
    """
    Generate a MAGI base-model video conditioned on a pre-resized uint8 RGB image.

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`AutoTokenizer`) vae (`AutoencoderKLMagi`) transformer
          (`MagiTransformer3DModel`) text_conditioning (`MagiTextConditioningModel`) scheduler (`MagiEulerScheduler`)
          guider (`MagiClassifierFreeGuidance`) video_processor (`VideoProcessor`)

      Configs:
          latent_scaling_factor (default: 0.18215)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 800):
              Padded T5 caption length.
          clean_caption (`bool`, *optional*, defaults to True):
              Apply the official two-pass text cleaning.
          image (`Tensor`):
              Pre-resized uint8 RGB images, shaped (batch, 3, height, width).
          height (`int`, *optional*, defaults to 720):
              Video height in pixels.
          width (`int`, *optional*, defaults to 720):
              Video width in pixels.
          chunk_width (`int`, *optional*, defaults to 6):
              Latent frames per chunk.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Optional initial FP32 noise for all generated chunks.
          num_frames (`int`, *optional*, defaults to 96):
              Requested new frames; prefix plus new frames rounds up to full latent chunks.
          num_inference_steps (`int`, *optional*, defaults to 64):
              Number of Euler updates per generated chunk.
          window_size (`int`, *optional*, defaults to 4):
              Maximum number of simultaneously denoised chunks.
          noise2clean_kvrange (`tuple`, *optional*, defaults to (5, 4, 3, 2)):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`, *optional*, defaults to 1):
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`, *optional*, defaults to 0.9999):
              Model evaluation time for clean-prefix cache extraction.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          output_type (`str`, *optional*, defaults to np):
              Output format: pt, np, pil, or latent.

      Outputs:
          text_embeds (`Tensor`):
              Per-prompt FP32 T5 features before special-token insertion.
          text_attention_mask (`Tensor`):
              Per-prompt boolean T5 keep-mask.
          conditioning_latents (`Tensor`):
              Per-prompt scaled VAE prefix, before video-batch expansion.
          latents (`Tensor`):
              Denoised latents.
          prompt_embeds (`Tensor`):
              HQ/duration conditioned chunk text features.
          prompt_attention_mask (`Tensor`):
              Conditional text keep-mask.
          negative_prompt_embeds (`Tensor`):
              Learned null text features.
          negative_prompt_attention_mask (`Tensor`):
              Null text keep-mask.
          prefix_latents (`Tensor`):
              Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          clean_kv_cache (`tuple`):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
          chunk_start (`int`):
              First active chunk index.
          chunk_end (`int`):
              Exclusive end index of the active chunk window.
          refresh_cache (`bool`):
              Whether this iteration prepends a finalized chunk to refresh its clean KV.
          chunk_step_indices (`list`):
              Denoising step indices for active chunks, ordered from oldest to newest.
          chunk_times (`Tensor`):
              Current per-chunk model times shaped (batch, active_chunks).
          next_chunk_times (`Tensor`):
              Next Euler endpoints shaped (batch, active_chunks).
          model_times (`Tensor`):
              Model times including an optional clean-refresh chunk.
          latent_model_input (`Tensor`):
              Current latent window including an optional clean-refresh chunk.
          window_prompt_embeds (`Tensor`):
              Conditional features for the current window, with null features for a clean-refresh chunk.
          window_prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the current window text features.
          kv_ranges (`tuple`):
              Exclusive token attention ranges indexing the cached prefix plus the current window.
          velocity (`Tensor`):
              Three-way guided FP32 velocities for active chunks only.
          videos (`list`):
              The generated videos.
    """

    model_name = "magi"
    block_classes = [
        MagiTextEncoderStep,
        MagiImageVaeEncoderStep,
        MagiPrepareConditionedLatentsStep,
        MagiPrefixDenoiseStep,
        MagiPrefixVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "prepare_latents", "denoise", "decode"]

    @property
    def description(self):
        return "Generate a MAGI base-model video conditioned on a pre-resized uint8 RGB image."


# auto_docstring
class MagiVideoToVideoBlocks(MagiImageToVideoBlocks):
    """
    Continue pre-resized uint8 RGB video frames with a MAGI base model.

      Components:
          text_encoder (`T5EncoderModel`) tokenizer (`AutoTokenizer`) vae (`AutoencoderKLMagi`) transformer
          (`MagiTransformer3DModel`) text_conditioning (`MagiTextConditioningModel`) scheduler (`MagiEulerScheduler`)
          guider (`MagiClassifierFreeGuidance`) video_processor (`VideoProcessor`)

      Configs:
          latent_scaling_factor (default: 0.18215)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          max_sequence_length (`int`, *optional*, defaults to 800):
              Padded T5 caption length.
          clean_caption (`bool`, *optional*, defaults to True):
              Apply the official two-pass text cleaning.
          video (`Tensor`):
              Pre-resized uint8 RGB prefix, shaped (batch, 3, frames, height, width).
          height (`int`, *optional*, defaults to 720):
              Video height in pixels.
          width (`int`, *optional*, defaults to 720):
              Video width in pixels.
          chunk_width (`int`, *optional*, defaults to 6):
              Latent frames per chunk.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          latents (`Tensor`, *optional*):
              Optional initial FP32 noise for all generated chunks.
          num_frames (`int`, *optional*, defaults to 96):
              Requested new frames; prefix plus new frames rounds up to full latent chunks.
          num_inference_steps (`int`, *optional*, defaults to 64):
              Number of Euler updates per generated chunk.
          window_size (`int`, *optional*, defaults to 4):
              Maximum number of simultaneously denoised chunks.
          noise2clean_kvrange (`tuple`, *optional*, defaults to (5, 4, 3, 2)):
              Positive attention-window lengths in chunks, from early to late denoising stages.
          clean_chunk_kvrange (`int`, *optional*, defaults to 1):
              Positive attention-window length used when recomputing clean chunks.
          clean_t (`float`, *optional*, defaults to 0.9999):
              Model evaluation time for clean-prefix cache extraction.
          attention_kwargs (`dict`, *optional*):
              Optional keyword arguments passed to Transformer attention.
          cache_device (`str`, *optional*):
              Optional device for clean-prefix KV storage; use cpu to offload between layer evaluations.
          output_type (`str`, *optional*, defaults to np):
              Output format: pt, np, pil, or latent.

      Outputs:
          text_embeds (`Tensor`):
              Per-prompt FP32 T5 features before special-token insertion.
          text_attention_mask (`Tensor`):
              Per-prompt boolean T5 keep-mask.
          conditioning_latents (`Tensor`):
              Per-prompt scaled VAE prefix, before video-batch expansion.
          latents (`Tensor`):
              Denoised latents.
          prompt_embeds (`Tensor`):
              HQ/duration conditioned chunk text features.
          prompt_attention_mask (`Tensor`):
              Conditional text keep-mask.
          negative_prompt_embeds (`Tensor`):
              Learned null text features.
          negative_prompt_attention_mask (`Tensor`):
              Null text keep-mask.
          prefix_latents (`Tensor`):
              Optional full-chunk clean prefix; replaces the leading latent slots and remains unchanged.
          num_chunks (`int`):
              Total number of latent chunks, including the supplied prefix.
          prefix_chunks (`int`):
              Number of supplied full-chunk prefix chunks.
          chunk_tokens (`int`):
              Number of Transformer tokens per latent chunk.
          steps_per_stage (`int`):
              Number of iterations before the active chunk window moves forward.
          num_window_steps (`int`):
              Total number of asynchronous window iterations.
          timestep_schedule (`Tensor`):
              FP32 schedule including the final Euler integration endpoint.
          clean_kv_cache (`tuple`):
              Per-layer clean-prefix key/value tensors, or None before any prefix is cached; excludes the final
              generated chunk.
          completed_chunks (`list`):
              Indices of supplied prefix chunks and finalized generated chunks.
          chunk_start (`int`):
              First active chunk index.
          chunk_end (`int`):
              Exclusive end index of the active chunk window.
          refresh_cache (`bool`):
              Whether this iteration prepends a finalized chunk to refresh its clean KV.
          chunk_step_indices (`list`):
              Denoising step indices for active chunks, ordered from oldest to newest.
          chunk_times (`Tensor`):
              Current per-chunk model times shaped (batch, active_chunks).
          next_chunk_times (`Tensor`):
              Next Euler endpoints shaped (batch, active_chunks).
          model_times (`Tensor`):
              Model times including an optional clean-refresh chunk.
          latent_model_input (`Tensor`):
              Current latent window including an optional clean-refresh chunk.
          window_prompt_embeds (`Tensor`):
              Conditional features for the current window, with null features for a clean-refresh chunk.
          window_prompt_attention_mask (`Tensor`):
              Boolean keep-mask matching the current window text features.
          kv_ranges (`tuple`):
              Exclusive token attention ranges indexing the cached prefix plus the current window.
          velocity (`Tensor`):
              Three-way guided FP32 velocities for active chunks only.
          videos (`list`):
              The generated videos.
    """

    model_name = "magi"
    block_classes = [
        MagiTextEncoderStep,
        MagiVideoVaeEncoderStep,
        MagiPrepareConditionedLatentsStep,
        MagiPrefixDenoiseStep,
        MagiPrefixVaeDecoderStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "prepare_latents", "denoise", "decode"]

    @property
    def description(self):
        return "Continue pre-resized uint8 RGB video frames with a MAGI base model."
