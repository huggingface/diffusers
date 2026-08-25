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

import torch

from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    LTX2ConditionPrepareAudioLatentsStep,
    LTX2ConditionPrepareCoordsStep,
    LTX2ConditionSetTimestepsStep,
    LTX2DFRPlanStep,
    LTX2DFRPrepareLatentsStep,
    LTX2TextInputStep,
)
from .decoders import LTX2AudioDecoderStep, LTX2DFRSplitKeyframesStep, LTX2DiffusionVaeDecoderStep
from .denoise import LTX2ConditionDenoiseStep
from .modular_blocks_ltx2 import (
    LTX2AutoConditionEncoderStep,
    LTX2AutoDurationStep,
    LTX2AutoPromptEnhancerStep,
    LTX2TextConditioningStep,
)


# auto_docstring
class LTX2DFRCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise stage for one DFR pass. Identical to `LTX2ConditionCoreDenoiseStep` except for the prepare-latents
    block, which appends the generated keyframe slots and the optional spatial detailing reference. Everything
    downstream is unchanged: the slot marker rides to the transformer as a `denoiser_input_fields` output, and the slot
    coordinates ride in `appended_coords`.

      Components:
          transformer (`LTX2VideoTransformer3DModel`) vae (`AutoencoderKLLTX2Video`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider
          (`LTX2Guidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond).
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond).
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond).
          negative_connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (uncond).
          negative_connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (uncond).
          negative_connector_attention_mask (`Tensor`):
              Binary text attention mask (uncond).
          slot_frame_indices (`list`):
              Pixel-frame positions of the generated keyframe slots, from `LTX2DFRPlanStep`.
          keyframes_latents (`Tensor`, *optional*):
              `[B, C, num_slots, H, W]` content seeding the keyframe slots: a previous DFR pass's `keyframes_latents`
              upsampled to this pass's resolution. Denormalized, like every latent crossing the pipeline boundary.
              Slots start from noise when omitted.
          detailing_reference_latents (`Tensor`, *optional*):
              `[B, C, F, H, W]` latents appended as a fully clean in-context reference for the spatial detailing
              IC-LoRA: the previous pass's output at its own resolution, denormalized. Only meaningful with that
              adapter loaded.
          detailing_reference_downscale_factor (`int`, *optional*, defaults to 2):
              Ratio between this pass's resolution and the reference's, used to scale the reference tokens' spatial
              coordinates into the target coordinate space. Must match the factor the IC-LoRA was trained with.
          condition_latents (`list`, *optional*):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`):
              The padded canvas frame count, from `LTX2DFRPlanStep`.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to True):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (uncond), expanded per prompt.
          negative_connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (uncond), expanded per prompt.
          negative_connector_attention_mask (`Tensor`):
              Binary text attention mask (uncond), expanded per prompt.
          latents (`Tensor`):
              Packed noisy video latents, with keyframe, slot and reference tokens appended.
          conditioning_mask (`Tensor`):
              Packed per-token conditioning strengths of shape [B, S, 1] in [0, 1]: 1 at fully-conditioned positions, 0
              at free positions, including every keyframe slot.
          clean_latents (`Tensor`):
              Clean condition latents at conditioned positions, zeros elsewhere; same shape as `latents`.
          video_keyframes_mask (`Tensor`):
              Packed [B, S, 1] marker, 1 on tokens whose latent frame encodes a single pixel frame -- the causal first
              frame and every generated keyframe slot. Those tokens receive the transformer's
              `keyframes_abs_pos_embedding`.
          appended_coords (`Tensor`):
              RoPE coordinates of shape [B, 3, num_appended_tokens, 2] for the appended keyframe, slot and reference
              tokens, in the order they were appended.
          base_token_count (`int`):
              Number of generated-video tokens, i.e. the sequence length before appended tokens.
          slot_token_slice (`slice`):
              Slice of the packed sequence holding the generated keyframe slot tokens.
          noise_scale (`float`):
              The resolved initial noise level, forwarded to the audio latents step.
          timesteps (`Tensor`):
              TODO: Add description.
          num_inference_steps (`int`):
              TODO: Add description.
          audio_scheduler (`None`):
              Independent deep copy of `scheduler` used to update the audio latents in the loop.
          audio_latents (`Tensor`):
              Packed noisy audio latents.
          audio_num_frames (`int`):
              Number of audio latent frames.
          video_coords (`Tensor`):
              Video RoPE patch coordinates, with the keyframe-condition coordinates appended.
          audio_coords (`Tensor`):
              Audio RoPE patch coordinates.
    """

    model_name = "ltx2.5-dfr"
    block_classes = [
        LTX2TextInputStep,
        LTX2DFRPrepareLatentsStep,
        LTX2ConditionSetTimestepsStep,
        LTX2ConditionPrepareAudioLatentsStep,
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Core denoise stage for one DFR pass. Identical to `LTX2ConditionCoreDenoiseStep` except for the "
            "prepare-latents block, which appends the generated keyframe slots and the optional spatial detailing "
            "reference. Everything downstream is unchanged: the slot marker rides to the transformer as a "
            "`denoiser_input_fields` output, and the slot coordinates ride in `appended_coords`."
        )


# auto_docstring
class LTX2DFRDecoderStep(SequentialPipelineBlocks):
    """
    Decode stage for DFR: splits the generated keyframe slots out of the denoised sequence and trims the canvas
    padding, then denoises the video latents with the diffusion decoder and vocodes the audio latents (or returns
    latents).

      Components:
          diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) video_processor (`VideoProcessor`) audio_vae
          (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          base_token_count (`int`):
              Number of generated-video tokens, i.e. the sequence length before appended tokens.
          slot_token_slice (`slice`):
              Slice of the packed sequence holding the generated keyframe slot tokens.
          num_frames (`int`):
              The padded canvas frame count the sequence was generated on.
          requested_num_frames (`int`):
              The frame count the caller asked for, which the canvas is trimmed back to.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          dtype (`dtype`):
              The dtype of the model inputs, can be generated in input step.
          audio_latents (`Tensor`):
              Denoised audio latents.
          audio_num_frames (`int`):
              Number of audio latent frames, used to unpack the audio latent sequence.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
          keyframes_latents (`Tensor`):
              Denormalized `[B, C, num_slots, H, W]` generated keyframe slots. Upsample these alongside the video
              latents to seed the detailing pass.
    """

    model_name = "ltx2.5-dfr"
    block_classes = [LTX2DFRSplitKeyframesStep, LTX2DiffusionVaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["split_keyframes", "video_decode", "audio_decode"]

    @property
    def description(self):
        return (
            "Decode stage for DFR: splits the generated keyframe slots out of the denoised sequence and trims the "
            "canvas padding, then denoises the video latents with the diffusion decoder and vocodes the audio "
            "latents (or returns latents)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
            OutputParam(
                "keyframes_latents",
                type_hint=torch.Tensor,
                description=(
                    "Denormalized `[B, C, num_slots, H, W]` generated keyframe slots. Upsample these alongside the "
                    "video latents to seed the detailing pass."
                ),
            ),
        ]


# auto_docstring
class LTX2DFRBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for one LTX-2.5 Diffusion Fidelity Rendering (DFR) pass (joint video + audio).
      DFR generates on a canvas padded to a whole number of keyframe segments and spends one extra latent frame of
      tokens per segment border on a *keyframe slot*: a single-pixel-frame latent the model fills in. Relaxing the
      effective temporal compression at those positions means the surrounding video is conditioned on genuinely new
      frames rather than interpolated ones. The full recipe is two passes of these blocks. The first runs at half the
      target resolution and returns `videos`/`keyframes_latents` as latents; upsample both with
      `LTX2LatentUpsamplePipeline`, load the spatial detailing IC-LoRA, and run the blocks again at full resolution
      with `latents`, `keyframes_latents` and `detailing_reference_latents` supplied. Needs a transformer whose config
      sets `use_keyframes_abs_pos_embedding`, which LTX-2.5 checkpoints ship.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `condition`: requires `conditions`, `prompt`

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) transformer
          (`LTX2VideoTransformer3DModel`) vae (`AutoencoderKLLTX2Video`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`) diffusion_decoder
          (`LTX2VideoDiffusionDecoderModel`) video_processor (`VideoProcessor`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          conditions (`list`, *optional*):
              `LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
              generated video.
          enable_prompt_enhancement (`bool`, *optional*, defaults to False):
              Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.
          system_prompt (`str`, *optional*):
              System prompt for enhancement. Defaults to `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT` when a `PIL.Image.Image`
              condition frame is available, else `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`.
          prompt_max_new_tokens (`int`, *optional*):
              Maximum number of new tokens to generate during prompt enhancement. Defaults to 600, the LTX-2.5 Gemma-4
              enhancer's budget.
          prompt_enhancement_kwargs (`dict`, *optional*):
              Keyword arguments for the enhancer's `.generate` call. Defaults to greedy decoding.
          prompt_enhancement_seed (`int`, *optional*, defaults to 10):
              Random seed for prompt enhancement (inert under LTX-2.5's greedy decoding).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          min_seconds (`float`, *optional*, defaults to 1.0):
              Lower bound on the auto-predicted duration.
          max_seconds (`float`, *optional*, defaults to 20.0):
              Upper bound on the auto-predicted duration. Must be strictly greater than `min_seconds`.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          num_frames (`int`):
              The number of frames the caller asked for, before the canvas is padded onto the segment grid. Omit to
              auto-predict via the `duration_head` (see `LTX2AutoDurationStep`).
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          keyframes_latents (`Tensor`, *optional*):
              `[B, C, num_slots, H, W]` content seeding the keyframe slots: a previous DFR pass's `keyframes_latents`
              upsampled to this pass's resolution. Denormalized, like every latent crossing the pipeline boundary.
              Slots start from noise when omitted.
          detailing_reference_latents (`Tensor`, *optional*):
              `[B, C, F, H, W]` latents appended as a fully clean in-context reference for the spatial detailing
              IC-LoRA: the previous pass's output at its own resolution, denormalized. Only meaningful with that
              adapter loaded.
          detailing_reference_downscale_factor (`int`, *optional*, defaults to 2):
              Ratio between this pass's resolution and the reference's, used to scale the reference tokens' spatial
              coordinates into the target coordinate space. Must match the factor the IC-LoRA was trained with.
          condition_latents (`list`, *optional*):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          audio_latents (`Tensor`, *optional*):
              Optional pre-encoded audio latents; random noise is used when not provided.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to True):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
          keyframes_latents (`Tensor`):
              Denormalized `[B, C, num_slots, H, W]` generated keyframe slots. Upsample these alongside the video
              latents to seed the detailing pass.
    """

    model_name = "ltx2.5-dfr"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2DFRPlanStep,
        LTX2AutoConditionEncoderStep,
        LTX2DFRCoreDenoiseStep,
        LTX2DFRDecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "duration",
        "plan",
        "condition_encoder",
        "denoise",
        "decode",
    ]
    _workflow_map = {
        "text2video": {"prompt": True},
        "condition": {"conditions": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Modular pipeline blocks for one LTX-2.5 Diffusion Fidelity Rendering (DFR) pass (joint video + audio).\n"
            "DFR generates on a canvas padded to a whole number of keyframe segments and spends one extra latent "
            "frame of tokens per segment border on a *keyframe slot*: a single-pixel-frame latent the model fills "
            "in. Relaxing the effective temporal compression at those positions means the surrounding video is "
            "conditioned on genuinely new frames rather than interpolated ones.\n"
            "The full recipe is two passes of these blocks. The first runs at half the target resolution and "
            "returns `videos`/`keyframes_latents` as latents; upsample both with `LTX2LatentUpsamplePipeline`, load "
            "the spatial detailing IC-LoRA, and run the blocks again at full resolution with `latents`, "
            "`keyframes_latents` and `detailing_reference_latents` supplied. Needs a transformer whose config sets "
            "`use_keyframes_abs_pos_embedding`, which LTX-2.5 checkpoints ship."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
            OutputParam(
                "keyframes_latents",
                type_hint=torch.Tensor,
                description=(
                    "Denormalized `[B, C, num_slots, H, W]` generated keyframe slots. Upsample these alongside the "
                    "video latents to seed the detailing pass."
                ),
            ),
        ]
