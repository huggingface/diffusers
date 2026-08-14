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

from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .decoders import (
    LTX2AudioDecoderStep,
    LTX2DiffusionVaeDecoderStep,
    LTX2TrimConditionTokensStep,
)
from .modular_blocks_ltx2 import (
    LTX2AutoConditionEncoderStep,
    LTX2AutoCoreDenoiseStep,
    LTX2AutoDurationStep,
    LTX2AutoPromptEnhancerStep,
    LTX2AutoReferenceEncoderStep,
    LTX2AutoVaeEncoderStep,
    LTX2TextConditioningStep,
)


# auto_docstring
class LTX25DecoderStep(SequentialPipelineBlocks):
    """
    Decode stage for LTX-2.5: denoises the video latents with the diffusion decoder and vocodes the audio latents (or
    returns latents).

      Components:
          diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) video_processor (`VideoProcessor`) audio_vae
          (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              The number of frames in the generated video.
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
    """

    model_name = "ltx2.5"
    block_classes = [LTX2DiffusionVaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["video_decode", "audio_decode"]

    @property
    def description(self):
        return (
            "Decode stage for LTX-2.5: denoises the video latents with the diffusion decoder and vocodes the audio "
            "latents (or returns latents)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX25ConditionDecoderStep(SequentialPipelineBlocks):
    """
    Decode stage for LTX-2.5 condition workflows: drops the appended keyframe-condition tokens, then denoises the video
    latents with the diffusion decoder and vocodes the audio latents (or returns latents).

      Components:
          diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) video_processor (`VideoProcessor`) audio_vae
          (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          base_token_count (`int`):
              Number of generated-video tokens, i.e. the sequence length before appended tokens.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              The number of frames in the generated video.
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
    """

    model_name = "ltx2.5"
    block_classes = [LTX2TrimConditionTokensStep, LTX2DiffusionVaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["trim_condition_tokens", "video_decode", "audio_decode"]

    @property
    def description(self):
        return (
            "Decode stage for LTX-2.5 condition workflows: drops the appended keyframe-condition tokens, then "
            "denoises the video latents with the diffusion decoder and vocodes the audio latents (or returns "
            "latents)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX25AutoDecoderStep(AutoPipelineBlocks):
    """
    Auto decode block for LTX-2.5 that selects the decoder based on inputs.
       - `LTX25ConditionDecoderStep` when `base_token_count` is present, i.e. the denoised sequence carries appended
         keyframe / reference tokens (condition, in-context).
       - `LTX25DecoderStep` otherwise (text-to-video, image-to-video).

      Components:
          diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) video_processor (`VideoProcessor`) audio_vae
          (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          base_token_count (`int`, *optional*):
              Number of generated-video tokens, i.e. the sequence length before appended tokens.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*, defaults to 121):
              The number of frames in the generated video.
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
    """

    model_name = "ltx2.5"
    # Mirrors `LTX2AutoDecoderStep` on the same `base_token_count` trigger; only the video decoder differs.
    block_classes = [LTX25ConditionDecoderStep, LTX25DecoderStep]
    block_names = ["condition", "default"]
    block_trigger_inputs = ["base_token_count", None]

    @property
    def description(self):
        return (
            "Auto decode block for LTX-2.5 that selects the decoder based on inputs.\n"
            " - `LTX25ConditionDecoderStep` when `base_token_count` is present, i.e. the denoised sequence carries "
            "appended keyframe / reference tokens (condition, in-context).\n"
            " - `LTX25DecoderStep` otherwise (text-to-video, image-to-video)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX25AutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks for LTX-2.5 supporting text-to-video, image-to-video, condition-to-video and in-context (IC-LoRA)
    generation (joint video + audio). Identical to `LTX2AutoBlocks` except that the video decoder is
    `LTX2DiffusionVaeDecoderStep`, since the diffusion decoder is the native default from LTX-2.5 on. To decode with
    the convolutional VAE instead, swap the decode block: `blocks.sub_blocks["decode"] = LTX2AutoDecoderStep()`.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`
        - `condition`: requires `conditions`, `prompt`
        - `in_context`: requires `reference_conditions`, `num_frames`, `prompt`

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) transformer (`LTX2VideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`)
          audio_guider (`LTX2Guidance`) diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) vocoder (`LTX2Vocoder`)

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
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          image_crf (`int`, *optional*):
              H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the
              model was trained against. `None` (default) resolves from the text-encoder generation (33 through
              LTX-2.3, 18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when
              re-compression runs.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens the IC-LoRA
              adapter attends to.
          reference_downscale_factor (`int`, *optional*, defaults to 1):
              Ratio between the target and reference resolutions; 2 means the reference is preprocessed at half the
              target resolution. Spatial coordinates are scaled by this factor so the reference tokens land in the
              target coordinate space. Must match the factor the IC-LoRA was trained with.
          conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
              Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
              (default) leaves attention unmasked.
          conditioning_attention_mask (`Tensor`, *optional*):
              Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
              attention strength. Downsampled to the reference's latent grid and multiplied by
              `conditioning_attention_strength`.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          condition_latents (`list`, *optional*):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_latents (`Tensor`, *optional*):
              Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference conditions
              were supplied (`LTX2AutoReferenceEncoderStep` is skipped).
          reference_coords (`Tensor`, *optional*):
              RoPE coordinates for the reference tokens.
          reference_token_counts (`list`, *optional*):
              Per-reference token counts, in `reference_conditions` order.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          reference_cross_mask (`Tensor`, *optional*):
              Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          audio_latents (`Tensor`):
              Optional pre-encoded audio latents; random noise is used when not provided.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to True):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2.5"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2AutoVaeEncoderStep,
        LTX2AutoConditionEncoderStep,
        LTX2AutoReferenceEncoderStep,
        LTX2AutoCoreDenoiseStep,
        LTX25AutoDecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "duration",
        "vae_encoder",
        "condition_encoder",
        "reference_encoder",
        "denoise",
        "decode",
    ]
    # `num_frames` on `in_context` is a requirement, not a trigger: the in-context checkpoints ship without a
    # `duration_head`, so the workflow drops `LTX2AutoDurationStep` and `LTX2ConditionEncoderStep` raises if
    # `num_frames` is still `None`. Matches `LTX2InContextBlocks`, which omits the duration step outright.
    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
        "condition": {"conditions": True, "prompt": True},
        "in_context": {"reference_conditions": True, "num_frames": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Auto blocks for LTX-2.5 supporting text-to-video, image-to-video, condition-to-video and in-context "
            "(IC-LoRA) generation (joint video + audio). Identical to `LTX2AutoBlocks` except that the video decoder "
            "is `LTX2DiffusionVaeDecoderStep`, since the diffusion decoder is the native default from LTX-2.5 on. To "
            'decode with the convolutional VAE instead, swap the decode block: `blocks.sub_blocks["decode"] = '
            "LTX2AutoDecoderStep()`."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]
