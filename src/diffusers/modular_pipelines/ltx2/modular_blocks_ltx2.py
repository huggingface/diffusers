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

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    LTX2BuildVideoSelfAttentionMaskStep,
    LTX2ConditionPrepareAudioLatentsStep,
    LTX2ConditionPrepareCoordsStep,
    LTX2ConditionPrepareLatentsStep,
    LTX2ConditionSetTimestepsStep,
    LTX2Image2VideoPrepareLatentsStep,
    LTX2InContextPrepareLatentsStep,
    LTX2PrepareAudioLatentsStep,
    LTX2PrepareCoordsStep,
    LTX2PrepareLatentsStep,
    LTX2SetTimestepsStep,
    LTX2TextInputStep,
)
from .decoders import (
    LTX2AudioDecoderStep,
    LTX2DiffusionVaeDecoderStep,
    LTX2TrimConditionTokensStep,
    LTX2VaeDecoderStep,
)
from .denoise import LTX2ConditionDenoiseStep, LTX2DenoiseStep, LTX2Image2VideoDenoiseStep
from .encoders import (
    LTX2ConditionEncoderStep,
    LTX2ConditionPromptEnhancerStep,
    LTX2DurationStep,
    LTX2ImageToVideoPromptEnhancerStep,
    LTX2PromptEnhancerStep,
    LTX2ReferenceEncoderStep,
    LTX2TextConnectorStep,
    LTX2TextEncoderStep,
    LTX2VaeEncoderStep,
)


logger = logging.get_logger(__name__)


# auto_docstring
class LTX2AutoPromptEnhancerStep(ConditionalPipelineBlocks):
    """
    Conditional prompt-enhancer step, run only when `enable_prompt_enhancement` is truthy.
       - `LTX2ConditionPromptEnhancerStep` when `conditions` are provided (condition-to-video, in-context); grounds the
         rewrite in the first `PIL.Image.Image` frame found in `conditions`, falling back to a text-only rewrite when
         there is none.
       - `LTX2ImageToVideoPromptEnhancerStep` when an `image` is provided (image-to-video).
       - `LTX2PromptEnhancerStep` otherwise (text-to-video).
       - Skipped when `enable_prompt_enhancement` is falsy / not provided.

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`)

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

      Outputs:
          prompt (`list`):
              The prompt(s) after prompt-enhancer rewriting.
    """

    model_name = "ltx2"
    block_classes = [LTX2ConditionPromptEnhancerStep, LTX2ImageToVideoPromptEnhancerStep, LTX2PromptEnhancerStep]
    block_names = ["condition", "image2video", "text2video"]
    block_trigger_inputs = ["conditions", "image", "enable_prompt_enhancement"]

    def select_block(self, conditions=None, image=None, enable_prompt_enhancement=False) -> str | None:
        # `conditions` is checked before `image`: the condition and in-context workflows place their reference
        # frames in `conditions` and never take a raw `image`.
        if not enable_prompt_enhancement:
            return None
        if conditions is not None:
            return "condition"
        return "image2video" if image is not None else "text2video"

    @property
    def description(self):
        return (
            "Conditional prompt-enhancer step, run only when `enable_prompt_enhancement` is truthy.\n"
            " - `LTX2ConditionPromptEnhancerStep` when `conditions` are provided (condition-to-video, in-context); "
            "grounds the rewrite in the first `PIL.Image.Image` frame found in `conditions`, falling back to a "
            "text-only rewrite when there is none.\n"
            " - `LTX2ImageToVideoPromptEnhancerStep` when an `image` is provided (image-to-video).\n"
            " - `LTX2PromptEnhancerStep` otherwise (text-to-video).\n"
            " - Skipped when `enable_prompt_enhancement` is falsy / not provided."
        )


# auto_docstring
class LTX2TextConditioningStep(SequentialPipelineBlocks):
    """
    Text-conditioning stage for LTX-2.X: encodes the prompt(s), then runs the text connectors to produce the
    video/audio-branch connector embeddings the denoiser consumes. Outputs stay at one row per prompt -- the denoise
    stage expands them by `num_videos_per_prompt` -- so they can be reused across denoise runs.

      Components:
          text_encoder (`PreTrainedModel`) tokenizer (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the prompt.
          prompt_attention_mask (`Tensor`):
              Binary attention mask for `prompt_embeds`.
          negative_prompt_embeds (`Tensor`):
              Packed per-layer Gemma hidden states for the negative prompt.
          negative_prompt_attention_mask (`Tensor`):
              Binary attention mask for `negative_prompt_embeds`.
          batch_size (`int`):
              The number of prompts being denoised (before per-prompt expansion).
          dtype (`dtype`):
              The dtype of the prompt embeddings.
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
    """

    model_name = "ltx2"
    block_classes = [LTX2TextEncoderStep, LTX2TextConnectorStep]
    block_names = ["text_encoder", "connectors"]

    @property
    def description(self):
        return (
            "Text-conditioning stage for LTX-2.X: encodes the prompt(s), then runs the text connectors to produce "
            "the video/audio-branch connector embeddings the denoiser consumes. Outputs stay at one row per prompt "
            "-- the denoise stage expands them by `num_videos_per_prompt` -- so they can be reused across denoise "
            "runs."
        )


# auto_docstring
class LTX2AutoDurationStep(ConditionalPipelineBlocks):
    """
    Conditional duration-prediction step, run only when `num_frames` is omitted.
       - `LTX2DurationStep` predicts `num_frames` from the connector text conditioning via the `duration_head`.
       - Skipped when `num_frames` is supplied as an integer.

      Components:
          duration_head (`LTX2DurationHead`)

      Inputs:
          min_seconds (`float`, *optional*, defaults to 1.0):
              Lower bound on the auto-predicted duration.
          max_seconds (`float`, *optional*, defaults to 20.0):
              Upper bound on the auto-predicted duration. Must be strictly greater than `min_seconds`.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning from the connector (positive prompt).
          connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning from the connector (positive prompt).
          batch_size (`int`, *optional*):
              The number of prompts being denoised, used to expand conditioning per prompt.

      Outputs:
          num_frames (`int`):
              The predicted number of frames to generate.
    """

    model_name = "ltx2"
    block_classes = [LTX2DurationStep]
    block_names = ["duration"]
    block_trigger_inputs = ["num_frames"]

    def select_block(self, num_frames=None) -> str | None:
        return "duration" if num_frames is None else None

    @property
    def description(self):
        return (
            "Conditional duration-prediction step, run only when `num_frames` is omitted.\n"
            " - `LTX2DurationStep` predicts `num_frames` from the connector text conditioning via the `duration_head`.\n"
            " - Skipped when `num_frames` is supplied as an integer."
        )


# auto_docstring
class LTX2AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the reference `image` into latents for image-to-video.
       - `LTX2VaeEncoderStep` runs when `image` is provided.
       - Skipped otherwise.

      Components:
          vae (`AutoencoderKLLTX2Video`) text_encoder (`PreTrainedModel`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          image_crf (`int`, *optional*):
              H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the
              model was trained against. `None` (default) resolves from the text-encoder generation (33 through
              LTX-2.3, 18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when
              re-compression runs.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          image_latents (`Tensor`):
              Normalized image latents (a single latent frame) for image-to-video conditioning.
    """

    model_name = "ltx2"
    block_classes = [LTX2VaeEncoderStep]
    block_names = ["vae_encoder"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the reference `image` into latents for image-to-video.\n"
            " - `LTX2VaeEncoderStep` runs when `image` is provided.\n"
            " - Skipped otherwise."
        )


# auto_docstring
class LTX2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (text-to-video) that expands the text conditioning by `num_videos_per_prompt`, prepares video/audio
    latents and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

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
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Interpolation factor between random noise and any provided latents. `None` (default) resolves to 0.0,
              which keeps the provided latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
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
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextInputStep,
        LTX2SetTimestepsStep,
        LTX2PrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordsStep,
        LTX2DenoiseStep,
    ]
    block_names = [
        "input",
        "set_timesteps",
        "prepare_latents",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Denoise block (text-to-video) that expands the text conditioning by `num_videos_per_prompt`, prepares "
            "video/audio latents and runs the joint denoising loop."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX2Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (image-to-video) that expands the text conditioning by `num_videos_per_prompt`, adds image
    conditioning and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

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
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Interpolation factor between random noise and any provided latents. `None` (default) resolves to 0.0,
              which keeps the provided latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          image_latents (`Tensor`):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
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
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2TextInputStep,
        LTX2SetTimestepsStep,
        LTX2PrepareLatentsStep,
        LTX2Image2VideoPrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordsStep,
        LTX2Image2VideoDenoiseStep,
    ]
    block_names = [
        "input",
        "set_timesteps",
        "prepare_latents",
        "prepare_i2v_latents",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Denoise block (image-to-video) that expands the text conditioning by `num_videos_per_prompt`, adds "
            "image conditioning and runs the joint denoising loop."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX2ConditionCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (condition-to-video) that expands the text conditioning by `num_videos_per_prompt`, applies the frame
    conditions to the video latents and runs the joint denoising loop.

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
          condition_latents (`list`):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
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
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    # NOTE: prepare-latents runs *before* set-timesteps here, unlike the text-to-video / image-to-video steps. The
    # resolution-aware shift `mu` is computed from the packed latent sequence length, which for condition workflows
    # includes the appended keyframe tokens, so the latents have to exist first. This mirrors `LTX2ConditionPipeline`
    # (its section 4 runs before section 5).
    block_classes = [
        LTX2TextInputStep,
        LTX2ConditionPrepareLatentsStep,
        LTX2ConditionSetTimestepsStep,
        LTX2ConditionPrepareAudioLatentsStep,
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "prepare_audio_latents", "prepare_coords", "denoise"]

    @property
    def description(self):
        return (
            "Denoise block (condition-to-video) that expands the text conditioning by `num_videos_per_prompt`, "
            "applies the frame conditions to the video latents and runs the joint denoising loop."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX2AutoConditionEncoderStep(ConditionalPipelineBlocks):
    """
    Conditional condition-encoder step, run only for the condition and in-context workflows.
       - `LTX2ConditionEncoderStep` VAE-encodes the frame `conditions` into per-condition latents.
       - Also runs when only `reference_conditions` are supplied, emitting empty per-condition lists for
         `LTX2InContextPrepareLatentsStep`.
       - Skipped for text-to-video and image-to-video.

      Components:
          vae (`AutoencoderKLLTX2Video`) text_encoder (`PreTrainedModel`)

      Inputs:
          conditions (`list`, *optional*):
              `LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
              generated video.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          condition_latents (`list`):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp the temporal extent of single-frame keyframe
              coordinates.
    """

    model_name = "ltx2"
    block_classes = [LTX2ConditionEncoderStep]
    block_names = ["condition_encoder"]
    block_trigger_inputs = ["conditions", "reference_conditions"]

    def select_block(self, conditions=None, reference_conditions=None) -> str | None:
        # Also runs for a reference-only in-context request: `LTX2InContextPrepareLatentsStep` requires the
        # per-condition lists, and this step emits them empty when there are no frame `conditions`.
        if conditions is not None or reference_conditions:
            return "condition_encoder"
        return None

    @property
    def description(self):
        return (
            "Conditional condition-encoder step, run only for the condition and in-context workflows.\n"
            " - `LTX2ConditionEncoderStep` VAE-encodes the frame `conditions` into per-condition latents.\n"
            " - Also runs when only `reference_conditions` are supplied, emitting empty per-condition lists for "
            "`LTX2InContextPrepareLatentsStep`.\n"
            " - Skipped for text-to-video and image-to-video."
        )


# auto_docstring
class LTX2AutoReferenceEncoderStep(ConditionalPipelineBlocks):
    """
    Conditional reference-encoder step, run only when `reference_conditions` are supplied.
       - `LTX2ReferenceEncoderStep` encodes the reference videos into extra latent tokens.
       - Skipped otherwise, for IC-LoRAs that take no reference video.

      Components:
          vae (`AutoencoderKLLTX2Video`) transformer (`LTX2VideoTransformer3DModel`) video_processor (`VideoProcessor`)

      Inputs:
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
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          reference_latents (`Tensor`):
              Packed reference tokens of shape [1, total_reference_tokens, C].
          reference_coords (`Tensor`):
              RoPE coordinates for the reference tokens, of shape [1, 3, total_reference_tokens, 2].
          reference_token_counts (`list`):
              Per-reference token counts, in `reference_conditions` order.
          reference_cross_mask (`Tensor`):
              Per-reference-token noisy<->reference attention strengths of shape [1, total_reference_tokens], or `None`
              when attention is left unmasked.
    """

    model_name = "ltx2"
    block_classes = [LTX2ReferenceEncoderStep]
    block_names = ["reference_encoder"]
    block_trigger_inputs = ["reference_conditions"]

    def select_block(self, reference_conditions=None) -> str | None:
        # Falsy covers both `None` and an empty list. `LTX2InContextPipeline` likewise treats a missing reference
        # as "no reference tokens" rather than an error: IC-LoRAs that carry their behavior in the adapter weights
        # (camera control, style, ...) take no reference video at all.
        return "reference_encoder" if reference_conditions else None

    @property
    def description(self):
        return (
            "Conditional reference-encoder step, run only when `reference_conditions` are supplied.\n"
            " - `LTX2ReferenceEncoderStep` encodes the reference videos into extra latent tokens.\n"
            " - Skipped otherwise, for IC-LoRAs that take no reference video."
        )


# auto_docstring
class LTX2AutoBuildVideoSelfAttentionMaskStep(ConditionalPipelineBlocks):
    """
    Conditional video self-attention mask step, run only when the reference tokens carry a per-token attention
    strength.
       - `LTX2BuildVideoSelfAttentionMaskStep` when `conditioning_attention_strength < 1.0` or a
         `conditioning_attention_mask` is supplied.
       - Skipped otherwise, leaving attention unmasked.

      Inputs:
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          base_token_count (`int`, *optional*):
              Number of generated-video tokens, i.e. the sequence length before appended tokens.
          num_ref_tokens (`int`, *optional*):
              Number of reference tokens, which sit at the very end of the sequence.
          reference_cross_mask (`Tensor`, *optional*):
              Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
          reference_token_counts (`list`, *optional*):
              Per-reference token counts, used to split `reference_cross_mask` into attention groups.

      Outputs:
          video_self_attention_mask (`Tensor`):
              Multiplicative self-attention mask of shape [B, S, S] with values in [0, 1].
    """

    model_name = "ltx2"
    block_classes = [LTX2BuildVideoSelfAttentionMaskStep]
    block_names = ["attention_mask"]
    block_trigger_inputs = [
        "reference_conditions",
        "conditioning_attention_strength",
        "conditioning_attention_mask",
    ]

    def select_block(
        self, reference_conditions=None, conditioning_attention_strength=None, conditioning_attention_mask=None
    ) -> str | None:
        # The mask only ever governs noisy<->reference attention, so it is meaningless without reference tokens.
        # Beyond that, an all-ones mask at full strength is the same as no mask, so only build one when it can
        # actually bite -- the same `mask_needed` test `LTX2ReferenceEncoderStep` uses to decide whether to emit
        # `reference_cross_mask` at all. `conditioning_attention_strength` defaults to `None` rather than its
        # input default of 1.0 because `get_execution_blocks` passes every unset trigger input as an explicit
        # `None`; `None` and 1.0 both mean "leave attention unmasked".
        if not reference_conditions:
            return None
        if conditioning_attention_mask is not None:
            return "attention_mask"
        if conditioning_attention_strength is not None and conditioning_attention_strength < 1.0:
            return "attention_mask"
        return None

    @property
    def description(self):
        return (
            "Conditional video self-attention mask step, run only when the reference tokens carry a per-token "
            "attention strength.\n"
            " - `LTX2BuildVideoSelfAttentionMaskStep` when `conditioning_attention_strength < 1.0` or a "
            "`conditioning_attention_mask` is supplied.\n"
            " - Skipped otherwise, leaving attention unmasked."
        )


# auto_docstring
class LTX2InContextCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (in-context) that expands the text conditioning by `num_videos_per_prompt`, folds the frame
    conditions and the IC-LoRA reference tokens into one latent sequence and runs the joint denoising loop. Reuses the
    condition denoise step unchanged: reference tokens are pinned by the same x0 blend as frame conditions, matching
    the reference implementation's uniform treatment of both.

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
          condition_latents (`list`):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them); only their `strength` is read here. Omit for IC-LoRAs that
              carry their behavior in the adapter weights and take no reference video.
          reference_latents (`Tensor`, *optional*):
              Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference conditions
              were supplied (`LTX2AutoReferenceEncoderStep` is skipped).
          reference_coords (`Tensor`, *optional*):
              RoPE coordinates for the reference tokens.
          reference_token_counts (`list`, *optional*):
              Per-reference token counts, in `reference_conditions` order.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
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
          reference_cross_mask (`Tensor`, *optional*):
              Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
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
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    # Same ordering rationale as `LTX2ConditionCoreDenoiseStep`: prepare-latents precedes set-timesteps because
    # `mu` is read off the packed sequence length, which here includes both keyframe and reference tokens.
    block_classes = [
        LTX2TextInputStep,
        LTX2InContextPrepareLatentsStep,
        LTX2AutoBuildVideoSelfAttentionMaskStep,
        LTX2ConditionSetTimestepsStep,
        LTX2ConditionPrepareAudioLatentsStep,
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "attention_mask",
        "set_timesteps",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Denoise block (in-context) that expands the text conditioning by `num_videos_per_prompt`, folds the "
            "frame conditions and the IC-LoRA reference tokens into one latent sequence and runs the joint denoising "
            "loop. Reuses the condition denoise step unchanged: "
            "reference tokens are pinned by the same x0 blend as frame conditions, matching the reference "
            "implementation's uniform treatment of both."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX2AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Auto denoise block that selects the workflow based on inputs.
       - `LTX2InContextCoreDenoiseStep` when `reference_conditions` are provided (in-context / IC-LoRA).
       - `LTX2ConditionCoreDenoiseStep` when `condition_latents` are provided (condition-to-video).
       - `LTX2Image2VideoCoreDenoiseStep` when `image_latents` is provided.
       - `LTX2CoreDenoiseStep` otherwise (text-to-video).

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
          condition_latents (`list`, *optional*):
              Per-condition normalized VAE latents of shape [1, C, F, H, W].
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them); only their `strength` is read here. Omit for IC-LoRAs that
              carry their behavior in the adapter weights and take no reference video.
          reference_latents (`Tensor`, *optional*):
              Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference conditions
              were supplied (`LTX2AutoReferenceEncoderStep` is skipped).
          reference_coords (`Tensor`, *optional*):
              RoPE coordinates for the reference tokens.
          reference_token_counts (`list`, *optional*):
              Per-reference token counts, in `reference_conditions` order.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
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
          reference_cross_mask (`Tensor`, *optional*):
              Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
          num_inference_steps (`int`):
              The number of denoising steps.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          audio_latents (`Tensor`):
              Optional pre-encoded audio latents; random noise is used when not provided.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          use_cross_timestep (`bool`, *optional*, defaults to True):
              Whether to condition the transformer on a separate per-token cross timestep (LTX-2.3+).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2InContextCoreDenoiseStep,
        LTX2ConditionCoreDenoiseStep,
        LTX2Image2VideoCoreDenoiseStep,
        LTX2CoreDenoiseStep,
    ]
    block_names = ["in_context", "condition", "image2video", "text2video"]
    block_trigger_inputs = ["reference_conditions", "condition_latents", "image_latents"]
    default_block_name = "text2video"

    def select_block(self, reference_conditions=None, condition_latents=None, image_latents=None) -> str | None:
        # An IC-LoRA that takes no reference video lands on the condition branch, which is the right answer rather
        # than a fallback: `LTX2InContextPrepareLatentsStep` with no reference tokens does exactly what
        # `LTX2ConditionPrepareLatentsStep` does, and the extra `num_ref_tokens` it emits is only read by the
        # attention-mask step, which is skipped in that case anyway.
        if reference_conditions:
            return "in_context"
        if condition_latents is not None:
            return "condition"
        if image_latents is not None:
            return "image2video"
        return "text2video"

    @property
    def description(self):
        return (
            "Auto denoise block that selects the workflow based on inputs.\n"
            " - `LTX2InContextCoreDenoiseStep` when `reference_conditions` are provided (in-context / IC-LoRA).\n"
            " - `LTX2ConditionCoreDenoiseStep` when `condition_latents` are provided (condition-to-video).\n"
            " - `LTX2Image2VideoCoreDenoiseStep` when `image_latents` is provided.\n"
            " - `LTX2CoreDenoiseStep` otherwise (text-to-video)."
        )


# auto_docstring
class LTX2DecoderStep(SequentialPipelineBlocks):
    """
    Decode stage: VAE-decodes the video latents and vocodes the audio latents (or returns latents).

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLLTX2Audio`)
          vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
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

    model_name = "ltx2"
    block_classes = [LTX2VaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["video_decode", "audio_decode"]

    @property
    def description(self):
        return "Decode stage: VAE-decodes the video latents and vocodes the audio latents (or returns latents)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2ConditionDecoderStep(SequentialPipelineBlocks):
    """
    Decode stage for condition workflows: drops the appended keyframe-condition tokens, then VAE-decodes the video
    latents and vocodes the audio latents (or returns latents).

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLLTX2Audio`)
          vocoder (`LTX2Vocoder`)

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
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
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

    model_name = "ltx2"
    block_classes = [LTX2TrimConditionTokensStep, LTX2VaeDecoderStep, LTX2AudioDecoderStep]
    block_names = ["trim_condition_tokens", "video_decode", "audio_decode"]

    @property
    def description(self):
        return (
            "Decode stage for condition workflows: drops the appended keyframe-condition tokens, then VAE-decodes "
            "the video latents and vocodes the audio latents (or returns latents)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2AutoDecoderStep(AutoPipelineBlocks):
    """
    Auto decode block that selects the decoder based on inputs.
       - `LTX2ConditionDecoderStep` when `base_token_count` is present, i.e. the denoised sequence carries appended
         keyframe / reference tokens (condition, in-context).
       - `LTX2DecoderStep` otherwise (text-to-video, image-to-video).

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) audio_vae (`AutoencoderKLLTX2Audio`)
          vocoder (`LTX2Vocoder`)

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
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`, *optional*, defaults to 1):
              Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can
              be generated in input step.
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

    model_name = "ltx2"
    # `base_token_count` is emitted only by the condition / in-context prepare-latents steps, so it is exactly the
    # signal for "the sequence carries appended tokens that have to come off before decoding".
    block_classes = [LTX2ConditionDecoderStep, LTX2DecoderStep]
    block_names = ["condition", "default"]
    block_trigger_inputs = ["base_token_count", None]

    @property
    def description(self):
        return (
            "Auto decode block that selects the decoder based on inputs.\n"
            " - `LTX2ConditionDecoderStep` when `base_token_count` is present, i.e. the denoised sequence carries "
            "appended keyframe / reference tokens (condition, in-context).\n"
            " - `LTX2DecoderStep` otherwise (text-to-video, image-to-video)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 text-to-video (joint video + audio).

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) vocoder (`LTX2Vocoder`)

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
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Interpolation factor between random noise and any provided latents. `None` (default) resolves to 0.0,
              which keeps the provided latents.
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
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2CoreDenoiseStep,
        LTX2DecoderStep,
    ]
    block_names = ["prompt_enhancer", "text_encoder", "duration", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX-2 text-to-video (joint video + audio)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2ImageToVideoBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 image-to-video (joint video + audio).

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`LTX2VideoTransformer3DModel`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`)
          audio_guider (`LTX2Guidance`) vocoder (`LTX2Vocoder`)

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
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          num_inference_steps (`int`, *optional*, defaults to 30):
              The number of denoising steps.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Interpolation factor between random noise and any provided latents. `None` (default) resolves to 0.0,
              which keeps the provided latents.
          image_latents (`Tensor`):
              VAE-encoded reference-image latents used for image-to-video conditioning.
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
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2AutoVaeEncoderStep,
        LTX2Image2VideoCoreDenoiseStep,
        LTX2DecoderStep,
    ]
    block_names = ["prompt_enhancer", "text_encoder", "duration", "vae_encoder", "denoise", "decode"]

    @property
    def description(self):
        return "Modular pipeline blocks for LTX-2 image-to-video (joint video + audio)."

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2ConditionBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 condition-to-video (joint video + audio): image or video conditions placed at
    arbitrary latent frame indices of the generated video.

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
          (`AutoencoderKLLTX2Video`) transformer (`LTX2VideoTransformer3DModel`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider
          (`LTX2Guidance`) video_processor (`VideoProcessor`) vocoder (`LTX2Vocoder`)

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
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to auto-predict via the `duration_head` (see
              `LTX2AutoDurationStep`).
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
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
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2ConditionEncoderStep,
        LTX2ConditionCoreDenoiseStep,
        LTX2ConditionDecoderStep,
    ]
    block_names = ["prompt_enhancer", "text_encoder", "duration", "condition_encoder", "denoise", "decode"]

    @property
    def description(self):
        return (
            "Modular pipeline blocks for LTX-2 condition-to-video (joint video + audio): image or video conditions "
            "placed at arbitrary latent frame indices of the generated video."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2InContextBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for LTX-2 in-context (IC-LoRA) video generation (joint video + audio). Reference videos are
    encoded into extra latent tokens the IC-LoRA adapter attends to; frame `conditions` may be used alongside them.
    Load the IC-LoRA on the pipeline before running. `reference_conditions` is optional: IC-LoRAs that carry their
    behavior in the adapter weights (camera control, style, ...) take no reference video, matching
    `LTX2InContextPipeline`.

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) vae (`AutoencoderKLLTX2Video`) transformer
          (`LTX2VideoTransformer3DModel`) video_processor (`VideoProcessor`) scheduler
          (`FlowMatchEulerDiscreteScheduler`) audio_vae (`AutoencoderKLLTX2Audio`) guider (`LTX2Guidance`) audio_guider
          (`LTX2Guidance`) vocoder (`LTX2Vocoder`)

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
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
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
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          reference_latents (`Tensor`, *optional*):
              Packed reference tokens of shape [1, total_reference_tokens, C], or `None` when no reference conditions
              were supplied (`LTX2AutoReferenceEncoderStep` is skipped).
          reference_coords (`Tensor`, *optional*):
              RoPE coordinates for the reference tokens.
          reference_token_counts (`list`, *optional*):
              Per-reference token counts, in `reference_conditions` order.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          reference_cross_mask (`Tensor`, *optional*):
              Per-reference-token noisy<->reference attention strengths of shape [1, num_ref_tokens].
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
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    # No `LTX2AutoDurationStep`: the in-context pipeline ships without a `duration_head`, matching
    # `LTX2InContextPipeline`. `num_frames` is therefore required -- `LTX2ConditionEncoderStep` raises if it is
    # still `None` by the time the conditions are encoded. `reference_conditions` is optional; when omitted the
    # reference encoder and the attention-mask step are both skipped and the graph reduces to the condition path.
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2ConditionEncoderStep,
        LTX2AutoReferenceEncoderStep,
        LTX2InContextCoreDenoiseStep,
        LTX2ConditionDecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "condition_encoder",
        "reference_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Modular pipeline blocks for LTX-2 in-context (IC-LoRA) video generation (joint video + audio). "
            "Reference videos are encoded into extra latent tokens the IC-LoRA adapter attends to; frame "
            "`conditions` may be used alongside them. Load the IC-LoRA on the pipeline before running. "
            "`reference_conditions` is optional: IC-LoRAs that carry their behavior in the adapter weights "
            "(camera control, style, ...) take no reference video, matching `LTX2InContextPipeline`."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX2AutoBlocks(SequentialPipelineBlocks):
    """
    Auto blocks for LTX-2 supporting text-to-video, image-to-video, condition-to-video and in-context (IC-LoRA)
    generation (joint video + audio).

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
          audio_guider (`LTX2Guidance`) vocoder (`LTX2Vocoder`)

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
          decode_timestep (`None`, *optional*, defaults to 0.0):
              The timestep at which the VAE decodes the final latents.
          decode_noise_scale (`None`, *optional*):
              Noise interpolation factor applied to the latents at the decode timestep.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2"
    block_classes = [
        LTX2AutoPromptEnhancerStep,
        LTX2TextConditioningStep,
        LTX2AutoDurationStep,
        LTX2AutoVaeEncoderStep,
        LTX2AutoConditionEncoderStep,
        LTX2AutoReferenceEncoderStep,
        LTX2AutoCoreDenoiseStep,
        LTX2AutoDecoderStep,
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
            "Auto blocks for LTX-2 supporting text-to-video, image-to-video, condition-to-video and in-context "
            "(IC-LoRA) generation (joint video + audio)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


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

    model_name = "ltx2"
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

    model_name = "ltx2"
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

    model_name = "ltx2"
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


class LTX25AutoBlocks(LTX2AutoBlocks):
    """
    Auto blocks for LTX-2.5, supporting text-to-video, image-to-video, condition-to-video and in-context (IC-LoRA)
    generation (joint video + audio).

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`
        - `condition`: requires `conditions`, `prompt`
        - `in_context`: requires `reference_conditions`, `num_frames`, `prompt`
    """

    model_name = "ltx2"
    # Both lists are restated rather than inherited: `block_classes` and `block_names` are positional, so a subclass
    # that overrides one and inherits the other silently desynchronizes when the parent's graph changes. Keep these in
    # step with `LTX2AutoBlocks`; only the `decode` entry differs.
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

    @property
    def description(self):
        return (
            "Auto blocks for LTX-2.5 supporting text-to-video, image-to-video, condition-to-video and in-context "
            "(IC-LoRA) generation (joint video + audio). Identical to `LTX2AutoBlocks` except that the video decoder "
            "is `LTX2DiffusionVaeDecoderStep`, since the diffusion decoder is the native default from LTX-2.5 on. To "
            'decode with the convolutional VAE instead, swap the decode block: `blocks.sub_blocks["decode"] = '
            "LTX2AutoDecoderStep()`."
        )
