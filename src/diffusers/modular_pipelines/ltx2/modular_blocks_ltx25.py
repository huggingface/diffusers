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

from ...pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InputParam, InsertableDict, OutputParam
from .before_denoise import (
    LTX2ConditionPrepareAudioLatentsStep,
    LTX2ConditionPrepareCoordsStep,
    LTX2ConditionPrepareLatentsStep,
    LTX2ConditionSetTimestepsStep,
    LTX2ConditionStage2PrepareLatentsStep,
    LTX2Image2VideoPrepareLatentsStep,
    LTX2InContextPrepareLatentsStep,
    LTX2LatentUpsampleStep,
    LTX2PrepareAudioLatentsStep,
    LTX2PrepareCoordsStep,
    LTX2PrepareLatentsStep,
    LTX2SetTimestepsStep,
    LTX2Stage2PrepareAudioLatentsStep,
    LTX2Stage2PrepareLatentsStep,
    LTX2TextInputStep,
)
from .decoders import (
    LTX2AudioDecoderStep,
    LTX2DiffusionVaeDecoderStep,
    LTX2TrimConditionTokensStep,
    LTX2UnpackLatentsStep,
)
from .denoise import (
    LTX2ConditionDenoiseStep,
    LTX2DenoiseStep,
    LTX2Image2VideoDenoiseStep,
)
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
class LTX25AutoPromptEnhancerStep(ConditionalPipelineBlocks):
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
          enable_prompt_enhancement (`bool`, *optional*, defaults to False):
              Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.

      Outputs:
          prompt (`list`):
              The prompt(s) after prompt-enhancer rewriting.
    """

    model_name = "ltx2.5"
    block_classes = [LTX2ConditionPromptEnhancerStep, LTX2ImageToVideoPromptEnhancerStep, LTX2PromptEnhancerStep]
    block_names = ["condition", "image2video", "text2video"]
    block_trigger_inputs = ["conditions", "image", "enable_prompt_enhancement"]

    @property
    def inputs(self):
        # The trigger belongs to this wrapper, not to the enhancer steps: each of them always enhances.
        inputs = super().inputs
        inputs.append(
            InputParam(
                "enable_prompt_enhancement",
                type_hint=bool,
                default=False,
                description="Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.",
            )
        )
        return inputs

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
class LTX25TextConditioningStep(SequentialPipelineBlocks):
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
              Packed per-layer Gemma hidden states for the negative prompt, `None` when not encoded.
          negative_prompt_attention_mask (`Tensor`):
              Binary attention mask for `negative_prompt_embeds`, `None` when not encoded.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond).
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond).
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond).
          negative_connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (uncond), `None` when no negative prompt was encoded.
          negative_connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (uncond), `None` when no negative prompt was encoded.
          negative_connector_attention_mask (`Tensor`):
              Binary text attention mask (uncond), `None` when no negative prompt was encoded.
    """

    model_name = "ltx2.5"
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
class LTX25AutoDurationStep(ConditionalPipelineBlocks):
    """
    Conditional duration-prediction step, run only when `num_frames` is omitted.
       - `LTX2DurationStep` predicts `num_frames` from the connector text conditioning via the `duration_head`.
       - Skipped when `num_frames` is supplied as an integer.

      Components:
          duration_head (`LTX2DurationHead`)

      Inputs:
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to have this step predict it with the `duration_head`;
              the denoise blocks then take the predicted count.
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

      Outputs:
          num_frames (`int`):
              The predicted number of frames to generate.
    """

    model_name = "ltx2.5"
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
class LTX25AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the reference `image` into latents for image-to-video.
       - `LTX2VaeEncoderStep` runs when `image` is provided.
       - Skipped otherwise.

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          image_crf (`int`, *optional*):
              H.264 CRF used to re-compress the conditioning `image` before VAE encode, matching the compression the
              model was trained against. `None` (default) uses the pipeline's `default_image_crf` (33 through LTX-2.3,
              18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when re-compression runs.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          image_latents (`Tensor`):
              Image latents for image-to-video conditioning: a single latent frame of shape [B, C, 1, H, W]
              (normalized, not packed).
    """

    model_name = "ltx2.5"
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
class LTX25AutoConditionEncoderStep(ConditionalPipelineBlocks):
    """
    Conditional condition-encoder step, run only for the condition and in-context workflows.
       - `LTX2ConditionEncoderStep` VAE-encodes the frame `conditions` into per-condition latents.
       - Also runs when only `reference_conditions` are supplied, emitting empty per-condition lists for
         `LTX2InContextPrepareLatentsStep`.
       - Skipped for text-to-video and image-to-video.

      Components:
          vae (`AutoencoderKLLTX2Video`)

      Inputs:
          conditions (`list`, *optional*):
              `LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
              generated video.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          condition_latents (`list`):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp the temporal extent of single-frame keyframe
              coordinates.
    """

    model_name = "ltx2.5"
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
class LTX25AutoReferenceEncoderStep(ConditionalPipelineBlocks):
    """
    Conditional reference-encoder step, run only when `reference_conditions` are supplied.
       - `LTX2ReferenceEncoderStep` encodes the reference videos into extra latent tokens.
       - Skipped otherwise, for IC-LoRAs that take no reference video.

      Components:
          vae (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`)

      Inputs:
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens the IC-LoRA
              adapter attends to.
          reference_downscale_factor (`int`, *optional*, defaults to 1):
              Ratio between the target and reference resolutions; 2 means the reference is preprocessed at half the
              target resolution. Must match the factor the IC-LoRA was trained with.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`, *optional*):
              The number of frames in the generated video.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          reference_latents (`list`):
              Per-reference VAE latents of shape [1, C, F, H, W] (normalized, not packed), in `reference_conditions`
              order.
    """

    model_name = "ltx2.5"
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


LTX25_T2V_BLOCKS = InsertableDict(
    [
        ("set_timesteps", LTX2SetTimestepsStep(sigmas_default=DISTILLED_SIGMA_VALUES)),
        ("prepare_latents", LTX2PrepareLatentsStep()),
        ("prepare_audio_latents", LTX2PrepareAudioLatentsStep()),
        ("prepare_coords", LTX2PrepareCoordsStep()),
        ("denoise", LTX2DenoiseStep()),
        ("unpack", LTX2UnpackLatentsStep()),
    ]
)


# auto_docstring
class LTX25CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (text-to-video) that expands the text conditioning by `num_videos_per_prompt`, prepares video/audio
    latents and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`):
              The number of frames in the generated video.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    block_classes = LTX25_T2V_BLOCKS.values()
    block_names = LTX25_T2V_BLOCKS.keys()

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


LTX25_T2V_STAGE_2_BLOCKS = InsertableDict(
    [
        (
            "prepare_latents",
            LTX2Stage2PrepareLatentsStep(sigmas_name="stage_2_sigmas", sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES),
        ),
        (
            "set_timesteps",
            LTX2SetTimestepsStep(
                sigmas_name="stage_2_sigmas",
                timesteps_name="stage_2_timesteps",
                sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES,
            ),
        ),
        (
            "prepare_audio_latents",
            LTX2Stage2PrepareAudioLatentsStep(
                sigmas_name="stage_2_sigmas", sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES
            ),
        ),
        ("prepare_coords", LTX2PrepareCoordsStep()),
        ("denoise", LTX2DenoiseStep()),
        ("unpack", LTX2UnpackLatentsStep()),
    ]
)


# auto_docstring
class LTX25Stage2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (text-to-video, second pass) that expands the text conditioning by `num_videos_per_prompt`, re-noises
    the supplied video/audio latents to `noise_scale` and runs the joint denoising loop over them on `stage_2_sigmas`
    (the LTX-2 stage-2 distilled schedule by default) -- the refinement pass of the two-stage recipe, or any run that
    starts from existing latents.

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          latents (`Tensor`):
              Video latents to refine, of shape [B, C, F, H, W] (normalized, not packed).
          noise_scale (`float`, *optional*):
              Noise level the latents are re-noised to before the pass. `None` (default) resolves to `sigmas[0]` when
              custom `sigmas` are supplied, else 1.0.
          stage_2_sigmas (`list`, *optional*, defaults to [0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          stage_2_timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          audio_latents (`Tensor`):
              Audio latents to refine, of shape [B, C, L, M] (normalized, not packed).
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    block_classes = LTX25_T2V_STAGE_2_BLOCKS.values()
    block_names = LTX25_T2V_STAGE_2_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Denoise block (text-to-video, second pass) that expands the text conditioning by "
            "`num_videos_per_prompt`, re-noises the supplied video/audio latents to `noise_scale` and runs the joint "
            "denoising loop over them on `stage_2_sigmas` (the LTX-2 stage-2 distilled schedule by default) -- the "
            "refinement pass of the two-stage recipe, or any run that starts from existing latents."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX25Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (image-to-video) that expands the text conditioning by `num_videos_per_prompt`, adds image
    conditioning and runs the joint denoising loop.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`LTX2VideoTransformer3DModel`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`):
              The number of frames in the generated video.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          image_latents (`Tensor`):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    block_classes = [
        LTX2SetTimestepsStep(sigmas_default=DISTILLED_SIGMA_VALUES),
        LTX2PrepareLatentsStep,
        LTX2Image2VideoPrepareLatentsStep,
        LTX2PrepareAudioLatentsStep,
        LTX2PrepareCoordsStep,
        LTX2Image2VideoDenoiseStep,
        LTX2UnpackLatentsStep,
    ]
    block_names = [
        "set_timesteps",
        "prepare_latents",
        "prepare_i2v_latents",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
        "unpack",
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
class LTX25Image2VideoStage2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (image-to-video, second pass) that expands the text conditioning by `num_videos_per_prompt`,
    re-noises the supplied video/audio latents to `noise_scale`, adds image conditioning and runs the joint denoising
    loop over them on `stage_2_sigmas` (the LTX-2 stage-2 distilled schedule by default).

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          latents (`Tensor`):
              Video latents to refine, of shape [B, C, F, H, W] (normalized, not packed).
          noise_scale (`float`, *optional*):
              Noise level the latents are re-noised to before the pass. `None` (default) resolves to `sigmas[0]` when
              custom `sigmas` are supplied, else 1.0.
          stage_2_sigmas (`list`, *optional*, defaults to [0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          stage_2_timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          image_latents (`Tensor`):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          audio_latents (`Tensor`):
              Audio latents to refine, of shape [B, C, L, M] (normalized, not packed).
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    block_classes = [
        LTX2Stage2PrepareLatentsStep(sigmas_name="stage_2_sigmas", sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES),
        LTX2SetTimestepsStep(
            sigmas_name="stage_2_sigmas",
            timesteps_name="stage_2_timesteps",
            sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES,
        ),
        LTX2Image2VideoPrepareLatentsStep,
        LTX2Stage2PrepareAudioLatentsStep(sigmas_name="stage_2_sigmas", sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES),
        LTX2PrepareCoordsStep,
        LTX2Image2VideoDenoiseStep,
        LTX2UnpackLatentsStep,
    ]
    block_names = [
        "prepare_latents",
        "set_timesteps",
        "prepare_i2v_latents",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
        "unpack",
    ]

    @property
    def description(self):
        return (
            "Denoise block (image-to-video, second pass) that expands the text conditioning by "
            "`num_videos_per_prompt`, re-noises the supplied video/audio latents to `noise_scale`, adds image "
            "conditioning and runs the joint denoising loop over them on `stage_2_sigmas` (the LTX-2 stage-2 "
            "distilled schedule by default)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX25ConditionCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (condition-to-video) that expands the text conditioning by `num_videos_per_prompt`, applies the frame
    conditions to the video latents and runs the joint denoising loop.

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          condition_latents (`list`):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`):
              The number of frames in the generated video.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    # NOTE: prepare-latents runs *before* set-timesteps here, unlike the text-to-video / image-to-video steps. The
    # resolution-aware shift `mu` is computed from the packed latent sequence length, which for condition workflows
    # includes the appended keyframe tokens, so the latents have to exist first. This mirrors `LTX2ConditionPipeline`
    # (its section 4 runs before section 5).
    block_classes = [
        LTX2ConditionPrepareLatentsStep(sigmas_default=DISTILLED_SIGMA_VALUES),
        LTX2ConditionSetTimestepsStep(sigmas_default=DISTILLED_SIGMA_VALUES),
        LTX2ConditionPrepareAudioLatentsStep,
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
        LTX2TrimConditionTokensStep,
        LTX2UnpackLatentsStep,
    ]
    block_names = [
        "prepare_latents",
        "set_timesteps",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
        "trim_condition_tokens",
        "unpack",
    ]

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
class LTX25ConditionStage2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (condition-to-video, second pass) that expands the text conditioning by `num_videos_per_prompt`,
    applies the frame conditions on top of the supplied video latents, re-noises them and the supplied audio latents to
    `noise_scale` and runs the joint denoising loop over them on `stage_2_sigmas` (the LTX-2 stage-2 distilled schedule
    by default).

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          condition_latents (`list`):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          latents (`Tensor`):
              Video latents to refine, of shape [B, C, F, H, W] (normalized, not packed) of the generated video only
              (no appended condition tokens).
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          noise_scale (`float`, *optional*):
              Noise level the un-conditioned tokens are re-noised to. `None` (default) resolves to `sigmas[0]` when
              custom `sigmas` are supplied, else 1.0.
          stage_2_sigmas (`list`, *optional*, defaults to [0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          stage_2_timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          audio_latents (`Tensor`):
              Audio latents to refine, of shape [B, C, L, M] (normalized, not packed).
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    # NOTE: prepare-latents runs *before* set-timesteps here, unlike the text-to-video / image-to-video steps. The
    # resolution-aware shift `mu` is computed from the packed latent sequence length, which for condition workflows
    # includes the appended keyframe tokens, so the latents have to exist first. This mirrors `LTX2ConditionPipeline`
    # (its section 4 runs before section 5).
    block_classes = [
        LTX2ConditionStage2PrepareLatentsStep(
            sigmas_name="stage_2_sigmas", sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES
        ),
        LTX2ConditionSetTimestepsStep(
            sigmas_name="stage_2_sigmas",
            timesteps_name="stage_2_timesteps",
            sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES,
        ),
        LTX2Stage2PrepareAudioLatentsStep(sigmas_name="stage_2_sigmas", sigmas_default=STAGE_2_DISTILLED_SIGMA_VALUES),
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
        LTX2TrimConditionTokensStep,
        LTX2UnpackLatentsStep,
    ]
    block_names = [
        "prepare_latents",
        "set_timesteps",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
        "trim_condition_tokens",
        "unpack",
    ]

    @property
    def description(self):
        return (
            "Denoise block (condition-to-video, second pass) that expands the text conditioning by "
            "`num_videos_per_prompt`, applies the frame conditions on top of the supplied video latents, re-noises "
            "them and the supplied audio latents to `noise_scale` and runs the joint denoising loop over them on "
            "`stage_2_sigmas` (the LTX-2 stage-2 distilled schedule by default)."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("audio_latents", type_hint=torch.Tensor, description="Denoised audio latents."),
        ]


# auto_docstring
class LTX25InContextCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Denoise block (in-context) that expands the text conditioning by `num_videos_per_prompt`, folds the frame
    conditions and the IC-LoRA reference tokens into one latent sequence and runs the joint denoising loop. Reuses the
    condition denoise step unchanged: reference tokens are pinned by the same x0 blend as frame conditions, matching
    the reference implementation's uniform treatment of both.

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          condition_latents (`list`):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`):
              Per-condition conditioning strengths.
          condition_indices (`list`):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them); only their `strength` is read here. Omit for IC-LoRAs that
              carry their behavior in the adapter weights and take no reference video.
          reference_latents (`list`, *optional*):
              Per-reference VAE latents of shape [1, C, F, H, W] (normalized, not packed) from
              `LTX2ReferenceEncoderStep`, or `None` when no reference conditions were supplied
              (`LTX2AutoReferenceEncoderStep` is skipped).
          reference_downscale_factor (`int`, *optional*, defaults to 1):
              Ratio between the target and reference resolutions. The reference tokens' spatial coordinates are scaled
              by it so they land in the target coordinate space, preserving the positional relationship the IC-LoRA was
              trained on.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`):
              The number of frames in the generated video.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
              Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
              (default) leaves attention unmasked.
          conditioning_attention_mask (`Tensor`, *optional*):
              Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
              attention strength. Downsampled to each reference's latent grid and multiplied by
              `conditioning_attention_strength`.
          timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    # Same ordering rationale as `LTX25ConditionCoreDenoiseStep`: prepare-latents precedes set-timesteps because
    # `mu` is read off the packed sequence length, which here includes both keyframe and reference tokens.
    block_classes = [
        LTX2InContextPrepareLatentsStep(sigmas_default=DISTILLED_SIGMA_VALUES),
        LTX2ConditionSetTimestepsStep(sigmas_default=DISTILLED_SIGMA_VALUES),
        LTX2ConditionPrepareAudioLatentsStep,
        LTX2ConditionPrepareCoordsStep,
        LTX2ConditionDenoiseStep,
        LTX2TrimConditionTokensStep,
        LTX2UnpackLatentsStep,
    ]
    block_names = [
        "prepare_latents",
        "set_timesteps",
        "prepare_audio_latents",
        "prepare_coords",
        "denoise",
        "trim_condition_tokens",
        "unpack",
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
class LTX25AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Auto denoise block that selects the workflow based on inputs.
       - `LTX25InContextCoreDenoiseStep` when `reference_conditions` are provided (in-context / IC-LoRA).
       - `LTX25ConditionCoreDenoiseStep` when `condition_latents` are provided (condition-to-video).
       - `LTX25Image2VideoCoreDenoiseStep` when `image_latents` is provided.
       - `LTX25CoreDenoiseStep` otherwise (text-to-video).

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          condition_latents (`list`, *optional*):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them); only their `strength` is read here. Omit for IC-LoRAs that
              carry their behavior in the adapter weights and take no reference video.
          reference_latents (`list`, *optional*):
              Per-reference VAE latents of shape [1, C, F, H, W] (normalized, not packed) from
              `LTX2ReferenceEncoderStep`, or `None` when no reference conditions were supplied
              (`LTX2AutoReferenceEncoderStep` is skipped).
          reference_downscale_factor (`int`, *optional*, defaults to 1):
              Ratio between the target and reference resolutions. The reference tokens' spatial coordinates are scaled
              by it so they land in the target coordinate space, preserving the positional relationship the IC-LoRA was
              trained on.
          height (`int`, *optional*, defaults to 512):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 704):
              The width in pixels of the generated image.
          num_frames (`int`):
              The number of frames in the generated video.
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
              Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
              (default) leaves attention unmasked.
          conditioning_attention_mask (`Tensor`, *optional*):
              Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
              attention strength. Downsampled to each reference's latent grid and multiplied by
              `conditioning_attention_strength`.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    block_classes = [
        LTX25InContextCoreDenoiseStep,
        LTX25ConditionCoreDenoiseStep,
        LTX25Image2VideoCoreDenoiseStep,
        LTX25CoreDenoiseStep,
    ]
    block_names = ["in_context", "condition", "image2video", "text2video"]
    block_trigger_inputs = ["reference_conditions", "condition_latents", "image_latents"]
    default_block_name = "text2video"

    def select_block(self, reference_conditions=None, condition_latents=None, image_latents=None) -> str | None:
        # An IC-LoRA that takes no reference video lands on the condition branch, which is the right answer rather
        # than a fallback: `LTX2InContextPrepareLatentsStep` with no reference tokens does exactly what
        # `LTX2ConditionPrepareLatentsStep` does, and the extra `num_ref_tokens` it emits is only read by the
        # attention-mask construction, which is skipped in that case anyway.
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
            " - `LTX25InContextCoreDenoiseStep` when `reference_conditions` are provided (in-context / IC-LoRA).\n"
            " - `LTX25ConditionCoreDenoiseStep` when `condition_latents` are provided (condition-to-video).\n"
            " - `LTX25Image2VideoCoreDenoiseStep` when `image_latents` is provided.\n"
            " - `LTX25CoreDenoiseStep` otherwise (text-to-video).\n"
        )


# auto_docstring
class LTX25AutoStage2CoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Auto denoise block for the second pass of the two-stage recipe, selecting the workflow based on inputs. Each branch re-noises the video / audio latents in state on `stage_2_sigmas` instead of sampling fresh noise:
       - `LTX25ConditionStage2CoreDenoiseStep` when `condition_latents` are provided (condition-to-video; also the
         second pass of an in-context run, whose references shape the first pass only).
       - `LTX25Image2VideoStage2CoreDenoiseStep` when `image_latents` is provided.
       - `LTX25Stage2CoreDenoiseStep` otherwise (text-to-video).

      Components:
          transformer (`LTX2VideoTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`LTX2Guidance`) audio_guider (`LTX2Guidance`)

      Inputs:
          condition_latents (`list`, *optional*):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          latents (`Tensor`):
              Video latents to refine, of shape [B, C, F, H, W] (normalized, not packed) of the generated video only
              (no appended condition tokens).
          frame_rate (`float`, *optional*, defaults to 24.0):
              Frames per second of the generated video.
          noise_scale (`float`, *optional*):
              Noise level the un-conditioned tokens are re-noised to. `None` (default) resolves to `sigmas[0]` when
              custom `sigmas` are supplied, else 1.0.
          stage_2_sigmas (`list`, *optional*, defaults to [0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          batch_size (`int`):
              The number of prompts being denoised, used to expand conditioning per prompt.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          stage_2_timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          audio_latents (`Tensor`):
              Audio latents to refine, of shape [B, C, L, M] (normalized, not packed).
          dtype (`dtype`):
              The dtype the model inputs are cast to.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          connector_prompt_embeds (`Tensor`):
              Video-branch text conditioning (cond), expanded per prompt.
          connector_audio_prompt_embeds (`Tensor`):
              Audio-branch text conditioning (cond), expanded per prompt.
          connector_attention_mask (`Tensor`):
              Binary text attention mask (cond), expanded per prompt.
          negative_connector_prompt_embeds (`Tensor`, *optional*):
              Video-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_audio_prompt_embeds (`Tensor`, *optional*):
              Audio-branch text conditioning (uncond); read only under classifier-free guidance.
          negative_connector_attention_mask (`Tensor`, *optional*):
              Binary text attention mask (uncond); read only under classifier-free guidance.
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          audio_latents (`Tensor`):
              Denoised audio latents.
    """

    model_name = "ltx2.5"
    block_classes = [
        LTX25ConditionStage2CoreDenoiseStep,
        LTX25Image2VideoStage2CoreDenoiseStep,
        LTX25Stage2CoreDenoiseStep,
    ]
    block_names = ["condition", "image2video", "text2video"]
    block_trigger_inputs = ["condition_latents", "image_latents"]
    default_block_name = "text2video"

    def select_block(self, condition_latents=None, image_latents=None) -> str | None:
        # The second pass of an in-context run is a plain condition pass: the references only shape the first pass,
        # as in `LTX2InContextPipeline`.
        if condition_latents is not None:
            return "condition"
        if image_latents is not None:
            return "image2video"
        return "text2video"

    @property
    def description(self):
        return (
            "Auto denoise block for the second pass of the two-stage recipe, selecting the workflow based on "
            "inputs. Each branch re-noises the video / audio latents in state on `stage_2_sigmas` instead of "
            "sampling fresh noise:\n"
            " - `LTX25ConditionStage2CoreDenoiseStep` when `condition_latents` are provided (condition-to-video; "
            "also the second pass of an in-context run, whose references shape the first pass only).\n"
            " - `LTX25Image2VideoStage2CoreDenoiseStep` when `image_latents` is provided.\n"
            " - `LTX25Stage2CoreDenoiseStep` otherwise (text-to-video)."
        )


# auto_docstring
class LTX25DecoderStep(SequentialPipelineBlocks):
    """
    Decode stage for LTX-2.5: denoises the video latents with the diffusion decoder and vocodes the audio latents.

      Components:
          diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) video_processor (`VideoProcessor`) audio_vae
          (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          latents (`Tensor`):
              Video latents of shape [B, C, F, H, W] (normalized, not packed).
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          audio_latents (`Tensor`):
              Audio latents of shape [B, C, L, M] (normalized, not packed).

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
            "latents."
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
    the convolutional VAE instead, swap the decode block: `blocks.sub_blocks["decode"] = LTX2DecoderStep()`.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`
        - `condition`: requires `conditions`, `prompt`
        - `in_context`: requires `reference_conditions`, `num_frames`, `prompt`

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) transformer (`LTX2VideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)
          diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) audio_vae (`AutoencoderKLLTX2Audio`) vocoder
          (`LTX2Vocoder`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          conditions (`list`, *optional*):
              `LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
              generated video.
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
          enable_prompt_enhancement (`bool`, *optional*, defaults to False):
              Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to have this step predict it with the `duration_head`;
              the denoise blocks then take the predicted count.
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
              model was trained against. `None` (default) uses the pipeline's `default_image_crf` (33 through LTX-2.3,
              18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when re-compression runs.
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens the IC-LoRA
              adapter attends to.
          reference_downscale_factor (`int`, *optional*, defaults to 1):
              Ratio between the target and reference resolutions; 2 means the reference is preprocessed at half the
              target resolution. Must match the factor the IC-LoRA was trained with.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          condition_latents (`list`, *optional*):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_latents (`list`, *optional*):
              Per-reference VAE latents of shape [1, C, F, H, W] (normalized, not packed) from
              `LTX2ReferenceEncoderStep`, or `None` when no reference conditions were supplied
              (`LTX2AutoReferenceEncoderStep` is skipped).
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
              Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
              (default) leaves attention unmasked.
          conditioning_attention_mask (`Tensor`, *optional*):
              Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
              attention strength. Downsampled to each reference's latent grid and multiplied by
              `conditioning_attention_strength`.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
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
        LTX25AutoPromptEnhancerStep,
        LTX25TextConditioningStep,
        LTX25AutoDurationStep,
        LTX25AutoVaeEncoderStep,
        LTX25AutoConditionEncoderStep,
        LTX25AutoReferenceEncoderStep,
        LTX2TextInputStep,
        LTX25AutoCoreDenoiseStep,
        LTX25DecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "duration",
        "vae_encoder",
        "condition_encoder",
        "reference_encoder",
        "input",
        "denoise",
        "decode",
    ]
    # `num_frames` on `in_context` is a requirement, not a trigger: the in-context checkpoints ship without a
    # `duration_head`, so the workflow drops `LTX25AutoDurationStep` and `LTX2ConditionEncoderStep` raises if
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
            "LTX2DecoderStep()`."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]


# auto_docstring
class LTX25UpsampleStep(SequentialPipelineBlocks):
    """
    Upsample stage of the two-stage recipe: `LTX2LatentUpsampleStep` as an LTX-2.5 block, so that popped into its own
    pipeline it resolves the LTX-2.5 latent statistics when no autoencoder is loaded.

      Components:
          latent_upsampler (`LTX2LatentUpsamplerModel`) transformer (`LTX2VideoTransformer3DModel`)

      Inputs:
          latents (`Tensor`):
              Video latents to upsample, of shape [B, C, F, H, W] (normalized, not packed).

      Outputs:
          latents (`Tensor`):
              Upsampled video latents of shape [B, C, F, 2H, 2W] (normalized, not packed).
          height (`int`):
              Height of the upsampled latents, in pixels.
          width (`int`):
              Width of the upsampled latents, in pixels.
    """

    model_name = "ltx2.5"
    block_classes = [LTX2LatentUpsampleStep]
    block_names = ["latent_upsample"]

    @property
    def description(self):
        return (
            "Upsample stage of the two-stage recipe: `LTX2LatentUpsampleStep` as an LTX-2.5 block, so that popped "
            "into its own pipeline it resolves the LTX-2.5 latent statistics when no autoencoder is loaded."
        )


# auto_docstring
class LTX25TwoStageBlocks(SequentialPipelineBlocks):
    """
    Blocks for the LTX-2.5 distilled two-stage recipe (joint video + audio) in one call, for every workflow
    `LTX25AutoBlocks` supports: a first pass at the requested `height` / `width`, a 2x latent upsample, and a second
    pass that refines at the upsampled resolution, with the diffusion decoder at the end -- so the output is twice the
    size asked for, as with the standard pipelines. `stage_1` is the same auto denoise step as `LTX25AutoBlocks`;
    `stage_2` selects the workflow's second-pass group, which re-noises the upsampled latents on `stage_2_sigmas`.
    Image and frame conditions are re-encoded at the upsampled resolution ahead of the second pass, as the standard
    pipelines do on their second call. The text conditioning is expanded by `num_videos_per_prompt` once, by the
    `input` step, so `stage_1` / `upsample` / `stage_2` / `decode` can each be popped and run as their own pipeline:
    `stage_1` followed by `decode` previews the first pass, and popping `stage_1` and `upsample` leaves a standalone
    second pass that takes `latents` / `audio_latents`.

      Supported workflows:
        - `text2video`: requires `prompt`
        - `image2video`: requires `image`, `prompt`
        - `condition`: requires `conditions`, `prompt`
        - `in_context`: requires `reference_conditions`, `num_frames`, `prompt`

      Components:
          prompt_enhancer (`PreTrainedModel`) processor (`ProcessorMixin`) text_encoder (`PreTrainedModel`) tokenizer
          (`PreTrainedTokenizerBase`) connectors (`LTX2TextConnectors`) duration_head (`LTX2DurationHead`) vae
          (`AutoencoderKLLTX2Video`) video_processor (`VideoProcessor`) transformer (`LTX2VideoTransformer3DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`) guider (`LTX2Guidance`) audio_guider (`LTX2Guidance`)
          latent_upsampler (`LTX2LatentUpsamplerModel`) diffusion_decoder (`LTX2VideoDiffusionDecoderModel`) audio_vae
          (`AutoencoderKLLTX2Audio`) vocoder (`LTX2Vocoder`)

      Inputs:
          prompt (`str`, *optional*):
              The prompt or prompts to guide image generation.
          conditions (`list`, *optional*):
              `LTX2VideoCondition` (or list of them) placing image/video conditions at latent frame indices of the
              generated video.
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
          enable_prompt_enhancement (`bool`, *optional*, defaults to False):
              Whether to run the prompt enhancer. Opt-in, matching the Lightricks reference pipelines.
          negative_prompt (`str`, *optional*):
              The prompt or prompts not to guide the image generation.
          max_sequence_length (`int`, *optional*, defaults to 1024):
              Maximum sequence length for prompt encoding.
          num_frames (`int`, *optional*):
              The number of frames in the generated video. Omit to have this step predict it with the `duration_head`;
              the denoise blocks then take the predicted count.
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
              model was trained against. `None` (default) uses the pipeline's `default_image_crf` (33 through LTX-2.3,
              18 for LTX-2.5). Pass `0` to skip re-compression. Requires a `PIL.Image.Image` when re-compression runs.
          reference_conditions (`list`, *optional*):
              `LTX2ReferenceCondition` (or list of them) whose videos are encoded into extra latent tokens the IC-LoRA
              adapter attends to.
          reference_downscale_factor (`int`, *optional*, defaults to 1):
              Ratio between the target and reference resolutions; 2 means the reference is preprocessed at half the
              target resolution. Must match the factor the IC-LoRA was trained with.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          condition_latents (`list`, *optional*):
              Per-condition VAE latents of shape [1, C, F, H, W] (normalized, not packed).
          condition_strengths (`list`, *optional*):
              Per-condition conditioning strengths.
          condition_indices (`list`, *optional*):
              Per-condition latent frame index at which the condition is applied.
          condition_pixel_frames (`list`, *optional*):
              Per-condition trimmed pixel frame count, used to clamp single-frame keyframe coords.
          reference_latents (`list`, *optional*):
              Per-reference VAE latents of shape [1, C, F, H, W] (normalized, not packed) from
              `LTX2ReferenceEncoderStep`, or `None` when no reference conditions were supplied
              (`LTX2AutoReferenceEncoderStep` is skipped).
          noise_scale (`float`, *optional*):
              Initial noise level for the un-conditioned tokens. `None` (default) resolves to `sigmas[0]` when custom
              `sigmas` are supplied, else 1.0.
          sigmas (`list`, *optional*, defaults to [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          conditioning_attention_strength (`float`, *optional*, defaults to 1.0):
              Scalar in [0, 1] controlling how strongly the noisy tokens and reference tokens attend to each other. 1.0
              (default) leaves attention unmasked.
          conditioning_attention_mask (`Tensor`, *optional*):
              Optional pixel-space mask of shape (1, 1, F, H, W) with values in [0, 1] giving spatially varying
              attention strength. Downsampled to each reference's latent grid and multiplied by
              `conditioning_attention_strength`.
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          image_latents (`Tensor`, *optional*):
              VAE-encoded reference-image latents used for image-to-video conditioning.
          stage_2_sigmas (`list`, *optional*, defaults to [0.909375, 0.725, 0.421875]):
              Custom sigmas for the denoising process.
          stage_2_timesteps (`Tensor`, *optional*):
              Timesteps for the denoising process.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
          audio (`Tensor`):
              The generated audio waveform.
    """

    model_name = "ltx2.5-two-stage"
    block_classes = [
        LTX25AutoPromptEnhancerStep,
        LTX25TextConditioningStep,
        LTX25AutoDurationStep,
        LTX25AutoVaeEncoderStep,
        LTX25AutoConditionEncoderStep,
        LTX25AutoReferenceEncoderStep,
        LTX2TextInputStep,
        LTX25AutoCoreDenoiseStep,
        LTX25UpsampleStep,
        LTX25AutoVaeEncoderStep,
        LTX25AutoConditionEncoderStep,
        LTX25AutoStage2CoreDenoiseStep,
        LTX25DecoderStep,
    ]
    block_names = [
        "prompt_enhancer",
        "text_encoder",
        "duration",
        "vae_encoder",
        "condition_encoder",
        "reference_encoder",
        "input",
        "stage_1",
        "upsample",
        "stage_2_vae_encoder",
        "stage_2_condition_encoder",
        "stage_2",
        "decode",
    ]
    _workflow_map = {
        "text2video": {"prompt": True},
        "image2video": {"image": True, "prompt": True},
        "condition": {"conditions": True, "prompt": True},
        "in_context": {"reference_conditions": True, "num_frames": True, "prompt": True},
    }

    @property
    def description(self):
        return (
            "Blocks for the LTX-2.5 distilled two-stage recipe (joint video + audio) in one call, for every workflow "
            "`LTX25AutoBlocks` supports: a first pass at the requested `height` / `width`, a 2x latent upsample, and "
            "a second pass that refines at the upsampled resolution, with the diffusion decoder at the end -- so the "
            "output is twice the size asked for, as with the standard pipelines. `stage_1` is the same auto denoise "
            "step as `LTX25AutoBlocks`; `stage_2` selects the workflow's second-pass group, which re-noises the "
            "upsampled latents on `stage_2_sigmas`. Image and frame conditions are re-encoded at the upsampled "
            "resolution ahead of the second pass, as the standard pipelines do on their second call. The text "
            "conditioning is expanded by `num_videos_per_prompt` once, by the `input` step, so `stage_1` / "
            "`upsample` / `stage_2` / `decode` can each be popped and run as their own pipeline: `stage_1` followed "
            "by `decode` previews the first pass, and popping `stage_1` and `upsample` leaves a standalone second "
            "pass that takes `latents` / `audio_latents`."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("videos"),
            OutputParam("audio", type_hint=torch.Tensor, description="The generated audio waveform."),
        ]
