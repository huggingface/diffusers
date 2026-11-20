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

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    PAN2Image2VideoInputStep,
    PAN2Image2VideoPrepareConditionLatentsStep,
    PAN2PrepareConditionLatentsStep,
    PAN2PrepareLatentsStep,
    PAN2SetTimestepsStep,
    PAN2TextInputStep,
)
from .decoders import PAN2VaeDecoderStep
from .denoise import PAN2DenoiseStep
from .encoders import PAN2TextEncoderStep, PAN2VaeEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class PAN2AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    VAE encoder step that encodes the first frame into image latents.
       - `PAN2VaeEncoderStep` is used when `image` is provided (image-to-video).
       - If `image` is not provided, this step is skipped (text-to-video).

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          image (`Image`, *optional*):
              The first frame, a single image.
          height (`int`, *optional*):
              The height in pixels of the generated video, 704 by default.
          width (`int`, *optional*):
              The width in pixels of the generated video, 1248 by default.

      Outputs:
          image_latents (`Tensor`):
              The normalized latents of the first frame, of shape `(1, 48, 1, h, w)`.
    """

    model_name = "pan2"
    block_classes = [PAN2VaeEncoderStep]
    block_names = ["i2v"]
    block_trigger_inputs = ["image"]

    @property
    def description(self):
        return (
            "VAE encoder step that encodes the first frame into image latents.\n"
            " - `PAN2VaeEncoderStep` is used when `image` is provided (image-to-video).\n"
            " - If `image` is not provided, this step is skipped (text-to-video)."
        )


PAN2CoreDenoiseBlocks = InsertableDict(
    [
        ("input", PAN2TextInputStep()),
        ("set_timesteps", PAN2SetTimestepsStep()),
        ("prepare_latents", PAN2PrepareLatentsStep()),
        ("prepare_condition_latents", PAN2PrepareConditionLatentsStep()),
        ("denoise", PAN2DenoiseStep()),
    ]
)


# auto_docstring
class PAN2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that runs the denoising process for text-to-video.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`PAN2Transformer3DModel`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              Text embeddings used to guide the video generation.
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative text embeddings used to guide the video generation.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          height (`int`, *optional*):
              The height in pixels of the generated video, 704 by default.
          width (`int`, *optional*):
              The width in pixels of the generated video, 1248 by default.
          num_frames (`int`, *optional*, defaults to 313):
              Number of video frames to generate, of the form `4 * n + 1`.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents to use as inputs for video generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "pan2"
    block_classes = PAN2CoreDenoiseBlocks.values()
    block_names = PAN2CoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Core denoise step that runs the denoising process for text-to-video."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


PAN2Image2VideoCoreDenoiseBlocks = InsertableDict(
    [
        ("input", PAN2TextInputStep()),
        ("image_input", PAN2Image2VideoInputStep()),
        ("set_timesteps", PAN2SetTimestepsStep()),
        ("prepare_latents", PAN2PrepareLatentsStep()),
        ("prepare_condition_latents", PAN2Image2VideoPrepareConditionLatentsStep()),
        ("denoise", PAN2DenoiseStep()),
    ]
)


# auto_docstring
class PAN2Image2VideoCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that runs the denoising process for image-to-video, conditioned on the first frame.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`PAN2Transformer3DModel`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              Text embeddings used to guide the video generation.
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative text embeddings used to guide the video generation.
          image_latents (`Tensor`):
              The normalized first-frame latents. Can be generated from the vae_encoder step.
          num_inference_steps (`int`, *optional*, defaults to 50):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 313):
              Number of video frames to generate, of the form `4 * n + 1`.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents to use as inputs for video generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "pan2"
    block_classes = PAN2Image2VideoCoreDenoiseBlocks.values()
    block_names = PAN2Image2VideoCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return "Core denoise step that runs the denoising process for image-to-video, conditioned on the first frame."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class PAN2AutoCoreDenoiseStep(AutoPipelineBlocks):
    """
    Core denoise step for PAN 2.
       - `PAN2Image2VideoCoreDenoiseStep` is used when `image_latents` is provided (image-to-video).
       - `PAN2CoreDenoiseStep` is used otherwise (text-to-video).

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`) transformer (`PAN2Transformer3DModel`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of videos to generate per prompt.
          prompt_embeds (`Tensor`):
              Text embeddings used to guide the video generation.
          negative_prompt_embeds (`Tensor`, *optional*):
              Negative text embeddings used to guide the video generation.
          image_latents (`Tensor`, *optional*):
              The normalized first-frame latents. Can be generated from the vae_encoder step.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          num_frames (`int`, *optional*, defaults to 313):
              Number of video frames to generate, of the form `4 * n + 1`.
          latents (`Tensor`):
              Pre-generated noisy latents to use as inputs for video generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          height (`int`, *optional*):
              The height in pixels of the generated video, 704 by default.
          width (`int`, *optional*):
              The width in pixels of the generated video, 1248 by default.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "pan2"
    block_classes = [PAN2Image2VideoCoreDenoiseStep, PAN2CoreDenoiseStep]
    block_names = ["i2v", "t2v"]
    block_trigger_inputs = ["image_latents", None]

    @property
    def description(self):
        return (
            "Core denoise step for PAN 2.\n"
            " - `PAN2Image2VideoCoreDenoiseStep` is used when `image_latents` is provided (image-to-video).\n"
            " - `PAN2CoreDenoiseStep` is used otherwise (text-to-video)."
        )


# auto_docstring
class PAN2Blocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for text-to-video and image-to-video generation with PAN 2.

      Supported workflows:
        - `t2v`: requires `prompt`
        - `i2v`: requires `prompt`, `image`

      Components:
          text_encoder (`Qwen3_5TextModel`) tokenizer (`Qwen2Tokenizer`) safety_checker (`PAN2SafetyChecker`) vae
          (`AutoencoderKLWan`) video_processor (`VideoProcessor`) scheduler (`FlowMatchEulerDiscreteScheduler`)
          transformer (`PAN2Transformer3DModel`) guider (`ClassifierFreeGuidance`)

      Configs:
          enable_safety_checker (default: True)

      Inputs:
          prompt (`str`):
              The prompt to guide generation, a single string.
          negative_prompt (`str`, *optional*):
              The prompt not to guide generation, a single string. When guidance needs one and none (or an empty one)
              is passed, PAN 2's default negative prompt is used, its time ranges set to the clip's duration.
          max_sequence_length (`int`, *optional*, defaults to 1536):
              Maximum sequence length for prompt encoding.
          num_frames (`int`, *optional*, defaults to 313):
              Number of video frames to generate, of the form `4 * n + 1`.
          fps (`float`, *optional*, defaults to 24.0):
              The frame rate of the video, which sets its duration in the default negative prompt.
          image (`Image`, *optional*):
              The first frame, a single image.
          height (`int`, *optional*):
              The height in pixels of the generated video, 704 by default.
          width (`int`, *optional*):
              The width in pixels of the generated video, 1248 by default.
          num_videos_per_prompt (`int`, *optional*, defaults to 1):
              The number of videos to generate per prompt.
          image_latents (`Tensor`, *optional*):
              The normalized first-frame latents. Can be generated from the vae_encoder step.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigmas for the denoising process.
          latents (`Tensor`):
              Pre-generated noisy latents to use as inputs for video generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          output_type (`str`, *optional*, defaults to np):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "pan2"
    block_classes = [PAN2TextEncoderStep(), PAN2AutoVaeEncoderStep(), PAN2AutoCoreDenoiseStep(), PAN2VaeDecoderStep()]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "t2v": {"prompt": True},
        "i2v": {"prompt": True, "image": True},
    }

    @property
    def description(self):
        return "Modular pipeline blocks for text-to-video and image-to-video generation with PAN 2."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
