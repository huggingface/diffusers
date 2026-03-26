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
    StableDiffusion3Img2ImgPrepareLatentsStep,
    StableDiffusion3Img2ImgSetTimestepsStep,
    StableDiffusion3PrepareLatentsStep,
    StableDiffusion3SetTimestepsStep,
)
from .decoders import StableDiffusion3DecodeStep
from .denoise import StableDiffusion3DenoiseStep
from .encoders import (
    StableDiffusion3ProcessImagesInputStep,
    StableDiffusion3TextEncoderStep,
    StableDiffusion3VaeEncoderStep,
)
from .inputs import (
    StableDiffusion3AdditionalInputsStep,
    StableDiffusion3TextInputStep,
)


logger = logging.get_logger(__name__)


class StableDiffusion3Img2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes = [StableDiffusion3ProcessImagesInputStep(), StableDiffusion3VaeEncoderStep()]
    block_names = ["preprocess", "encode"]


class StableDiffusion3AutoVaeEncoderStep(AutoPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[StableDiffusion3Img2ImgVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs =["image"]


class StableDiffusion3BeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[StableDiffusion3PrepareLatentsStep(), StableDiffusion3SetTimestepsStep()]
    block_names = ["prepare_latents", "set_timesteps"]


class StableDiffusion3Img2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[
        StableDiffusion3PrepareLatentsStep(),
        StableDiffusion3Img2ImgSetTimestepsStep(),
        StableDiffusion3Img2ImgPrepareLatentsStep(),
    ]
    block_names = ["prepare_latents", "set_timesteps", "prepare_img2img_latents"]


class StableDiffusion3AutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[StableDiffusion3Img2ImgBeforeDenoiseStep, StableDiffusion3BeforeDenoiseStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class StableDiffusion3Img2ImgInputStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[StableDiffusion3TextInputStep(), StableDiffusion3AdditionalInputsStep()]
    block_names =["text_inputs", "additional_inputs"]


class StableDiffusion3AutoInputStep(AutoPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes = [StableDiffusion3Img2ImgInputStep, StableDiffusion3TextInputStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class StableDiffusion3CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[StableDiffusion3AutoInputStep, StableDiffusion3AutoBeforeDenoiseStep, StableDiffusion3DenoiseStep]
    block_names =["input", "before_denoise", "denoise"]
    @property
    def outputs(self):
        return [OutputParam.template("latents")]


AUTO_BLOCKS = InsertableDict([
        ("text_encoder", StableDiffusion3TextEncoderStep()),
        ("vae_encoder", StableDiffusion3AutoVaeEncoderStep()),
        ("denoise", StableDiffusion3CoreDenoiseStep()),
        ("decode", StableDiffusion3DecodeStep()),
    ]
)


class StableDiffusion3AutoBlocks(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes = AUTO_BLOCKS.values()
    block_names = AUTO_BLOCKS.keys()

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
    }

    @property
    def outputs(self):
        return [OutputParam.template("images")]
