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
    SD3Img2ImgPrepareLatentsStep,
    SD3Img2ImgSetTimestepsStep,
    SD3PrepareLatentsStep,
    SD3SetTimestepsStep,
)
from .decoders import SD3DecodeStep
from .denoise import SD3DenoiseStep
from .encoders import (
    SD3ProcessImagesInputStep,
    SD3TextEncoderStep,
    SD3VaeEncoderStep,
)
from .inputs import (
    SD3AdditionalInputsStep,
    SD3TextInputStep,
)


logger = logging.get_logger(__name__)


class SD3Img2ImgVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes = [SD3ProcessImagesInputStep(), SD3VaeEncoderStep()]
    block_names = ["preprocess", "encode"]


class SD3AutoVaeEncoderStep(AutoPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[SD3Img2ImgVaeEncoderStep]
    block_names = ["img2img"]
    block_trigger_inputs =["image"]


class SD3BeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[SD3PrepareLatentsStep(), SD3SetTimestepsStep()]
    block_names = ["prepare_latents", "set_timesteps"]


class SD3Img2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[
        SD3PrepareLatentsStep(),
        SD3Img2ImgSetTimestepsStep(),
        SD3Img2ImgPrepareLatentsStep(),
    ]
    block_names = ["prepare_latents", "set_timesteps", "prepare_img2img_latents"]


class SD3AutoBeforeDenoiseStep(AutoPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[SD3Img2ImgBeforeDenoiseStep, SD3BeforeDenoiseStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class SD3Img2ImgInputStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[SD3TextInputStep(), SD3AdditionalInputsStep()]
    block_names =["text_inputs", "additional_inputs"]


class SD3AutoInputStep(AutoPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes = [SD3Img2ImgInputStep, SD3TextInputStep]
    block_names = ["img2img", "text2image"]
    block_trigger_inputs = ["image_latents", None]


class SD3CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "stable-diffusion-3"
    block_classes =[SD3AutoInputStep, SD3AutoBeforeDenoiseStep, SD3DenoiseStep]
    block_names =["input", "before_denoise", "denoise"]
    @property
    def outputs(self):
        return [OutputParam.template("latents")]


AUTO_BLOCKS = InsertableDict([
        ("text_encoder", SD3TextEncoderStep()),
        ("vae_encoder", SD3AutoVaeEncoderStep()),
        ("denoise", SD3CoreDenoiseStep()),
        ("decode", SD3DecodeStep()),
    ]
)


class SD3AutoBlocks(SequentialPipelineBlocks):
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