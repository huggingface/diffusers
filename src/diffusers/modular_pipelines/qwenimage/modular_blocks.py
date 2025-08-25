# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import (
    QwenImageControlNetInputStep,
    QwenImageEditPrepareAdditionalInputsStep,
    QwenImageEditResizeStep,
    QwenImageInputsDynamicStep,
    QwenImagePrepareAdditionalInputsStep,
    QwenImagePrepareImageLatentsDynamicStep,
    QwenImagePrepareLatentsStep,
    QwenImageResizeDynamicStep,
    QwenImageSetTimestepsStep,
    QwenImageTextInputsStep,
)
from .decoders import QwenImageDecodeStep
from .denoise import (
    QwenImageControlNetLoopBeforeDenoiser,
    QwenImageDenoiseStep,
    QwenImageEditDenoiseStep,
)
from .encoders import QwenImageEditTextEncoderStep, QwenImageTextEncoderStep, QwenImageVaeEncoderDynamicStep


logger = logging.get_logger(__name__)


class QwenImageEditPrepareImageLatentsStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = [
        QwenImageInputsDynamicStep("image_latents"),
        QwenImagePrepareImageLatentsDynamicStep("image_latents"),
    ]

    block_names = ["input", "pachify"]

    @property
    def description(self) -> str:
        return "Step that prepares the image latents for the edit process"


controlnet_image_resize = QwenImageResizeDynamicStep("control_image")
controlnet_vae_encoder = QwenImageVaeEncoderDynamicStep(
    input_name="control_image", output_name="control_image_latents"
)


class QwenImageControlnetVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = [
        controlnet_image_resize,  # resize the controlnet image to height/width
        controlnet_vae_encoder,  # encode
    ]

    block_names = ["resize", "encode"]

    @property
    def description(self) -> str:
        return "Step that encodes the control image"


class QwenImageControlnetBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = [
        QwenImageInputsDynamicStep("control_image_latents"),  # duplicate to match batch_size
        QwenImagePrepareImageLatentsDynamicStep("control_image_latents"),  # pachify the latents
        QwenImageControlNetInputStep,  # prepare the controlnet inputs e.g. controlnet_keep, controlnet_conditioning_scale, etc.
    ]

    block_names = ["adjust_control_image_latent", "patchify_control_image_latent", "prepare_controlnet_inputs"]

    @property
    def description(self) -> str:
        return "Step that prepares the controlnet inputs. Insert before the Denoise Step, after set_timesteps step."


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep),
        ("input", QwenImageTextInputsStep),
        ("prepare_latents", QwenImagePrepareLatentsStep),
        ("set_timesteps", QwenImageSetTimestepsStep),
        ("prepare_additional_inputs", QwenImagePrepareAdditionalInputsStep),
        ("denoise", QwenImageDenoiseStep),
        ("decode", QwenImageDecodeStep),
    ]
)

CONTROLNET_BLOCKS = InsertableDict(
    [
        ("controlnet_vae_encoder", QwenImageControlnetVaeEncoderStep),
        ("controlnet_before_denoise", QwenImageControlnetBeforeDenoiseStep),
        (
            "controlnet_denoise_loop_before",
            QwenImageControlNetLoopBeforeDenoiser,
        ),  # insert before the denoiseloop_denoiser
    ]
)


EDIT_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageEditResizeStep),
        ("text_encoder", QwenImageEditTextEncoderStep),
        ("vae_encoder", QwenImageVaeEncoderDynamicStep(input_name="image", output_name="image_latents")),
        ("input", QwenImageTextInputsStep),
        ("prepare_image_latents", QwenImageEditPrepareImageLatentsStep),
        ("prepare_latents", QwenImagePrepareLatentsStep),
        ("set_timesteps", QwenImageSetTimestepsStep),
        ("prepare_additional_inputs", QwenImageEditPrepareAdditionalInputsStep),
        ("denoise", QwenImageEditDenoiseStep),
        ("decode", QwenImageDecodeStep),
    ]
)

ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "edit": EDIT_BLOCKS,
}
