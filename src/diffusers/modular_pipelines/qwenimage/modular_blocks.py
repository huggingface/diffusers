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
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import (
    QwenImageControlNetInputsStep,
    QwenImageEditRoPEInputsStep,
    QwenImageInpaintPrepareLatentsStep,
    QwenImagePackLatentsDynamicStep,
    QwenImagePrepareLatentsStep,
    QwenImageRoPEInputsStep,
    QwenImageSetTimestepsStep,
    QwenImageSetTimestepsWithStrengthStep,
)
from .decoders import QwenImageDecodeDynamicStep, QwenImageInpaintProcessImagesOutputStep
from .denoise import (
    QwenImageDenoiseStep,
    QwenImageEditDenoiseStep,
    QwenImageInpaintDenoiseStep,
    QwenImageLoopBeforeDenoiserControlNet,
)
from .encoders import (
    QwenImageEditResizeStep,
    QwenImageEditTextEncoderStep,
    QwenImageInpaintVaeEncoderStep,
    QwenImageTextEncoderStep,
    QwenImageVaeEncoderDynamicStep,
)
from .inputs import QwenImageInputsDynamicStep


logger = logging.get_logger(__name__)


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("input", QwenImageInputsDynamicStep()),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_additional_inputs", QwenImageRoPEInputsStep()),
        ("denoise", QwenImageDenoiseStep()),
        ("decode", QwenImageDecodeDynamicStep()),
    ]
)


INPAINT_BLOCKS = InsertableDict(
    [
        ("text_encoder", QwenImageTextEncoderStep()),
        ("vae_encoder", QwenImageInpaintVaeEncoderStep()),
        ("input", QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"])),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsWithStrengthStep()),
        ("prepare_inpaint_latents", QwenImageInpaintPrepareLatentsStep()),
        ("prepare_additional_inputs", QwenImageRoPEInputsStep()),
        ("denoise", QwenImageInpaintDenoiseStep()),
        ("decode", QwenImageDecodeDynamicStep(include_image_processor=False)),
        ("postprocess", QwenImageInpaintProcessImagesOutputStep()),
    ]
)

CONTROLNET_BLOCKS = InsertableDict(
    [
        (
            "controlnet_vae_encoder",
            QwenImageVaeEncoderDynamicStep(input_name="control_image", output_name="control_image_latents"),
        ),
        ("controlnet_before_denoise", QwenImageControlNetInputsStep()),
        (
            "controlnet_denoise_loop_before",
            QwenImageLoopBeforeDenoiserControlNet(),
        ),  # insert before the denoiseloop_denoiser
    ]
)


EDIT_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageEditResizeStep()),
        ("text_encoder", QwenImageEditTextEncoderStep()),
        (
            "vae_encoder",
            QwenImageVaeEncoderDynamicStep(input_name="image", output_name="image_latents", do_resize=False),
        ),
        ("input", QwenImageInputsDynamicStep(image_latent_input_names=["image_latents"])),
        ("prepare_image_latents", QwenImagePackLatentsDynamicStep("image_latents")),
        ("prepare_latents", QwenImagePrepareLatentsStep()),
        ("set_timesteps", QwenImageSetTimestepsStep()),
        ("prepare_additional_inputs", QwenImageEditRoPEInputsStep()),
        ("denoise", QwenImageEditDenoiseStep()),
        ("decode", QwenImageDecodeDynamicStep()),
    ]
)

ALL_BLOCKS = {
    "text2image": TEXT2IMAGE_BLOCKS,
    "edit": EDIT_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
}
