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
    QwenImageControlNetPrepareInputsStep,
    QwenImagePrepareLatentsWithStrengthStep,
    QwenImageEditRoPEInputsStep,
    QwenImageRoPEInputsStep,
    QwenImagePackLatentsDynamicStep,
    QwenImagePrepareLatentsStep,
    QwenImageSetTimestepsStep,
    QwenImageSetTimestepsWithStrengthStep,
    QwenImageCreateMaskLatentsStep,
)

from .input_output_processor import (
    QwenImageEditResizeStep,
    QwenImageInputsDynamicStep,
    QwenImageInpaintProcessImagesInputStep,
    QwenImageInpaintProcessImagesOutputStep,
)
from .decoders import QwenImageDecodeDynamicStep
from .denoise import (
    QwenImageControlNetLoopBeforeDenoiser,
    QwenImageDenoiseStep,
    QwenImageEditDenoiseStep,
    QwenImageInpaintDenoiseStep,
)
from .encoders import QwenImageEditTextEncoderStep, QwenImageTextEncoderStep, QwenImageVaeEncoderDynamicStep


logger = logging.get_logger(__name__)


class QwenImageInpaintVaeEncoderStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    """This step is used for processing image and mask inputs forinpainting tasks. It:
        - Processes and updates `image` and `mask_image`.
        - Creates `image_latents`.

    Components:
        image_processor (`InpaintProcessor`) [subfolder=]
        vae (`AutoencoderKLQwenImage`) [subfolder=]

    Inputs:
        image (`None`):
        mask_image (`None`):
        height (`None`):
        width (`None`):
        padding_mask_crop (`None`, optional):
        generator (`None`, optional):

    New Outputs:
        original_image (`Tensor`):
            The original image
        original_mask (`Tensor`):
            The original mask_imagge
        crop_coords (`List`):
            The crop coordinates to use for the preprocess/postprocess of the image and mask
        image_latents (`Tensor`):
            The latents representing the reference image
    """

    block_classes = [
        QwenImageInpaintProcessImagesInputStep,
        QwenImageVaeEncoderDynamicStep(input_name="image", output_name="image_latents", include_image_processor=False),  # encode
    ]

    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return (
            "This step is used for processing image and mask inputs for inpainting tasks. It:\n"
            " - Processes and updates `image` and `mask_image`.\n"
            " - Creates `image_latents`."
        )


class QwenImageInpaintPrepareLatentsStep(SequentialPipelineBlocks):
    model_name = "qwenimage"
    """This step prepares the latents/image_latents and mask inputs for the inpainting denoising step. It:
     - Patchify the image latents.
     - Add noise to the image latents to create the `latents` input for the denoiser.
     - Create the latents `mask` based on the processed `mask_image`.
     - Patchify the `mask` to match the shape of the image latents.

    Components:
        scheduler (`FlowMatchEulerDiscreteScheduler`) [subfolder=]

    Inputs:
        height (`None`, optional):
        width (`None`, optional):
        num_images_per_prompt (`None`, optional, defaults to 1):
        batch_size (`int`):
            Number of prompts, the final batch size of model inputs should be batch_size * num_images_per_prompt. Can be
            generated in input step.
        image_latents (`None`, optional):
        latents (`Tensor`):
            The initial random noised, can be generated in prepare latent step.
        timesteps (`Tensor`):
            The timesteps to use for the denoising process. Can be generated in set_timesteps step.
        mask_image (`Tensor`):
            The mask image to use for the inpainting process.

    Outputs:
        init_noise (`Tensor`):
            The initial random noised used for inpainting denoising.
        mask (`Tensor`):
          The mask latents to use for the inpainting process.
    """

    block_classes = [
        QwenImagePackLatentsDynamicStep("image_latents"),
        QwenImagePrepareLatentsWithStrengthStep,
        QwenImageCreateMaskLatentsStep,
        QwenImagePackLatentsDynamicStep("mask"),
    ]

    block_names = ["pack_image_latents", "add_noise_to_latents", "create_mask_latents", "pack_mask"]

    @property
    def description(self) -> str:
        return (
            "This step prepares the latents/image_latents and mask inputs for the inpainting denoising step. It:\n"
            " - Patchify the image latents.\n"
            " - Add noise to the image latents to create the latents input for the denoiser.\n"
            " - Create the latents `mask` based on the processedmask image.\n"
            " - Patchify the mask latents to match the shape of the image latents."
        )


class QwenImageControlnetBeforeDenoiseStep(SequentialPipelineBlocks):
    model_name = "qwenimage"

    block_classes = [
        QwenImagePackLatentsDynamicStep("control_image_latents"),  # pachify the latents
        QwenImageControlNetPrepareInputsStep,  # prepare the controlnet inputs e.g. controlnet_keep, controlnet_conditioning_scale, etc.
    ]

    block_names = ["pachify_control_image_latent", "prepare_controlnet_inputs"]

    @property
    def description(self) -> str:
        return "Step that prepares the controlnet inputs. Insert before the Denoise Step, after set_timesteps step."


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
        ("controlnet_vae_encoder",  QwenImageVaeEncoderDynamicStep(input_name="control_image", output_name="control_image_latents")),
        ("controlnet_before_denoise", QwenImageControlnetBeforeDenoiseStep()),
        (
            "controlnet_denoise_loop_before",
            QwenImageControlNetLoopBeforeDenoiser(),
        ),  # insert before the denoiseloop_denoiser
    ]
)


EDIT_BLOCKS = InsertableDict(
    [
        ("image_resize", QwenImageEditResizeStep()),
        ("text_encoder", QwenImageEditTextEncoderStep()),
        ("vae_encoder", QwenImageVaeEncoderDynamicStep(input_name="image", output_name="image_latents", do_resize=False)),
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
