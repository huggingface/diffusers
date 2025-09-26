# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import (
    StableDiffusionXLControlNetInputStep,
    StableDiffusionXLControlNetUnionInputStep,
    StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    StableDiffusionXLImg2ImgPrepareLatentsStep,
    StableDiffusionXLImg2ImgSetTimestepsStep,
    StableDiffusionXLInpaintPrepareLatentsStep,
    StableDiffusionXLInputStep,
    StableDiffusionXLPrepareAdditionalConditioningStep,
    StableDiffusionXLPrepareLatentsStep,
    StableDiffusionXLSetTimestepsStep,
)
from .decoders import (
    StableDiffusionXLDecodeStep,
    StableDiffusionXLInpaintOverlayMaskStep,
)
from .denoise import (
    StableDiffusionXLControlNetDenoiseStep,
    StableDiffusionXLDenoiseStep,
    StableDiffusionXLInpaintControlNetDenoiseStep,
    StableDiffusionXLInpaintDenoiseStep,
)
from .encoders import (
    StableDiffusionXLInpaintVaeEncoderStep,
    StableDiffusionXLIPAdapterStep,
    StableDiffusionXLTextEncoderStep,
    StableDiffusionXLVaeEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto blocks & sequential blocks & mappings


# vae encoder (run before before_denoise)
class StableDiffusionXLAutoVaeEncoderStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLInpaintVaeEncoderStep, StableDiffusionXLVaeEncoderStep]
    block_names = ["inpaint", "img2img"]
    block_trigger_inputs = ["mask_image", "image"]

    @property
    def description(self):
        return (
            "Vae encoder step that encode the image inputs into their latent representations.\n"
            + "This is an auto pipeline block that works for both inpainting and img2img tasks.\n"
            + " - `StableDiffusionXLInpaintVaeEncoderStep` (inpaint) is used when `mask_image` is provided.\n"
            + " - `StableDiffusionXLVaeEncoderStep` (img2img) is used when only `image` is provided."
            + " - if neither `mask_image` nor `image` is provided, step will be skipped."
        )


# optional ip-adapter (run before input step)
class StableDiffusionXLAutoIPAdapterStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLIPAdapterStep]
    block_names = ["ip_adapter"]
    block_trigger_inputs = ["ip_adapter_image"]

    @property
    def description(self):
        return "Run IP Adapter step if `ip_adapter_image` is provided. This step should be placed before the 'input' step.\n"


# before_denoise: text2img
class StableDiffusionXLBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLSetTimestepsStep,
        StableDiffusionXLPrepareLatentsStep,
        StableDiffusionXLPrepareAdditionalConditioningStep,
    ]
    block_names = ["set_timesteps", "prepare_latents", "prepare_add_cond"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `StableDiffusionXLSetTimestepsStep` is used to set the timesteps\n"
            + " - `StableDiffusionXLPrepareLatentsStep` is used to prepare the latents\n"
            + " - `StableDiffusionXLPrepareAdditionalConditioningStep` is used to prepare the additional conditioning\n"
        )


# before_denoise: img2img
class StableDiffusionXLImg2ImgBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLImg2ImgSetTimestepsStep,
        StableDiffusionXLImg2ImgPrepareLatentsStep,
        StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    ]
    block_names = ["set_timesteps", "prepare_latents", "prepare_add_cond"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for img2img task.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `StableDiffusionXLImg2ImgSetTimestepsStep` is used to set the timesteps\n"
            + " - `StableDiffusionXLImg2ImgPrepareLatentsStep` is used to prepare the latents\n"
            + " - `StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep` is used to prepare the additional conditioning\n"
        )


# before_denoise: inpainting
class StableDiffusionXLInpaintBeforeDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLImg2ImgSetTimestepsStep,
        StableDiffusionXLInpaintPrepareLatentsStep,
        StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep,
    ]
    block_names = ["set_timesteps", "prepare_latents", "prepare_add_cond"]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step for inpainting task.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `StableDiffusionXLImg2ImgSetTimestepsStep` is used to set the timesteps\n"
            + " - `StableDiffusionXLInpaintPrepareLatentsStep` is used to prepare the latents\n"
            + " - `StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep` is used to prepare the additional conditioning\n"
        )


# before_denoise: all task (text2img, img2img, inpainting)
class StableDiffusionXLAutoBeforeDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        StableDiffusionXLInpaintBeforeDenoiseStep,
        StableDiffusionXLImg2ImgBeforeDenoiseStep,
        StableDiffusionXLBeforeDenoiseStep,
    ]
    block_names = ["inpaint", "img2img", "text2img"]
    block_trigger_inputs = ["mask", "image_latents", None]

    @property
    def description(self):
        return (
            "Before denoise step that prepare the inputs for the denoise step.\n"
            + "This is an auto pipeline block that works for text2img, img2img and inpainting tasks as well as controlnet, controlnet_union.\n"
            + " - `StableDiffusionXLInpaintBeforeDenoiseStep` (inpaint) is used when both `mask` and `image_latents` are provided.\n"
            + " - `StableDiffusionXLImg2ImgBeforeDenoiseStep` (img2img) is used when only `image_latents` is provided.\n"
            + " - `StableDiffusionXLBeforeDenoiseStep` (text2img) is used when both `image_latents` and `mask` are not provided.\n"
        )


# optional controlnet input step (after before_denoise, before denoise)
# works for both controlnet and controlnet_union
class StableDiffusionXLAutoControlNetInputStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLControlNetUnionInputStep, StableDiffusionXLControlNetInputStep]
    block_names = ["controlnet_union", "controlnet"]
    block_trigger_inputs = ["control_mode", "control_image"]

    @property
    def description(self):
        return (
            "Controlnet Input step that prepare the controlnet input.\n"
            + "This is an auto pipeline block that works for both controlnet and controlnet_union.\n"
            + " (it should be called right before the denoise step)"
            + " - `StableDiffusionXLControlNetUnionInputStep` is called to prepare the controlnet input when `control_mode` and `control_image` are provided.\n"
            + " - `StableDiffusionXLControlNetInputStep` is called to prepare the controlnet input when `control_image` is provided."
            + " - if neither `control_mode` nor `control_image` is provided, step will be skipped."
        )


# denoise: controlnet (text2img, img2img, inpainting)
class StableDiffusionXLAutoControlNetDenoiseStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLInpaintControlNetDenoiseStep, StableDiffusionXLControlNetDenoiseStep]
    block_names = ["inpaint_controlnet_denoise", "controlnet_denoise"]
    block_trigger_inputs = ["mask", "controlnet_cond"]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents with controlnet. "
            "This is a auto pipeline block that using controlnet for text2img, img2img and inpainting tasks."
            "This block should not be used without a controlnet_cond input"
            " - `StableDiffusionXLInpaintControlNetDenoiseStep` (inpaint_controlnet_denoise) is used when mask is provided."
            " - `StableDiffusionXLControlNetDenoiseStep` (controlnet_denoise) is used when mask is not provided but controlnet_cond is provided."
            " - If neither mask nor controlnet_cond are provided, step will be skipped."
        )


# denoise: all task with or without controlnet (text2img, img2img, inpainting)
class StableDiffusionXLAutoDenoiseStep(AutoPipelineBlocks):
    block_classes = [
        StableDiffusionXLAutoControlNetDenoiseStep,
        StableDiffusionXLInpaintDenoiseStep,
        StableDiffusionXLDenoiseStep,
    ]
    block_names = ["controlnet_denoise", "inpaint_denoise", "denoise"]
    block_trigger_inputs = ["controlnet_cond", "mask", None]

    @property
    def description(self) -> str:
        return (
            "Denoise step that iteratively denoise the latents. "
            "This is a auto pipeline block that works for text2img, img2img and inpainting tasks. And can be used with or without controlnet."
            " - `StableDiffusionXLAutoControlNetDenoiseStep` (controlnet_denoise) is used when controlnet_cond is provided (support controlnet withtext2img, img2img and inpainting tasks)."
            " - `StableDiffusionXLInpaintDenoiseStep` (inpaint_denoise) is used when mask is provided (support inpainting tasks)."
            " - `StableDiffusionXLDenoiseStep` (denoise) is used when neither mask nor controlnet_cond are provided (support text2img and img2img tasks)."
        )


# decode: inpaint
class StableDiffusionXLInpaintDecodeStep(SequentialPipelineBlocks):
    block_classes = [StableDiffusionXLDecodeStep, StableDiffusionXLInpaintOverlayMaskStep]
    block_names = ["decode", "mask_overlay"]

    @property
    def description(self):
        return (
            "Inpaint decode step that decode the denoised latents into images outputs.\n"
            + "This is a sequential pipeline blocks:\n"
            + " - `StableDiffusionXLDecodeStep` is used to decode the denoised latents into images\n"
            + " - `StableDiffusionXLInpaintOverlayMaskStep` is used to overlay the mask on the image"
        )


# decode: all task (text2img, img2img, inpainting)
class StableDiffusionXLAutoDecodeStep(AutoPipelineBlocks):
    block_classes = [StableDiffusionXLInpaintDecodeStep, StableDiffusionXLDecodeStep]
    block_names = ["inpaint", "non-inpaint"]
    block_trigger_inputs = ["padding_mask_crop", None]

    @property
    def description(self):
        return (
            "Decode step that decode the denoised latents into images outputs.\n"
            + "This is an auto pipeline block that works for inpainting and non-inpainting tasks.\n"
            + " - `StableDiffusionXLInpaintDecodeStep` (inpaint) is used when `padding_mask_crop` is provided.\n"
            + " - `StableDiffusionXLDecodeStep` (non-inpaint) is used when `padding_mask_crop` is not provided."
        )


class StableDiffusionXLCoreDenoiseStep(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLInputStep,
        StableDiffusionXLAutoBeforeDenoiseStep,
        StableDiffusionXLAutoControlNetInputStep,
        StableDiffusionXLAutoDenoiseStep,
    ]
    block_names = ["input", "before_denoise", "controlnet_input", "denoise"]

    @property
    def description(self):
        return (
            "Core step that performs the denoising process. \n"
            + " - `StableDiffusionXLInputStep` (input) standardizes the inputs for the denoising step.\n"
            + " - `StableDiffusionXLAutoBeforeDenoiseStep` (before_denoise) prepares the inputs for the denoising step.\n"
            + " - `StableDiffusionXLAutoControlNetInputStep` (controlnet_input) prepares the controlnet input.\n"
            + " - `StableDiffusionXLAutoDenoiseStep` (denoise) iteratively denoises the latents.\n\n"
            + "This step support text-to-image, image-to-image, inpainting, with or without controlnet/controlnet_union/ip_adapter for Stable Diffusion XL:\n"
            + "- for image-to-image generation, you need to provide `image_latents`\n"
            + "- for inpainting, you need to provide `mask_image` and `image_latents`\n"
            + "- to run the controlnet workflow, you need to provide `control_image`\n"
            + "- to run the controlnet_union workflow, you need to provide `control_image` and `control_mode`\n"
            + "- to run the ip_adapter workflow, you need to load ip_adapter into your unet and provide `ip_adapter_embeds`\n"
            + "- for text-to-image generation, all you need to provide is prompt embeddings\n"
        )


# ip-adapter, controlnet, text2img, img2img, inpainting
class StableDiffusionXLAutoBlocks(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLTextEncoderStep,
        StableDiffusionXLAutoIPAdapterStep,
        StableDiffusionXLAutoVaeEncoderStep,
        StableDiffusionXLCoreDenoiseStep,
        StableDiffusionXLAutoDecodeStep,
    ]
    block_names = [
        "text_encoder",
        "ip_adapter",
        "vae_encoder",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using Stable Diffusion XL.\n"
            + "- for image-to-image generation, you need to provide either `image` or `image_latents`\n"
            + "- for inpainting, you need to provide `mask_image` and `image`, optionally you can provide `padding_mask_crop` \n"
            + "- to run the controlnet workflow, you need to provide `control_image`\n"
            + "- to run the controlnet_union workflow, you need to provide `control_image` and `control_mode`\n"
            + "- to run the ip_adapter workflow, you need to provide `ip_adapter_image`\n"
            + "- for text-to-image generation, all you need to provide is `prompt`"
        )


# controlnet (input + denoise step)
class StableDiffusionXLAutoControlnetStep(SequentialPipelineBlocks):
    block_classes = [
        StableDiffusionXLAutoControlNetInputStep,
        StableDiffusionXLAutoControlNetDenoiseStep,
    ]
    block_names = ["controlnet_input", "controlnet_denoise"]

    @property
    def description(self):
        return (
            "Controlnet auto step that prepare the controlnet input and denoise the latents. "
            + "It works for both controlnet and controlnet_union and supports text2img, img2img and inpainting tasks."
            + " (it should be replace at 'denoise' step)"
        )


TEXT2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", StableDiffusionXLTextEncoderStep),
        ("input", StableDiffusionXLInputStep),
        ("set_timesteps", StableDiffusionXLSetTimestepsStep),
        ("prepare_latents", StableDiffusionXLPrepareLatentsStep),
        ("prepare_add_cond", StableDiffusionXLPrepareAdditionalConditioningStep),
        ("denoise", StableDiffusionXLDenoiseStep),
        ("decode", StableDiffusionXLDecodeStep),
    ]
)

IMAGE2IMAGE_BLOCKS = InsertableDict(
    [
        ("text_encoder", StableDiffusionXLTextEncoderStep),
        ("vae_encoder", StableDiffusionXLVaeEncoderStep),
        ("input", StableDiffusionXLInputStep),
        ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
        ("prepare_latents", StableDiffusionXLImg2ImgPrepareLatentsStep),
        ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
        ("denoise", StableDiffusionXLDenoiseStep),
        ("decode", StableDiffusionXLDecodeStep),
    ]
)

INPAINT_BLOCKS = InsertableDict(
    [
        ("text_encoder", StableDiffusionXLTextEncoderStep),
        ("vae_encoder", StableDiffusionXLInpaintVaeEncoderStep),
        ("input", StableDiffusionXLInputStep),
        ("set_timesteps", StableDiffusionXLImg2ImgSetTimestepsStep),
        ("prepare_latents", StableDiffusionXLInpaintPrepareLatentsStep),
        ("prepare_add_cond", StableDiffusionXLImg2ImgPrepareAdditionalConditioningStep),
        ("denoise", StableDiffusionXLInpaintDenoiseStep),
        ("decode", StableDiffusionXLInpaintDecodeStep),
    ]
)

CONTROLNET_BLOCKS = InsertableDict(
    [
        ("denoise", StableDiffusionXLAutoControlnetStep),
    ]
)


IP_ADAPTER_BLOCKS = InsertableDict(
    [
        ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
    ]
)

AUTO_BLOCKS = InsertableDict(
    [
        ("text_encoder", StableDiffusionXLTextEncoderStep),
        ("ip_adapter", StableDiffusionXLAutoIPAdapterStep),
        ("vae_encoder", StableDiffusionXLAutoVaeEncoderStep),
        ("denoise", StableDiffusionXLCoreDenoiseStep),
        ("decode", StableDiffusionXLAutoDecodeStep),
    ]
)


ALL_BLOCKS = {
    "text2img": TEXT2IMAGE_BLOCKS,
    "img2img": IMAGE2IMAGE_BLOCKS,
    "inpaint": INPAINT_BLOCKS,
    "controlnet": CONTROLNET_BLOCKS,
    "ip_adapter": IP_ADAPTER_BLOCKS,
    "auto": AUTO_BLOCKS,
}
