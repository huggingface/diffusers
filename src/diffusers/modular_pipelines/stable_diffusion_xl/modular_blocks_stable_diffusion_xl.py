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
from ..modular_pipeline_utils import OutputParam
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
# auto_docstring
class StableDiffusionXLAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using Stable Diffusion
    XL.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `image`, `prompt`
        - `inpainting`: requires `mask_image`, `image`, `prompt`
        - `controlnet_text2image`: requires `control_image`, `prompt`
        - `controlnet_image2image`: requires `control_image`, `image`, `prompt`
        - `controlnet_inpainting`: requires `control_image`, `mask_image`, `image`, `prompt`
        - `controlnet_union_text2image`: requires `control_image`, `control_mode`, `prompt`
        - `controlnet_union_image2image`: requires `control_image`, `control_mode`, `image`, `prompt`
        - `controlnet_union_inpainting`: requires `control_image`, `control_mode`, `mask_image`, `image`, `prompt`
        - `ip_adapter_text2image`: requires `ip_adapter_image`, `prompt`
        - `ip_adapter_image2image`: requires `ip_adapter_image`, `image`, `prompt`
        - `ip_adapter_inpainting`: requires `ip_adapter_image`, `mask_image`, `image`, `prompt`
        - `ip_adapter_controlnet_text2image`: requires `ip_adapter_image`, `control_image`, `prompt`
        - `ip_adapter_controlnet_image2image`: requires `ip_adapter_image`, `control_image`, `image`, `prompt`
        - `ip_adapter_controlnet_inpainting`: requires `ip_adapter_image`, `control_image`, `mask_image`, `image`,
          `prompt`
        - `ip_adapter_controlnet_union_text2image`: requires `ip_adapter_image`, `control_image`, `control_mode`,
          `prompt`
        - `ip_adapter_controlnet_union_image2image`: requires `ip_adapter_image`, `control_image`, `control_mode`,
          `image`, `prompt`
        - `ip_adapter_controlnet_union_inpainting`: requires `ip_adapter_image`, `control_image`, `control_mode`,
          `mask_image`, `image`, `prompt`

      Components:
          text_encoder (`CLIPTextModel`) text_encoder_2 (`CLIPTextModelWithProjection`) tokenizer (`CLIPTokenizer`)
          tokenizer_2 (`CLIPTokenizer`) guider (`ClassifierFreeGuidance`) image_encoder
          (`CLIPVisionModelWithProjection`) feature_extractor (`CLIPImageProcessor`) unet (`UNet2DConditionModel`) vae
          (`AutoencoderKL`) image_processor (`VaeImageProcessor`) mask_processor (`VaeImageProcessor`) scheduler
          (`EulerDiscreteScheduler`) controlnet (`ControlNetUnionModel`) control_image_processor (`VaeImageProcessor`)

      Configs:
          force_zeros_for_empty_prompt (default: True) requires_aesthetics_score (default: False)

      Inputs:
          prompt (`None`, *optional*):
              TODO: Add description.
          prompt_2 (`None`, *optional*):
              TODO: Add description.
          negative_prompt (`None`, *optional*):
              TODO: Add description.
          negative_prompt_2 (`None`, *optional*):
              TODO: Add description.
          cross_attention_kwargs (`None`, *optional*):
              TODO: Add description.
          clip_skip (`None`, *optional*):
              TODO: Add description.
          ip_adapter_image (`Image | ndarray | Tensor | list | list | list`, *optional*):
              The image(s) to be used as ip adapter
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          image (`None`, *optional*):
              TODO: Add description.
          mask_image (`None`, *optional*):
              TODO: Add description.
          padding_mask_crop (`None`, *optional*):
              TODO: Add description.
          dtype (`dtype`, *optional*):
              The dtype of the model inputs
          generator (`None`, *optional*):
              TODO: Add description.
          preprocess_kwargs (`dict | NoneType`, *optional*):
              A kwargs dictionary that if specified is passed along to the `ImageProcessor` as defined under
              `self.image_processor` in [diffusers.image_processor.VaeImageProcessor]
          num_images_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          ip_adapter_embeds (`list`, *optional*):
              Pre-generated image embeddings for IP-Adapter. Can be generated from ip_adapter step.
          negative_ip_adapter_embeds (`list`, *optional*):
              Pre-generated negative image embeddings for IP-Adapter. Can be generated from ip_adapter step.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          denoising_end (`None`, *optional*):
              TODO: Add description.
          strength (`None`, *optional*, defaults to 0.3):
              TODO: Add description.
          denoising_start (`None`, *optional*):
              TODO: Add description.
          latents (`None`):
              TODO: Add description.
          image_latents (`Tensor`, *optional*):
              The latents representing the reference image for image-to-image/inpainting generation. Can be generated
              in vae_encode step.
          mask (`Tensor`, *optional*):
              The mask for the inpainting generation. Can be generated in vae_encode step.
          masked_image_latents (`Tensor`, *optional*):
              The masked image latents for the inpainting generation (only for inpainting-specific unet). Can be
              generated in vae_encode step.
          original_size (`None`, *optional*):
              TODO: Add description.
          target_size (`None`, *optional*):
              TODO: Add description.
          negative_original_size (`None`, *optional*):
              TODO: Add description.
          negative_target_size (`None`, *optional*):
              TODO: Add description.
          crops_coords_top_left (`None`, *optional*, defaults to (0, 0)):
              TODO: Add description.
          negative_crops_coords_top_left (`None`, *optional*, defaults to (0, 0)):
              TODO: Add description.
          aesthetic_score (`None`, *optional*, defaults to 6.0):
              TODO: Add description.
          negative_aesthetic_score (`None`, *optional*, defaults to 2.0):
              TODO: Add description.
          control_image (`None`, *optional*):
              TODO: Add description.
          control_mode (`None`, *optional*):
              TODO: Add description.
          control_guidance_start (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          control_guidance_end (`None`, *optional*, defaults to 1.0):
              TODO: Add description.
          controlnet_conditioning_scale (`None`, *optional*, defaults to 1.0):
              TODO: Add description.
          guess_mode (`None`, *optional*, defaults to False):
              TODO: Add description.
          crops_coords (`tuple | NoneType`, *optional*):
              The crop coordinates to use for preprocess/postprocess the image and mask, for inpainting task only. Can
              be generated in vae_encode step.
          controlnet_cond (`Tensor`, *optional*):
              The control image to use for the denoising process. Can be generated in prepare_controlnet_inputs step.
          conditioning_scale (`float`, *optional*):
              The controlnet conditioning scale value to use for the denoising process. Can be generated in
              prepare_controlnet_inputs step.
          controlnet_keep (`list`, *optional*):
              The controlnet keep values to use for the denoising process. Can be generated in
              prepare_controlnet_inputs step.
          **denoiser_input_fields (`None`, *optional*):
              All conditional model inputs that need to be prepared with guider. It should contain
              prompt_embeds/negative_prompt_embeds, add_time_ids/negative_add_time_ids,
              pooled_prompt_embeds/negative_pooled_prompt_embeds, and ip_adapter_embeds/negative_ip_adapter_embeds
              (optional).please add `kwargs_type=denoiser_input_fields` to their parameter spec (`OutputParam`) when
              they are created and added to the pipeline state
          eta (`None`, *optional*, defaults to 0.0):
              TODO: Add description.
          output_type (`None`, *optional*, defaults to pil):
              TODO: Add description.

      Outputs:
          images (`list`):
              Generated images.
    """

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

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"image": True, "prompt": True},
        "inpainting": {"mask_image": True, "image": True, "prompt": True},
        "controlnet_text2image": {"control_image": True, "prompt": True},
        "controlnet_image2image": {"control_image": True, "image": True, "prompt": True},
        "controlnet_inpainting": {"control_image": True, "mask_image": True, "image": True, "prompt": True},
        "controlnet_union_text2image": {"control_image": True, "control_mode": True, "prompt": True},
        "controlnet_union_image2image": {"control_image": True, "control_mode": True, "image": True, "prompt": True},
        "controlnet_union_inpainting": {
            "control_image": True,
            "control_mode": True,
            "mask_image": True,
            "image": True,
            "prompt": True,
        },
        "ip_adapter_text2image": {"ip_adapter_image": True, "prompt": True},
        "ip_adapter_image2image": {"ip_adapter_image": True, "image": True, "prompt": True},
        "ip_adapter_inpainting": {"ip_adapter_image": True, "mask_image": True, "image": True, "prompt": True},
        "ip_adapter_controlnet_text2image": {"ip_adapter_image": True, "control_image": True, "prompt": True},
        "ip_adapter_controlnet_image2image": {
            "ip_adapter_image": True,
            "control_image": True,
            "image": True,
            "prompt": True,
        },
        "ip_adapter_controlnet_inpainting": {
            "ip_adapter_image": True,
            "control_image": True,
            "mask_image": True,
            "image": True,
            "prompt": True,
        },
        "ip_adapter_controlnet_union_text2image": {
            "ip_adapter_image": True,
            "control_image": True,
            "control_mode": True,
            "prompt": True,
        },
        "ip_adapter_controlnet_union_image2image": {
            "ip_adapter_image": True,
            "control_image": True,
            "control_mode": True,
            "image": True,
            "prompt": True,
        },
        "ip_adapter_controlnet_union_inpainting": {
            "ip_adapter_image": True,
            "control_image": True,
            "control_mode": True,
            "mask_image": True,
            "image": True,
            "prompt": True,
        },
    }

    @property
    def description(self):
        return "Auto Modular pipeline for text-to-image, image-to-image, inpainting, and controlnet tasks using Stable Diffusion XL."

    @property
    def outputs(self):
        return [OutputParam.template("images")]
