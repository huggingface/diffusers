# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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
from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict, OutputParam
from .before_denoise import (
    Krea2ApplyStrengthStep,
    Krea2ImageInputsStep,
    Krea2PrepareImageLatentsStep,
    Krea2PrepareLatentsStep,
    Krea2PreparePositionIdsStep,
    Krea2PrepareReferencePositionIdsStep,
    Krea2ReferenceInputsStep,
    Krea2TurboSetTimestepsStep,
    Krea2TurboTextInputsStep,
)
from .denoise import Krea2TurboDenoiseStep, Krea2TurboInpaintDenoiseStep, Krea2TurboReferenceDenoiseStep
from .encoders import Krea2TurboReferenceTextEncoderStep, Krea2TurboTextEncoderStep
from .modular_blocks_krea2 import (
    Krea2AutoDecodeStep,
    Krea2AutoVaeEncoderStep,
    Krea2InpaintPrepareLatentsStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class Krea2TurboAutoTextEncoderStep(AutoPipelineBlocks):
    """
    Select text-only or reference-image-grounded Krea 2 Turbo prompt encoding.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. reference_image_processor
          (`Krea2ReferenceImageProcessor`): The Qwen3-VL processor used for image-grounded prompt encoding. tokenizer
          (`AutoTokenizer`): The tokenizer paired with the text encoder.

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          reference_image (`Image | list`, *optional*):
              First reference image(s), or scene reference for two-reference generation.
          reference_image_2 (`Image | list`, *optional*):
              Optional second reference image(s), used as the subject reference.
          reference_image_encoder_resolution (`int`, *optional*, defaults to 768):
              Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
    """

    model_name = "krea2"
    block_classes = [Krea2TurboReferenceTextEncoderStep, Krea2TurboTextEncoderStep]
    block_names = ["reference", "text"]
    block_trigger_inputs = ["reference_image", None]

    @property
    def description(self) -> str:
        return "Select text-only or reference-image-grounded Krea 2 Turbo prompt encoding."


CORE_DENOISE_BLOCKS = InsertableDict(
    [
        ("input", Krea2TurboTextInputsStep()),
        ("prepare_latents", Krea2PrepareLatentsStep()),
        ("set_timesteps", Krea2TurboSetTimestepsStep()),
        ("prepare_position_ids", Krea2PreparePositionIdsStep()),
        ("denoise", Krea2TurboDenoiseStep()),
    ]
)


# auto_docstring
class Krea2TurboCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for the distilled Krea 2 turbo text-to-image checkpoint: prepares the
    batch/latents/timesteps and the shared position ids, then runs the guidance-free denoising loop, producing the
    denoised packed latents for the decoder.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              The denoised packed latents (B, image_seq_len, in_channels).
    """

    model_name = "krea2"
    block_classes = list(CORE_DENOISE_BLOCKS.values())
    block_names = list(CORE_DENOISE_BLOCKS.keys())

    @property
    def description(self) -> str:
        return (
            "Core denoising workflow for the distilled Krea 2 turbo text-to-image checkpoint: prepares the "
            "batch/latents/timesteps and the shared position ids, then runs the guidance-free denoising loop, "
            "producing the denoised packed latents for the decoder."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents", description="The denoised packed latents (B, image_seq_len, in_channels).")
        ]


# auto_docstring
class Krea2TurboImg2ImgInputStep(SequentialPipelineBlocks):
    """
    Expand Krea 2 Turbo text and source-image inputs to the effective image-to-image batch.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The preprocessed inpainting mask.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          batch_size (`int`):
              Effective batch size (num prompts * num_images_per_prompt).
          dtype (`dtype`):
              The dtype of the text features.
          prompt_embeds (`Tensor`):
              Text features, batch-expanded.
          prompt_embeds_mask (`Tensor`):
              Text mask, batch-expanded.
          image_latents (`Tensor`):
              The latent representation of the input image.
          processed_mask_image (`Tensor`):
              The batch-expanded inpainting mask.
          height (`int`):
              The generation height inferred from the image.
          width (`int`):
              The generation width inferred from the image.
    """

    model_name = "krea2"
    block_classes = [Krea2TurboTextInputsStep, Krea2ImageInputsStep]
    block_names = ["text_inputs", "image_inputs"]

    @property
    def description(self) -> str:
        return "Expand Krea 2 Turbo text and source-image inputs to the effective image-to-image batch."


# auto_docstring
class Krea2TurboReferenceInputStep(SequentialPipelineBlocks):
    """
    Expand Krea 2 Turbo text and reference-image conditioning to the effective batch.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          reference_image_latents (`list`):
              Normalized reference-image latents from the VAE encoder in conditioning order.

      Outputs:
          batch_size (`int`):
              Effective batch size (num prompts * num_images_per_prompt).
          dtype (`dtype`):
              The dtype of the text features.
          prompt_embeds (`Tensor`):
              Text features, batch-expanded.
          prompt_embeds_mask (`Tensor`):
              Text mask, batch-expanded.
          reference_image_latents (`list`):
              Packed reference-image latents expanded to the effective batch in conditioning order.
    """

    model_name = "krea2"
    block_classes = [Krea2TurboTextInputsStep, Krea2ReferenceInputsStep]
    block_names = ["text_inputs", "reference_inputs"]

    @property
    def description(self) -> str:
        return "Expand Krea 2 Turbo text and reference-image conditioning to the effective batch."


# auto_docstring
class Krea2TurboReferenceCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Generate Krea 2 Turbo target latents from noise while attending to clean reference-image tokens.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          reference_image_latents (`list`):
              Normalized reference-image latents from the VAE encoder in conditioning order.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          reference_attention_scale (`float | list`, *optional*, defaults to 1.0):
              One multiplier for all references or one multiplier per reference in conditioning order.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TurboReferenceInputStep,
        Krea2PrepareLatentsStep,
        Krea2TurboSetTimestepsStep,
        Krea2PrepareReferencePositionIdsStep,
        Krea2TurboReferenceDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "prepare_position_ids", "denoise"]

    @property
    def description(self) -> str:
        return "Generate Krea 2 Turbo target latents from noise while attending to clean reference-image tokens."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2TurboImg2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Krea 2 Turbo image-to-image workflow with strength-adjusted denoising.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The preprocessed inpainting mask.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TurboImg2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2TurboSetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2PrepareImageLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2TurboDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "apply_strength",
        "prepare_image_latents",
        "prepare_position_ids",
        "denoise",
    ]

    @property
    def description(self) -> str:
        return "Core Krea 2 Turbo image-to-image workflow with strength-adjusted denoising."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2TurboInpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Krea 2 Turbo inpainting workflow with masked latent blending after every denoising step.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The preprocessed inpainting mask.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 8):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TurboImg2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2TurboSetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2InpaintPrepareLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2TurboInpaintDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "apply_strength",
        "prepare_inpaint_latents",
        "prepare_position_ids",
        "denoise",
    ]

    @property
    def description(self) -> str:
        return "Core Krea 2 Turbo inpainting workflow with masked latent blending after every denoising step."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2TurboAutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Select the Krea 2 Turbo text-to-image, image-to-image, or inpainting denoising workflow.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024 or None, depending on the workflow):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024 or None, depending on the workflow):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          reference_image_latents (`list`, *optional*):
              Normalized reference-image latents from the VAE encoder in conditioning order.
          reference_attention_scale (`float | list`, *optional*, defaults to 1.0):
              One multiplier for all references or one multiplier per reference in conditioning order.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The preprocessed inpainting mask.
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TurboCoreDenoiseStep,
        Krea2TurboReferenceCoreDenoiseStep,
        Krea2TurboInpaintCoreDenoiseStep,
        Krea2TurboImg2ImgCoreDenoiseStep,
    ]
    block_names = ["text2image", "reference", "inpaint", "img2img"]
    block_trigger_inputs = ["reference_image_latents", "processed_mask_image", "image_latents"]
    default_block_name = "text2image"

    def select_block(self, reference_image_latents=None, processed_mask_image=None, image_latents=None):
        if reference_image_latents is not None:
            return "reference"
        if processed_mask_image is not None:
            return "inpaint"
        if image_latents is not None:
            return "img2img"
        return "text2image"

    @property
    def description(self) -> str:
        return "Select the Krea 2 Turbo text-to-image, image-to-image, or inpainting denoising workflow."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2TurboAutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image, image-to-image, and inpainting using the distilled Krea 2 Turbo
    checkpoint.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `prompt`, `image`
        - `inpainting`: requires `prompt`, `image`, `mask_image`
        - `reference`: requires `prompt`, `reference_image`

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. reference_image_processor
          (`Krea2ReferenceImageProcessor`): The Qwen3-VL processor used for image-grounded prompt encoding. tokenizer
          (`AutoTokenizer`): The tokenizer paired with the text encoder. image_processor (`VaeImageProcessor`) vae
          (`AutoencoderKLQwenImage`) image_mask_processor (`InpaintProcessor`) transformer (`Krea2Transformer2DModel`)
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          reference_image (`Image | list`, *optional*):
              First reference image(s), or scene reference for two-reference generation.
          reference_image_2 (`Image | list`, *optional*):
              Optional second reference image(s), used as the subject reference.
          reference_image_encoder_resolution (`int`, *optional*, defaults to 768):
              Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.
          height (`int`, *optional*, defaults to 1024 or None, depending on the workflow):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024 or None, depending on the workflow):
              The width in pixels of the generated image.
          image (`Image | list`, *optional*):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`, *optional*):
              Mask image for inpainting.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sigmas (`list`, *optional*):
              Custom sigma schedule (defaults to a linear ramp).
          attention_kwargs (`dict`, *optional*):
              Additional kwargs for attention processors.
          reference_image_latents (`list`, *optional*):
              Normalized reference-image latents from the VAE encoder in conditioning order.
          reference_attention_scale (`float | list`, *optional*, defaults to 1.0):
              One multiplier for all references or one multiplier per reference in conditioning order.
          image_latents (`Tensor`, *optional*):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          processed_mask_image (`Tensor`, *optional*):
              The preprocessed inpainting mask.
          strength (`float`, *optional*, defaults to 0.9):
              Strength for img2img/inpainting.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          mask_overlay_kwargs (`dict`, *optional*):
              Arguments used to overlay a cropped inpainting result on the original image.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "krea2"
    block_classes = [
        Krea2TurboAutoTextEncoderStep,
        Krea2AutoVaeEncoderStep,
        Krea2TurboAutoCoreDenoiseStep,
        Krea2AutoDecodeStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]

    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"prompt": True, "image": True},
        "inpainting": {"prompt": True, "image": True, "mask_image": True},
        "reference": {"prompt": True, "reference_image": True},
    }

    @property
    def description(self) -> str:
        return (
            "Auto Modular pipeline for text-to-image, image-to-image, and inpainting using the distilled Krea 2 "
            "Turbo checkpoint."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]
