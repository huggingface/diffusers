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
    Krea2PrepareMaskLatentsStep,
    Krea2PreparePositionIdsStep,
    Krea2PrepareReferencePositionIdsStep,
    Krea2ReferenceInputsStep,
    Krea2SetTimestepsStep,
    Krea2TextInputsStep,
)
from .decoders import Krea2DecodeStep, Krea2InpaintDecodeStep
from .denoise import Krea2DenoiseStep, Krea2InpaintDenoiseStep, Krea2ReferenceDenoiseStep
from .encoders import (
    Krea2InpaintProcessImagesInputStep,
    Krea2ProcessImagesInputStep,
    Krea2ReferenceProcessImagesInputStep,
    Krea2ReferenceTextEncoderStep,
    Krea2ReferenceVaeEncoderStep,
    Krea2TextEncoderStep,
    Krea2VaeEncoderStep,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# auto_docstring
class Krea2AutoTextEncoderStep(AutoPipelineBlocks):
    """
    Select text-only or reference-image-grounded Krea 2 prompt encoding.

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. reference_image_processor
          (`Krea2ReferenceImageProcessor`): The Qwen3-VL processor used for image-grounded prompt encoding. tokenizer
          (`AutoTokenizer`): The tokenizer paired with the text encoder. guider (`ClassifierFreeGuidance`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The negative prompt(s) for CFG.
          reference_image (`Image | list`, *optional*):
              A reference image or ordered list of reference images shared by all prompts in the batch.
          reference_image_encoder_resolution (`int`, *optional*, defaults to 768):
              Maximum reference-image side length used by the Qwen3-VL encoder. Use 0 for native resolution.
          max_sequence_length (`int`, *optional*, defaults to 512):
              Maximum sequence length for prompt encoding.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          prompt_embeds_mask (`Tensor`):
              The encoder attention mask.
          negative_prompt_embeds (`Tensor`):
              The negative prompt embeddings.
          negative_prompt_embeds_mask (`Tensor`):
              The negative prompt embeddings mask.
    """

    model_name = "krea2"
    block_classes = [Krea2ReferenceTextEncoderStep, Krea2TextEncoderStep]
    block_names = ["reference", "text"]
    block_trigger_inputs = ["reference_image", None]

    @property
    def description(self) -> str:
        return "Select text-only or reference-image-grounded Krea 2 prompt encoding."


CORE_DENOISE_BLOCKS = InsertableDict(
    [
        ("input", Krea2TextInputsStep()),
        ("prepare_latents", Krea2PrepareLatentsStep()),
        ("set_timesteps", Krea2SetTimestepsStep()),
        ("prepare_position_ids", Krea2PreparePositionIdsStep()),
        ("denoise", Krea2DenoiseStep()),
    ]
)


# auto_docstring
class Krea2CoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoising workflow for Krea 2 text-to-image: prepares the batch/latents/timesteps and the shared position ids,
    then runs the symmetric-CFG denoising loop, producing the denoised packed latents for the decoder.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
          latents (`Tensor`, *optional*):
              Pre-generated noisy latents for image generation.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*, defaults to 28):
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
            "Core denoising workflow for Krea 2 text-to-image: prepares the batch/latents/timesteps and the shared "
            "position ids, then runs the symmetric-CFG denoising loop, producing the denoised packed latents for the "
            "decoder."
        )

    @property
    def outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents", description="The denoised packed latents (B, image_seq_len, in_channels).")
        ]


# auto_docstring
class Krea2Img2ImgVaeEncoderStep(SequentialPipelineBlocks):
    """
    Preprocess and VAE-encode an image for Krea 2 image-to-image generation.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.

      Outputs:
          processed_image (`Tensor`):
              The preprocessed input image.
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "krea2"
    block_classes = [Krea2ProcessImagesInputStep, Krea2VaeEncoderStep]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Preprocess and VAE-encode an image for Krea 2 image-to-image generation."


# auto_docstring
class Krea2InpaintVaeEncoderStep(SequentialPipelineBlocks):
    """
    Preprocess an image and mask and VAE-encode the image for Krea 2 inpainting.

      Components:
          image_mask_processor (`InpaintProcessor`) vae (`AutoencoderKLQwenImage`)

      Inputs:
          image (`Image | list`):
              Reference image(s) for denoising. Can be a single image or list of images.
          mask_image (`Image`):
              Mask image for inpainting.
          height (`int`, *optional*):
              The height in pixels of the generated image.
          width (`int`, *optional*):
              The width in pixels of the generated image.
          padding_mask_crop (`int`, *optional*):
              Padding for mask cropping in inpainting.

      Outputs:
          processed_image (`Tensor`):
              The preprocessed input image.
          processed_mask_image (`Tensor`):
              The preprocessed inpainting mask.
          mask_overlay_kwargs (`dict`):
              Arguments used to overlay a cropped inpainting result on the original image.
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "krea2"
    block_classes = [Krea2InpaintProcessImagesInputStep, Krea2VaeEncoderStep]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Preprocess an image and mask and VAE-encode the image for Krea 2 inpainting."


# auto_docstring
class Krea2ReferenceVaeEncoderBlocks(SequentialPipelineBlocks):
    """
    Preprocess and VAE-encode an image for reference-conditioned Krea 2 generation.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage`)

      Inputs:
          reference_image (`Image | list`):
              A reference image or ordered list of reference images shared by all prompts in the batch.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.

      Outputs:
          processed_reference_images (`list`):
              Reference images resized and normalized for VAE encoding in conditioning order.
          reference_image_latents (`list`):
              Normalized latent representations of the reference images in conditioning order.
    """

    model_name = "krea2"
    block_classes = [Krea2ReferenceProcessImagesInputStep, Krea2ReferenceVaeEncoderStep]
    block_names = ["preprocess", "encode"]

    @property
    def description(self) -> str:
        return "Preprocess and VAE-encode an image for reference-conditioned Krea 2 generation."


# auto_docstring
class Krea2AutoVaeEncoderStep(AutoPipelineBlocks):
    """
    Select the Krea 2 inpainting or image-to-image VAE encoder from the provided image inputs.

      Components:
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage`) image_mask_processor
          (`InpaintProcessor`)

      Inputs:
          reference_image (`Image | list`, *optional*):
              A reference image or ordered list of reference images shared by all prompts in the batch.
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

      Outputs:
          processed_reference_images (`list`):
              Reference images resized and normalized for VAE encoding in conditioning order.
          reference_image_latents (`list`):
              Normalized latent representations of the reference images in conditioning order.
          processed_image (`Tensor`):
              The preprocessed input image.
          processed_mask_image (`Tensor`):
              The preprocessed inpainting mask.
          mask_overlay_kwargs (`dict`):
              Arguments used to overlay a cropped inpainting result on the original image.
          image_latents (`Tensor`):
              The latent representation of the input image.
    """

    model_name = "krea2"
    block_classes = [Krea2ReferenceVaeEncoderBlocks, Krea2InpaintVaeEncoderStep, Krea2Img2ImgVaeEncoderStep]
    block_names = ["reference", "inpaint", "img2img"]
    block_trigger_inputs = ["reference_image", "mask_image", "image"]

    @property
    def description(self) -> str:
        return "Select the Krea 2 inpainting or image-to-image VAE encoder from the provided image inputs."


# auto_docstring
class Krea2Img2ImgInputStep(SequentialPipelineBlocks):
    """
    Expand Krea 2 text and source-image inputs to the effective image-to-image batch.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
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
          negative_prompt_embeds (`Tensor`):
              Negative text features, batch-expanded.
          negative_prompt_embeds_mask (`Tensor`):
              Negative text mask, batch-expanded.
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
    block_classes = [Krea2TextInputsStep, Krea2ImageInputsStep]
    block_names = ["text_inputs", "image_inputs"]

    @property
    def description(self) -> str:
        return "Expand Krea 2 text and source-image inputs to the effective image-to-image batch."


# auto_docstring
class Krea2InpaintPrepareLatentsStep(SequentialPipelineBlocks):
    """
    Add noise to Krea 2 source-image latents and prepare packed mask latents for inpainting.

      Components:
          scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          image_latents (`Tensor`):
              image latents used to guide the image generation. Can be generated from vae_encoder step.
          timesteps (`Tensor`):
              The selected denoising timesteps.
          processed_mask_image (`Tensor`):
              The preprocessed inpainting mask.
          height (`int`):
              The height in pixels of the generated image.
          width (`int`):
              The width in pixels of the generated image.
          dtype (`dtype`, *optional*, defaults to torch.float32):
              The dtype of the model inputs, can be generated in input step.

      Outputs:
          initial_noise (`Tensor`):
              The sampled initial noise.
          latents (`Tensor`):
              Denoised latents.
          mask (`Tensor`):
              The packed latent-space mask.
    """

    model_name = "krea2"
    block_classes = [Krea2PrepareImageLatentsStep, Krea2PrepareMaskLatentsStep]
    block_names = ["add_noise", "prepare_mask"]

    @property
    def description(self) -> str:
        return "Add noise to Krea 2 source-image latents and prepare packed mask latents for inpainting."


# auto_docstring
class Krea2ReferenceInputStep(SequentialPipelineBlocks):
    """
    Expand Krea 2 text and reference-image conditioning to the effective batch.

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
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
          negative_prompt_embeds (`Tensor`):
              Negative text features, batch-expanded.
          negative_prompt_embeds_mask (`Tensor`):
              Negative text mask, batch-expanded.
          reference_image_latents (`list`):
              Packed reference-image latents expanded to the effective batch in conditioning order.
    """

    model_name = "krea2"
    block_classes = [Krea2TextInputsStep, Krea2ReferenceInputsStep]
    block_names = ["text_inputs", "reference_inputs"]

    @property
    def description(self) -> str:
        return "Expand Krea 2 text and reference-image conditioning to the effective batch."


# auto_docstring
class Krea2ReferenceCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Generate Krea 2 target latents from noise while attending to clean reference-image tokens.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
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
          num_inference_steps (`int`, *optional*, defaults to 28):
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
        Krea2ReferenceInputStep,
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2PrepareReferencePositionIdsStep,
        Krea2ReferenceDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "prepare_position_ids", "denoise"]

    @property
    def description(self) -> str:
        return "Generate Krea 2 target latents from noise while attending to clean reference-image tokens."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2Img2ImgCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Krea 2 image-to-image workflow with strength-adjusted flow-matching denoising.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
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
          num_inference_steps (`int`, *optional*, defaults to 28):
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
        Krea2Img2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2PrepareImageLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2DenoiseStep,
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
        return "Core Krea 2 image-to-image workflow with strength-adjusted flow-matching denoising."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2InpaintCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core Krea 2 inpainting workflow with masked latent blending after every denoising step.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
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
          num_inference_steps (`int`, *optional*, defaults to 28):
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
        Krea2Img2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2InpaintPrepareLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2InpaintDenoiseStep,
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
        return "Core Krea 2 inpainting workflow with masked latent blending after every denoising step."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Select the Krea 2 text-to-image, image-to-image, or inpainting denoising workflow.

      Components:
          transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) guider
          (`ClassifierFreeGuidance`)

      Inputs:
          num_images_per_prompt (`int`, *optional*, defaults to 1):
              The number of images to generate per prompt.
          prompt_embeds (`Tensor`):
              Per-prompt stacked text features (B, text_seq_len, num_text_layers, text_hidden_dim).
          prompt_embeds_mask (`Tensor`):
              Per-prompt boolean text mask (B, text_seq_len).
          negative_prompt_embeds (`Tensor`, *optional*):
              Per-prompt negative text features.
          negative_prompt_embeds_mask (`Tensor`, *optional*):
              Per-prompt negative text mask.
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
        Krea2CoreDenoiseStep,
        Krea2ReferenceCoreDenoiseStep,
        Krea2InpaintCoreDenoiseStep,
        Krea2Img2ImgCoreDenoiseStep,
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
        return "Select the Krea 2 text-to-image, image-to-image, or inpainting denoising workflow."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


# auto_docstring
class Krea2AutoDecodeStep(AutoPipelineBlocks):
    """
    Select the standard or inpainting-aware Krea 2 decoder.

      Components:
          vae (`AutoencoderKLQwenImage`) image_mask_processor (`InpaintProcessor`) image_processor
          (`VaeImageProcessor`)

      Inputs:
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          height (`int`, *optional*, defaults to 1024):
              The height in pixels of the generated image.
          width (`int`, *optional*, defaults to 1024):
              The width in pixels of the generated image.
          latents (`Tensor`):
              Pre-generated noisy latents for image generation.
          mask_overlay_kwargs (`dict`, *optional*):
              Arguments used to overlay a cropped inpainting result on the original image.

      Outputs:
          images (`list`):
              Generated images.
    """

    model_name = "krea2"
    block_classes = [Krea2InpaintDecodeStep, Krea2DecodeStep]
    block_names = ["inpaint", "default"]
    block_trigger_inputs = ["mask", None]

    @property
    def description(self) -> str:
        return "Select the standard or inpainting-aware Krea 2 decoder."


# auto_docstring
class Krea2AutoBlocks(SequentialPipelineBlocks):
    """
    Auto Modular pipeline for text-to-image, image-to-image, and inpainting using Krea 2.

      Supported workflows:
        - `text2image`: requires `prompt`
        - `image2image`: requires `prompt`, `image`
        - `inpainting`: requires `prompt`, `image`, `mask_image`
        - `reference`: requires `prompt`, `reference_image`

      Components:
          text_encoder (`Qwen3VLModel`): The Qwen3-VL text encoder. reference_image_processor
          (`Krea2ReferenceImageProcessor`): The Qwen3-VL processor used for image-grounded prompt encoding. tokenizer
          (`AutoTokenizer`): The tokenizer paired with the text encoder. guider (`ClassifierFreeGuidance`)
          image_processor (`VaeImageProcessor`) vae (`AutoencoderKLQwenImage`) image_mask_processor
          (`InpaintProcessor`) transformer (`Krea2Transformer2DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          prompt (`str`):
              The prompt or prompts to guide image generation.
          negative_prompt (`str`, *optional*):
              The negative prompt(s) for CFG.
          reference_image (`Image | list`, *optional*):
              A reference image or ordered list of reference images shared by all prompts in the batch.
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
        Krea2AutoTextEncoderStep,
        Krea2AutoVaeEncoderStep,
        Krea2AutoCoreDenoiseStep,
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
        return "Auto Modular pipeline for text-to-image, image-to-image, and inpainting using Krea 2."

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]
