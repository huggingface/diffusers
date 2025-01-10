# Copyright 2024 The GLIGEN Authors and HuggingFace Team. All rights reserved.
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

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...image_processor import VaeImageProcessor
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention import GatedSelfAttentionDense
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from ..stable_diffusion import StableDiffusionPipelineOutput
from ..stable_diffusion.clip_image_project_model import CLIPImageProjection
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionGLIGENTextImagePipeline
        >>> from diffusers.utils import load_image

        >>> # Insert objects described by image at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        ...     "anhnct/Gligen_Inpainting_Text_Image", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> input_image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"
        ... )
        >>> prompt = "a backpack"
        >>> boxes = [[0.2676, 0.4088, 0.4773, 0.7183]]
        >>> phrases = None
        >>> gligen_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/backpack.jpeg"
        ... )

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=phrases,
        ...     gligen_inpaint_image=input_image,
        ...     gligen_boxes=boxes,
        ...     gligen_images=[gligen_image],
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... ).images

        >>> images[0].save("./gligen-inpainting-text-image-box.jpg")

        >>> # Generate an image described by the prompt and
        >>> # insert objects described by text and image at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        ...     "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a flower sitting on the beach"
        >>> boxes = [[0.0, 0.09, 0.53, 0.76]]
        >>> phrases = ["flower"]
        >>> gligen_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/pexels-pixabay-60597.jpg"
        ... )

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=phrases,
        ...     gligen_images=[gligen_image],
        ...     gligen_boxes=boxes,
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... ).images

        >>> images[0].save("./gligen-generation-text-image-box.jpg")

        >>> # Generate an image described by the prompt and
        >>> # transfer style described by image at the region defined by bounding boxes
        >>> pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        ...     "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a dragon flying on the sky"
        >>> boxes = [[0.4, 0.2, 1.0, 0.8], [0.0, 1.0, 0.0, 1.0]]  # Set `[0.0, 1.0, 0.0, 1.0]` for the style

        >>> gligen_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
        ... )

        >>> gligen_placeholder = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
        ... )

        >>> images = pipe(
        ...     prompt=prompt,
        ...     gligen_phrases=[
        ...         "dragon",
        ...         "placeholder",
        ...     ],  # Can use any text instead of `placeholder` token, because we will use mask here
        ...     gligen_images=[
        ...         gligen_placeholder,
        ...         gligen_image,
        ...     ],  # Can use any image in gligen_placeholder, because we will use mask here
        ...     input_phrases_mask=[1, 0],  # Set 0 for the placeholder token
        ...     input_images_mask=[0, 1],  # Set 0 for the placeholder image
        ...     gligen_boxes=boxes,
        ...     gligen_scheduled_sampling_beta=1,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ... ).images

        >>> images[0].save("./gligen-generation-text-image-box-style-transfer.jpg")
        ```
"""


class StableDiffusionGLIGENTextImagePipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with Grounded-Language-to-Image Generation (GLIGEN).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        processor ([`~transformers.CLIPProcessor`]):
            A `CLIPProcessor` to procces reference image.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        image_project ([`CLIPImageProjection`]):
            A `CLIPImageProjection` to project image embedding into phrases embedding space.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for
            more details about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        processor: CLIPProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        image_project: CLIPImageProjection,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            processor=processor,
            image_project=image_project,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        gligen_images,
        gligen_phrases,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if gligen_images is not None and gligen_phrases is not None:
            if len(gligen_images) != len(gligen_phrases):
                raise ValueError(
                    "`gligen_images` and `gligen_phrases` must have the same length when both are provided, but"
                    f" got: `gligen_images` with length {len(gligen_images)} != `gligen_phrases` with length {len(gligen_phrases)}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def enable_fuser(self, enabled=True):
        for module in self.unet.modules():
            if type(module) is GatedSelfAttentionDense:
                module.enabled = enabled

    def draw_inpaint_mask_from_boxes(self, boxes, size):
        """
        Create an inpainting mask based on given boxes. This function generates an inpainting mask using the provided
        boxes to mark regions that need to be inpainted.
        """
        inpaint_mask = torch.ones(size[0], size[1])
        for box in boxes:
            x0, x1 = box[0] * size[0], box[2] * size[0]
            y0, y1 = box[1] * size[1], box[3] * size[1]
            inpaint_mask[int(y0) : int(y1), int(x0) : int(x1)] = 0
        return inpaint_mask

    def crop(self, im, new_width, new_height):
        """
        Crop the input image to the specified dimensions.
        """
        width, height = im.size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return im.crop((left, top, right, bottom))

    def target_size_center_crop(self, im, new_hw):
        """
        Crop and resize the image to the target size while keeping the center.
        """
        width, height = im.size
        if width != height:
            im = self.crop(im, min(height, width), min(height, width))
        return im.resize((new_hw, new_hw), PIL.Image.LANCZOS)

    def complete_mask(self, has_mask, max_objs, device):
        """
        Based on the input mask corresponding value `0 or 1` for each phrases and image, mask the features
        corresponding to phrases and images.
        """
        mask = torch.ones(1, max_objs).type(self.text_encoder.dtype).to(device)
        if has_mask is None:
            return mask

        if isinstance(has_mask, int):
            return mask * has_mask
        else:
            for idx, value in enumerate(has_mask):
                mask[0, idx] = value
            return mask

    def get_clip_feature(self, input, normalize_constant, device, is_image=False):
        """
        Get image and phrases embedding by using CLIP pretrain model. The image embedding is transformed into the
        phrases embedding space through a projection.
        """
        if is_image:
            if input is None:
                return None
            inputs = self.processor(images=[input], return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.image_encoder.dtype)

            outputs = self.image_encoder(**inputs)
            feature = outputs.image_embeds
            feature = self.image_project(feature).squeeze(0)
            feature = (feature / feature.norm()) * normalize_constant
            feature = feature.unsqueeze(0)
        else:
            if input is None:
                return None
            inputs = self.tokenizer(input, return_tensors="pt", padding=True).to(device)
            outputs = self.text_encoder(**inputs)
            feature = outputs.pooler_output
        return feature

    def get_cross_attention_kwargs_with_grounded(
        self,
        hidden_size,
        gligen_phrases,
        gligen_images,
        gligen_boxes,
        input_phrases_mask,
        input_images_mask,
        repeat_batch,
        normalize_constant,
        max_objs,
        device,
    ):
        """
        Prepare the cross-attention kwargs containing information about the grounded input (boxes, mask, image
        embedding, phrases embedding).
        """
        phrases, images = gligen_phrases, gligen_images
        images = [None] * len(phrases) if images is None else images
        phrases = [None] * len(images) if phrases is None else phrases

        boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
        masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        phrases_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        image_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        phrases_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)
        image_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)

        text_features = []
        image_features = []
        for phrase, image in zip(phrases, images):
            text_features.append(self.get_clip_feature(phrase, normalize_constant, device, is_image=False))
            image_features.append(self.get_clip_feature(image, normalize_constant, device, is_image=True))

        for idx, (box, text_feature, image_feature) in enumerate(zip(gligen_boxes, text_features, image_features)):
            boxes[idx] = torch.tensor(box)
            masks[idx] = 1
            if text_feature is not None:
                phrases_embeddings[idx] = text_feature
                phrases_masks[idx] = 1
            if image_feature is not None:
                image_embeddings[idx] = image_feature
                image_masks[idx] = 1

        input_phrases_mask = self.complete_mask(input_phrases_mask, max_objs, device)
        phrases_masks = phrases_masks.unsqueeze(0).repeat(repeat_batch, 1) * input_phrases_mask
        input_images_mask = self.complete_mask(input_images_mask, max_objs, device)
        image_masks = image_masks.unsqueeze(0).repeat(repeat_batch, 1) * input_images_mask
        boxes = boxes.unsqueeze(0).repeat(repeat_batch, 1, 1)
        masks = masks.unsqueeze(0).repeat(repeat_batch, 1)
        phrases_embeddings = phrases_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1)
        image_embeddings = image_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1)

        out = {
            "boxes": boxes,
            "masks": masks,
            "phrases_masks": phrases_masks,
            "image_masks": image_masks,
            "phrases_embeddings": phrases_embeddings,
            "image_embeddings": image_embeddings,
        }

        return out

    def get_cross_attention_kwargs_without_grounded(self, hidden_size, repeat_batch, max_objs, device):
        """
        Prepare the cross-attention kwargs without information about the grounded input (boxes, mask, image embedding,
        phrases embedding) (All are zero tensor).
        """
        boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
        masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        phrases_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        image_masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        phrases_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)
        image_embeddings = torch.zeros(max_objs, hidden_size, device=device, dtype=self.text_encoder.dtype)

        out = {
            "boxes": boxes.unsqueeze(0).repeat(repeat_batch, 1, 1),
            "masks": masks.unsqueeze(0).repeat(repeat_batch, 1),
            "phrases_masks": phrases_masks.unsqueeze(0).repeat(repeat_batch, 1),
            "image_masks": image_masks.unsqueeze(0).repeat(repeat_batch, 1),
            "phrases_embeddings": phrases_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1),
            "image_embeddings": image_embeddings.unsqueeze(0).repeat(repeat_batch, 1, 1),
        }

        return out

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        gligen_scheduled_sampling_beta: float = 0.3,
        gligen_phrases: List[str] = None,
        gligen_images: List[PIL.Image.Image] = None,
        input_phrases_mask: Union[int, List[int]] = None,
        input_images_mask: Union[int, List[int]] = None,
        gligen_boxes: List[List[float]] = None,
        gligen_inpaint_image: Optional[PIL.Image.Image] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        gligen_normalize_constant: float = 28.7,
        clip_skip: int = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            gligen_phrases (`List[str]`):
                The phrases to guide what to include in each of the regions defined by the corresponding
                `gligen_boxes`. There should only be one phrase per bounding box.
            gligen_images (`List[PIL.Image.Image]`):
                The images to guide what to include in each of the regions defined by the corresponding `gligen_boxes`.
                There should only be one image per bounding box
            input_phrases_mask (`int` or `List[int]`):
                pre phrases mask input defined by the correspongding `input_phrases_mask`
            input_images_mask (`int` or `List[int]`):
                pre images mask input defined by the correspongding `input_images_mask`
            gligen_boxes (`List[List[float]]`):
                The bounding boxes that identify rectangular regions of the image that are going to be filled with the
                content described by the corresponding `gligen_phrases`. Each rectangular box is defined as a
                `List[float]` of 4 elements `[xmin, ymin, xmax, ymax]` where each value is between [0,1].
            gligen_inpaint_image (`PIL.Image.Image`, *optional*):
                The input image, if provided, is inpainted with objects described by the `gligen_boxes` and
                `gligen_phrases`. Otherwise, it is treated as a generation task on a blank input image.
            gligen_scheduled_sampling_beta (`float`, defaults to 0.3):
                Scheduled Sampling factor from [GLIGEN: Open-Set Grounded Text-to-Image
                Generation](https://arxiv.org/pdf/2301.07093.pdf). Scheduled Sampling factor is only varied for
                scheduled sampling during inference for improved quality and controllability.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            gligen_normalize_constant (`float`, *optional*, defaults to 28.7):
                The normalize value of the image embedding.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            gligen_images,
            gligen_phrases,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5.1 Prepare GLIGEN variables
        max_objs = 30
        if len(gligen_boxes) > max_objs:
            warnings.warn(
                f"More that {max_objs} objects found. Only first {max_objs} objects will be processed.",
                FutureWarning,
            )
            gligen_phrases = gligen_phrases[:max_objs]
            gligen_boxes = gligen_boxes[:max_objs]
            gligen_images = gligen_images[:max_objs]

        repeat_batch = batch_size * num_images_per_prompt

        if do_classifier_free_guidance:
            repeat_batch = repeat_batch * 2

        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        hidden_size = prompt_embeds.shape[2]

        cross_attention_kwargs["gligen"] = self.get_cross_attention_kwargs_with_grounded(
            hidden_size=hidden_size,
            gligen_phrases=gligen_phrases,
            gligen_images=gligen_images,
            gligen_boxes=gligen_boxes,
            input_phrases_mask=input_phrases_mask,
            input_images_mask=input_images_mask,
            repeat_batch=repeat_batch,
            normalize_constant=gligen_normalize_constant,
            max_objs=max_objs,
            device=device,
        )

        cross_attention_kwargs_without_grounded = {}
        cross_attention_kwargs_without_grounded["gligen"] = self.get_cross_attention_kwargs_without_grounded(
            hidden_size=hidden_size, repeat_batch=repeat_batch, max_objs=max_objs, device=device
        )

        # Prepare latent variables for GLIGEN inpainting
        if gligen_inpaint_image is not None:
            # if the given input image is not of the same size as expected by VAE
            # center crop and resize the input image to expected shape
            if gligen_inpaint_image.size != (self.vae.sample_size, self.vae.sample_size):
                gligen_inpaint_image = self.target_size_center_crop(gligen_inpaint_image, self.vae.sample_size)
            # Convert a single image into a batch of images with a batch size of 1
            # The resulting shape becomes (1, C, H, W), where C is the number of channels,
            # and H and W are the height and width of the image.
            # scales the pixel values to a range [-1, 1]
            gligen_inpaint_image = self.image_processor.preprocess(gligen_inpaint_image)
            gligen_inpaint_image = gligen_inpaint_image.to(dtype=self.vae.dtype, device=self.vae.device)
            # Run AutoEncoder to get corresponding latents
            gligen_inpaint_latent = self.vae.encode(gligen_inpaint_image).latent_dist.sample()
            gligen_inpaint_latent = self.vae.config.scaling_factor * gligen_inpaint_latent
            # Generate an inpainting mask
            # pixel value = 0, where the object is present (defined by bounding boxes above)
            #               1, everywhere else
            gligen_inpaint_mask = self.draw_inpaint_mask_from_boxes(gligen_boxes, gligen_inpaint_latent.shape[2:])
            gligen_inpaint_mask = gligen_inpaint_mask.to(
                dtype=gligen_inpaint_latent.dtype, device=gligen_inpaint_latent.device
            )
            gligen_inpaint_mask = gligen_inpaint_mask[None, None]
            gligen_inpaint_mask_addition = torch.cat(
                (gligen_inpaint_latent * gligen_inpaint_mask, gligen_inpaint_mask), dim=1
            )
            # Convert a single mask into a batch of masks with a batch size of 1
            gligen_inpaint_mask_addition = gligen_inpaint_mask_addition.expand(repeat_batch, -1, -1, -1).clone()

        int(gligen_scheduled_sampling_beta * len(timesteps))
        self.enable_fuser(True)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if latents.shape[1] != 4:
                    latents = torch.randn_like(latents[:, :4])

                if gligen_inpaint_image is not None:
                    gligen_inpaint_latent_with_noise = (
                        self.scheduler.add_noise(
                            gligen_inpaint_latent, torch.randn_like(gligen_inpaint_latent), torch.tensor([t])
                        )
                        .expand(latents.shape[0], -1, -1, -1)
                        .clone()
                    )
                    latents = gligen_inpaint_latent_with_noise * gligen_inpaint_mask + latents * (
                        1 - gligen_inpaint_mask
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if gligen_inpaint_image is not None:
                    latent_model_input = torch.cat((latent_model_input, gligen_inpaint_mask_addition), dim=1)

                # predict the noise residual with grounded information
                noise_pred_with_grounding = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # predict the noise residual without grounded information
                noise_pred_without_grounding = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs_without_grounded,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    # Using noise_pred_text from noise residual with grounded information and noise_pred_uncond from noise residual without grounded information
                    _, noise_pred_text = noise_pred_with_grounding.chunk(2)
                    noise_pred_uncond, _ = noise_pred_without_grounding.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_with_grounding

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
