# Copyright 2025 Salesforce.com, inc.
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
from typing import List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPTokenizer

from ...models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from ...schedulers import PNDMScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..blip_diffusion.blip_image_processing import BlipImageProcessor
from ..blip_diffusion.modeling_blip2 import Blip2QFormerModel
from ..blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel
from ..pipeline_utils import DeprecatedPipelineMixin, DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers.pipelines import BlipDiffusionControlNetPipeline
        >>> from diffusers.utils import load_image
        >>> from controlnet_aux import CannyDetector
        >>> import torch

        >>> blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
        ...     "Salesforce/blipdiffusion-controlnet", torch_dtype=torch.float16
        ... ).to("cuda")

        >>> style_subject = "flower"
        >>> tgt_subject = "teapot"
        >>> text_prompt = "on a marble table"

        >>> cldm_cond_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/kettle.jpg"
        ... ).resize((512, 512))
        >>> canny = CannyDetector()
        >>> cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
        >>> style_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/flower.jpg"
        ... )
        >>> guidance_scale = 7.5
        >>> num_inference_steps = 50
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


        >>> output = blip_diffusion_pipe(
        ...     text_prompt,
        ...     style_image,
        ...     cldm_cond_image,
        ...     style_subject,
        ...     tgt_subject,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=num_inference_steps,
        ...     neg_prompt=negative_prompt,
        ...     height=512,
        ...     width=512,
        ... ).images
        >>> output[0].save("image.png")
        ```
"""


class BlipDiffusionControlNetPipeline(DeprecatedPipelineMixin, DiffusionPipeline):
    """
    Pipeline for Canny Edge based Controlled subject-driven generation using Blip Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer for the text encoder
        text_encoder ([`ContextCLIPTextModel`]):
            Text encoder to encode the text prompt
        vae ([`AutoencoderKL`]):
            VAE model to map the latents to the image
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        scheduler ([`PNDMScheduler`]):
             A scheduler to be used in combination with `unet` to generate image latents.
        qformer ([`Blip2QFormerModel`]):
            QFormer model to get multi-modal embeddings from the text and image.
        controlnet ([`ControlNetModel`]):
            ControlNet model to get the conditioning image embedding.
        image_processor ([`BlipImageProcessor`]):
            Image Processor to preprocess and postprocess the image.
        ctx_begin_pos (int, `optional`, defaults to 2):
            Position of the context token in the text encoder.
    """

    _last_supported_version = "0.33.1"
    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: ContextCLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        qformer: Blip2QFormerModel,
        controlnet: ControlNetModel,
        image_processor: BlipImageProcessor,
        ctx_begin_pos: int = 2,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            qformer=qformer,
            controlnet=controlnet,
            image_processor=image_processor,
        )
        self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)

    def get_query_embeddings(self, input_image, src_subject):
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)

    # from the original Blip Diffusion code, specifies the target subject and augments the prompt by repeating it
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    # Copied from diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_prompt(self, query_embeds, prompt, device=None):
        device = device or self._execution_device

        # embeddings for prompt, with query_embeds as context
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        max_len -= self.qformer.config.num_query_tokens

        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        batch_size = query_embeds.shape[0]
        ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=ctx_begin_pos,
        )[0]

        return text_embeddings

    # Adapted from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        image = self.image_processor.preprocess(
            image,
            size={"width": width, "height": height},
            do_rescale=True,
            do_center_crop=False,
            do_normalize=False,
            return_tensors="pt",
        )["pixel_values"].to(device)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        condtioning_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            reference_image (`PIL.Image.Image`):
                The reference image to condition the generation on.
            condtioning_image (`PIL.Image.Image`):
                The conditioning canny edge image to condition the generation on.
            source_subject_category (`List[str]`):
                The source subject category.
            target_subject_category (`List[str]`):
                The target subject category.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by random sampling.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to 512):
                The height of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width of the generated image.
            seed (`int`, *optional*, defaults to 42):
                The seed to use for random generation.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            neg_prompt (`str`, *optional*, defaults to ""):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_strength (`float`, *optional*, defaults to 1.0):
                The strength of the prompt. Specifies the number of times the prompt is repeated along with prompt_reps
                to amplify the prompt.
            prompt_reps (`int`, *optional*, defaults to 20):
                The number of times the prompt is repeated along with prompt_strength to amplify the prompt.
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """
        device = self._execution_device

        reference_image = self.image_processor.preprocess(
            reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        )["pixel_values"]
        reference_image = reference_image.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]

        batch_size = len(prompt)

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt, device)
        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=height // scale_down_factor,
            width=width // scale_down_factor,
            generator=generator,
            latents=latents,
            dtype=self.unet.dtype,
            device=device,
        )
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        cond_image = self.prepare_control_image(
            image=condtioning_image,
            width=width,
            height=height,
            batch_size=batch_size,
            num_images_per_prompt=1,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_image,
                return_dict=False,
            )

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]

            if XLA_AVAILABLE:
                xm.mark_step()

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
