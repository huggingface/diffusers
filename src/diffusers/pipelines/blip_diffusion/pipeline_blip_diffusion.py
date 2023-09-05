# Copyright 2023 Salesforce.com, inc.
# Copyright 2023 The HuggingFace Team. All rights reserved.#
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
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from ...models import AutoencoderKL, UNet2DConditionModel
from .modeling_ctx_clip import ContextCLIPTextModel
from transformers import CLIPTokenizer
from ...pipelines import DiffusionPipeline
import torch
from ...schedulers import PNDMScheduler
from ...utils import (
    BaseOutput,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ...utils.pil_utils import PIL_INTERPOLATION
from torch import nn
from transformers.activations import QuickGELUActivation as QuickGELU
from .modeling_blip2 import Blip2QFormerModel
import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .blip_image_processing import BlipImageProcessor
from PIL import Image
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import re

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import BlipDiffusionPipeline
        >>> from PIL import Image
        >>> from diffusers.utils import load_image

        >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained('ayushtues/blipdiffusion')
        >>> blip_diffusion_pipe.to('cuda')

        >>> cond_subject = ["dog"]
        >>> tgt_subject = ["dog"]
        >>> text_prompt_input = ["swimming underwater"]


        >>> cond_image = load_image(
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
        ... )
        >>> iter_seed = 88888
        >>> guidance_scale = 7.5
        >>> num_inference_steps = 50
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

        
        >>> output = blip_diffusion_pipe(
        ...     text_prompt_input,
        ...     cond_image,
        ...     cond_subject,
        ...     tgt_subject,
        ...     seed=iter_seed + i,
        ...     guidance_scale=guidance_scale,
        ...     num_inference_steps=num_inference_steps,
        ...     neg_prompt=negative_prompt,
        ...     height=512,
        ...     width=512,
        ...     )
        >>> output[0].save("dog.png")
        ```
"""

class BlipDiffusionPipeline(DiffusionPipeline):
    """
    Pipeline for Zero-Shot Subject Driven Generation using Blip Diffusion.

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
        image_processor ([`BlipImageProcessor`]):
            Image Processor to preprocess and postprocess the image.
        ctx_begin_pos (int, `optional`, defaults to 2):
            Position of the context token in the text encoder.
    """
    
    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: ContextCLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: PNDMScheduler, qformer: Blip2QFormerModel,  image_processor: BlipImageProcessor, ctx_begin_pos: int = 2, mean : List[float] = None, std : List[float] = None):
        super().__init__()

        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,  vae=vae, unet=unet, scheduler=scheduler, qformer=qformer, image_processor=image_processor)
        self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)
    
    #TODO Complete this function
    def check_inputs(self, prompt, reference_image, source_subject_category, target_subject_category):
        pass

    def get_query_embeddings(self, input_image, src_subject):
        return self.forward_ctx_embeddings(input_image, src_subject)

    # from the original Blip Diffusion code, speciefies the target subject and augments the prompt by repeating it
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

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

        return latents

    def encode_prompt(self, query_embeds, prompt):
        #embeddings for prompt, with query_embeds as context
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        max_len -= self.qformer.config.num_query_tokens

        tokenized_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self.config.ctx_begin_pos],
        )[0]

        return text_embeddings


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt : Union[str, List[str]],
        reference_image : Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
        source_subject_category : Union[str, List[str]],
        target_subject_category : Union[str, List[str]],
        latents : Optional[torch.FloatTensor] = None,
        guidance_scale: float =7.5,
        height: int=512,
        width: int=512,
        seed: int=42,
        num_inference_steps: int=50,
        neg_prompt: Optional[str] = "",
        prompt_strength: float =1.0,
        prompt_reps: int=20,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt :
                The prompt or prompts to guide the image generation.
            reference_image :
                The reference image to condition the generation on.
            source_subject_category :
                The source subject category.
            target_subject_category :
                The target subject category.
            latents :
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by random sampling.
            guidance_scale :
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            height :
                The height of the generated image.
            width :
                The width of the generated image.
            seed :
                The seed to use for random generation.
            num_inference_steps :
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            neg_prompt :
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_strength :
                The strength of the prompt. Specifies the number of times the prompt is repeated along with prompt_reps
                to amplify the prompt.
            prompt_reps :
                The number of times the prompt is repeated along with prompt_strength to amplify the prompt.
        Examples:

        Returns:
            `List[PIL.Image.Image]` : The generated images.
        """

        reference_image = self.image_processor.preprocess(reference_image,  image_mean=self.config.mean, image_std=self.config.std, return_tensors='pt')['pixel_values']
        reference_image = reference_image.to(self.device)

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(
            query_embeds, prompt
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        #TODO - Handle batch size > 1
        latents = self.prepare_latents(batch_size=1, num_channels=self.unet.in_channels, height=height//8, width=width//8, generator=generator, latents=latents, dtype=self.unet.dtype, device=self.device)
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(self.scheduler.timesteps)

        for i, t in enumerate(iterator):

            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # TODO - Handle ddim as well
            # # compute the previous noisy sample x_t -> x_t-1
            # scheduler = self.ddim_scheduler if use_inversion else self.pndm_scheduler

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")

        return image

    def forward_ctx_embeddings(self, input_image, text_input, ratio=None):
        def compute_ctx_embeddings(input_image, text_input):
            ctx_embeddings = self.qformer(image_input=input_image, text_input=text_input, return_dict=False)
            return ctx_embeddings

        if isinstance(text_input, str):
            text_input = [text_input]


        if isinstance(text_input[0], str):
            text_input, input_image = [text_input], [input_image]

        all_ctx_embeddings = []

        for inp_image, inp_text in zip(input_image, text_input):
            ctx_embeddings = compute_ctx_embeddings(inp_image, inp_text)
            all_ctx_embeddings.append(ctx_embeddings)

        if ratio is not None:
            assert len(ratio) == len(all_ctx_embeddings)
            assert sum(ratio) == 1
        else:
            ratio = [1 / len(all_ctx_embeddings)] * len(all_ctx_embeddings)

        ctx_embeddings = torch.zeros_like(all_ctx_embeddings[0])

        for ratio, ctx_embeddings_ in zip(ratio, all_ctx_embeddings):
            ctx_embeddings += ratio * ctx_embeddings_

        return ctx_embeddings




