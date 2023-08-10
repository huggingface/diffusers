from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from ...models import AutoencoderKL, UNet2DConditionModel
from .modeling_ctx_clip import CtxCLIPTextModel
from transformers import CLIPTokenizer
from ...pipelines import DiffusionPipeline
import torch
from ...schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from ...utils import (
    BaseOutput,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from torch import nn
from transformers.activations import QuickGELUActivation as QuickGELU
from .modeling_blip2 import Blip2QFormerModel
import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import re

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images



# Create a class for the Blip Diffusion pipeline
class BlipDiffusionPipeline(DiffusionPipeline):
    
    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CtxCLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: PNDMScheduler, qformer: Blip2QFormerModel):
        super().__init__()

        self._CTX_BEGIN_POS = 2

        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,  vae=vae, unet=unet, scheduler=scheduler, qformer=qformer)

    def prepare_latents():
        pass

    def encode_prompt():
        pass
    
    def enable_sequential_cpu_offload():
        pass

    def enable_model_cpu_offload(self, gpu_id=0):
        pass

    def __call__(self):
        pass

    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    def _predict_noise(
        self,
        t,
        latent_model_input,
        text_embeddings,
        width=512,
        height=512,
        cond_image=None,
    ):
 
        down_block_res_samples, mid_block_res_sample = None, None

        noise_pred = self.unet(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )["sample"]

        return noise_pred

    def _init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=generator.device,
            )
        latent = latent.expand(
            batch_size,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        return latent.to(self.device)

    def _forward_prompt_embeddings(self, input_image, src_subject, prompt):
        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.forward_ctx_embeddings(input_image, src_subject)

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self._tokenize_text(prompt).to(self.device)
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self._CTX_BEGIN_POS],
        )[0]

        return text_embeddings

    @torch.no_grad()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        assert isinstance(image, torch.Tensor)

        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    def preprocess_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > 50:
            caption = " ".join(caption_words[:50])

        return caption
    
    def preprocess_image(self, image):
                
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

        return transform(image)


    @torch.no_grad()
    def generate(
        self,
        samples,
        latents=None,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
        neg_prompt="",
        controller=None,
        prompt_strength=1.0,
        prompt_reps=20,
        use_ddim=False,
    ):
        if controller is not None:
            self._register_attention_refine(controller)

        cond_image = samples["cond_images"]  # reference image
        cond_subject = samples["cond_subject"]  # source subject category
        tgt_subject = samples["tgt_subject"]  # target subject category
        prompt = samples["prompt"]
        cldm_cond_image = samples.get("cldm_cond_image", None)  # conditional image

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=tgt_subject,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )

        text_embeddings = self._forward_prompt_embeddings(
            cond_image, cond_subject, prompt
        )
        # 3. unconditional embedding
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

        latents = self._init_latent(latents, height, width, generator, batch_size=1)


        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(self.scheduler.timesteps)

        for i, t in enumerate(iterator):
            latents = self._denoise_latent_step(
                latents=latents,
                t=t,
                text_embeddings=text_embeddings,
                cond_image=cldm_cond_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                use_inversion=use_ddim,
            )

        image = self._latent_to_image(latents)
        


        return image

    def _latent_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        print(torch.mean(image))
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image

    def _denoise_latent_step(
        self,
        latents,
        t,
        text_embeddings,
        guidance_scale,
        height,
        width,
        cond_image=None,
        use_inversion=False,
    ):
        if use_inversion:
            noise_placeholder = []

        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        noise_pred = self._predict_noise(
            t=t,
            latent_model_input=latent_model_input,
            text_embeddings=text_embeddings,
            width=width,
            height=height,
            cond_image=cond_image,
        )

        if use_inversion:
            noise_placeholder.append(noise_pred[2].unsqueeze(0))

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if use_inversion:
            noise_placeholder.append(noise_pred[-1].unsqueeze(0))
            noise_pred = torch.cat(noise_placeholder)

        # TODO - Handle pndm_scheduler as well
        # # compute the previous noisy sample x_t -> x_t-1
        # scheduler = self.ddim_scheduler if use_inversion else self.pndm_scheduler

        latents = self.scheduler.step(
            noise_pred,
            t,
            latents,
        )["prev_sample"]


        return latents

    def _tokenize_text(self, text_input, with_query=True):
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        if with_query:
            max_len -= self.qformer.config.num_query_tokens

        tokenized_text = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return tokenized_text

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




