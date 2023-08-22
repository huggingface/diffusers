from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL
from ...models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from .modeling_ctx_clip import CtxCLIPTextModel
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
    numpy_to_pil
)
from ...utils.pil_utils import PIL_INTERPOLATION
from torch import nn
from transformers.activations import QuickGELUActivation as QuickGELU
from .modeling_blip2 import Blip2QFormerModel
import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import re

def prepare_cond_image(
        image, width, height, batch_size, device, do_classifier_free_guidance=True
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            # repeat_by = num_images_per_prompt
            raise NotImplementedError

        image = image.repeat_interleave(repeat_by, dim=0)

        # image = image.to(device=self.device, dtype=dtype)
        image = image.to(device=device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image



# Create a class for the Blip Diffusion pipeline
class BlipDiffusionPipeline(DiffusionPipeline):
    
    def __init__(self, tokenizer: CLIPTokenizer, text_encoder: CtxCLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: PNDMScheduler, qformer: Blip2QFormerModel, controlnet: ControlNetModel=None, ctx_begin_pos: int = 2):
        super().__init__()


        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,  vae=vae, unet=unet, scheduler=scheduler, qformer=qformer, controlnet=controlnet)
        self.register_to_config(ctx_begin_pos=ctx_begin_pos)
    
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
        tokenized_prompt = self._tokenize_text(prompt).to(self.device)
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self.config.ctx_begin_pos],
        )[0]

        return text_embeddings

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
    def __call__(
        self,
        prompt,
        reference_image,
        source_subject_category,
        target_subject_category,
        condtioning_image,
        latents=None,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
        neg_prompt="",
        prompt_strength=1.0,
        prompt_reps=20,
        use_ddim=False,
    ):

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(
            reference_image, source_subject_category, prompt
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

        #TODO - Handle batch size > 1
        latents = self.prepare_latents(batch_size=1, num_channels=self.unet.in_channels, height=height//8, width=width//8, generator=generator, latents=latents, dtype=self.unet.dtype, device=self.device)
        # set timesteps
        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(self.scheduler.timesteps)

        for i, t in enumerate(iterator):
            if use_ddim:
                noise_placeholder = []

            # expand the latents if we are doing classifier free guidance
            do_classifier_free_guidance = guidance_scale > 1.0

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            if self.controlnet is not None:
                cond_image = prepare_cond_image(
                    condtioning_image, width, height, batch_size=1, device=self.device
                )
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=cond_image,
                    # conditioning_scale=controlnet_condition_scale,
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None

            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )["sample"]

            if use_ddim:
                noise_placeholder.append(noise_pred[2].unsqueeze(0))

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if use_ddim:
                noise_placeholder.append(noise_pred[-1].unsqueeze(0))
                noise_pred = torch.cat(noise_placeholder)

            # TODO - Handle ddim as well
            # # compute the previous noisy sample x_t -> x_t-1
            # scheduler = self.ddim_scheduler if use_inversion else self.pndm_scheduler

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.postprocess_image(image, output_type="pil")

        return image

    # Follows diffusers.VaeImageProcessor.postprocess
    def postprocess_image(self, sample: torch.FloatTensor, output_type: str = "pil"):
        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(
                f"output_type={output_type} is not supported. Make sure to choose one of ['pt', 'np', or 'pil']"
            )

        # Equivalent to diffusers.VaeImageProcessor.denormalize
        sample = (sample / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return sample

        # Equivalent to diffusers.VaeImageProcessor.pt_to_numpy
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "np":
            return sample

        # Output_type must be 'pil'
        sample = numpy_to_pil(sample)
        return sample

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




