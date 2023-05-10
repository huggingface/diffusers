# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer, XLMRobertaTokenizerFast

from ...models import PriorTransformer, UNet2DConditionModel
from ...pipelines import DiffusionPipeline
from ...schedulers import UnCLIPScheduler

from .text_encoder import MultilingualCLIP
from .text_proj import KandinskyTextProjModel

from ...utils import (
    logging, 
    randn_tensor,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_new_h_w(h, w):
    new_h = h // 64
    if h % 64 != 0:
        new_h += 1
    new_w = w // 64
    if w % 64 != 0:
        new_w += 1
    return new_h * 8, new_w * 8

class KandinskyPriorPipeline(DiffusionPipeline):
    """
    Pipeline for generate image prior for Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        multiclip ([`MultilingualCLIP`]):
            A multilingual text encoder.
        multiclip_tokenizer ([`XLMRobertaTokenizerFast`]):
            Tokenizer for multiclip
    """

    def __init__(
        self,
        prior: PriorTransformer,
        prior_image_encoder: CLIPVisionModelWithProjection,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior_tokenizer: CLIPTokenizer,
        prior_scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_encoder=prior_image_encoder,
            #multiclip=multiclip,
            #multiclip_tokenizer=multiclip_tokenizer,
        )

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def create_zero_img_emb(self, batch_size, device):
        zero_img = torch.zeros(1, 3, 224, 224).to(device=device)
        zero_image_emb = self.prior_image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size,1)
        return zero_image_emb

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        
        batch_size = len(prompt) if isinstance(prompt, list) else 1
            # get prompt text embeddings
        text_inputs = self.prior_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.prior_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.prior_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.prior_tokenizer.batch_decode(
                untruncated_ids[:, self.prior_tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.prior_tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.prior_tokenizer.model_max_length]

        text_encoder_output = self.prior_text_encoder(text_input_ids.to(device))

        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
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

            uncond_input = self.prior_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.prior_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_text_encoder_output = self.prior_text_encoder(uncond_input.input_ids.to(device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask
    
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        # TO_DO
        return self.device

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        prior_num_inference_steps: int =5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prior_latents: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prior_guidance_scale: float = 4.0, 
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        if prompt == '' or prompt[0] == '':

            image_embeddings = self.create_zero_img_emb(batch_size=batch_size, device=device)
        
        else:

            do_classifier_free_guidance = prior_guidance_scale > 1.0
            prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # prior
            self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
            prior_timesteps_tensor = self.prior_scheduler.timesteps

            embedding_dim = self.prior.config.embedding_dim

            prior_latents = self.prepare_latents(
                (batch_size, embedding_dim),
                prompt_embeds.dtype,
                device,
                generator,
                prior_latents,
                self.prior_scheduler,
            )

            for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents

                predicted_image_embedding = self.prior(
                    latent_model_input,
                    timestep=t,
                    proj_embedding=prompt_embeds,
                    encoder_hidden_states=text_encoder_hidden_states,
                    attention_mask=text_mask,
                ).predicted_image_embedding

                if do_classifier_free_guidance:
                    predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                    predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                        predicted_image_embedding_text - predicted_image_embedding_uncond
                    )

                if i + 1 == prior_timesteps_tensor.shape[0]:
                    prev_timestep = None
                else:
                    prev_timestep = prior_timesteps_tensor[i + 1]

                prior_latents = self.prior_scheduler.step(
                    predicted_image_embedding,
                    timestep=t,
                    sample=prior_latents,
                    generator=generator,
                    prev_timestep=prev_timestep,
                ).prev_sample

            prior_latents = self.prior.post_process_latents(prior_latents)

            image_embeddings = prior_latents

        return image_embeddings


class KandinskyPipeline(DiffusionPipeline):
    """
    Pipeline for image based on text prompt and image prior for Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder: 
            to-add
        tokenizer: 
            to-add
        scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        text_proj ([`KandinskyTextProjModel`]):
            Utility class to prepare and combine the embeddings before they are passed to the decoder.
    """
    
    def __init__(
        self,
        text_encoder: MultilingualCLIP,
        tokenizer: XLMRobertaTokenizerFast,
        text_proj: KandinskyTextProjModel,
        unet: UNet2DConditionModel,
        scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
            unet=unet,
            scheduler=scheduler,
        )

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        text_mask = text_inputs.attention_mask.to(device)

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
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        prompt_embeds, text_encoder_hidden_states = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
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

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            uncond_text_input_ids = uncond_input.input_ids.to(device)
            uncond_text_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds, uncond_text_encoder_hidden_states = self.text_encoder(input_ids=uncond_text_input_ids, attention_mask=uncond_text_mask)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask
    

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        # TO_DO
        return self.device

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        #prompt_embeds: Optional[torch.FloatTensor] = None,
        #text_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict:  bool = True,
    ):

        if prompt is not None:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        else:
            batch_size = prompt_embeds.shape[0] //2
        
        device = self._execution_device

        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, text_encoder_hidden_states, _ = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        # TO_DO [2] add a step to create negative_image_embeds https://github.com/ai-forever/Kandinsky-2/blob/main/kandinsky2/kandinsky2_1_model.py#L322
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(device)
        
        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeds,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_encoder_hidden_states,
            )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_tensor = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels

        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size
        height, width = get_new_h_w(height, width)
        
        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_encoder_hidden_states.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        
        # expand the latents if we are doing classifier free guidance
        latents = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            noise_pred = self.unet(
                sample=latents, #[2, 4, 96, 96]
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
            ).sample

            # YiYi Notes: CFG is currently implemented exactly as original repo as a baseline, 
              # i.e. we apply cfg to predicted noise, and take predicted variance as it is (uncond + cond) 
              # this means the our latent shape is batch_size *2 instad batch_size

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                variance_pred_uncond, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred] * 2) 
                variance_pred = torch.cat([variance_pred_uncond, variance_pred_text])
                noise_pred = torch.cat([noise_pred, variance_pred], dim=1)

            if i + 1 == timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, prev_timestep=prev_timestep, generator=generator, batch_size=batch_size,
            ).prev_sample

        _, latents = latents.chunk(2)


        return latents 