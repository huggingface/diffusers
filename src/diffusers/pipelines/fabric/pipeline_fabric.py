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
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from packaging import version
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention import BasicTransformerBlock
from ...models.attention_processor import LoRAAttnProcessor
from ...schedulers import EulerAncestralDiscreteScheduler
from ...utils import (
    deprecate,
    logging,
)
from ..pipeline_utils import DiffusionPipeline
from . import FabricPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CrossAttnProcessor:
    def __init__(self):
        self.attntion_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        weights=None,  # shape: (batch_size, sequence_length)
        lora_scale=1.0,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if isinstance(attn.processor, LoRAAttnProcessor):
            query = attn.to_q(hidden_states) + lora_scale * attn.processor.to_q_lora(hidden_states)
        else:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if isinstance(attn.processor, LoRAAttnProcessor):
            key = attn.to_k(encoder_hidden_states) + lora_scale * attn.processor.to_k_lora(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states) + lora_scale * attn.processor.to_v_lora(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if weights is not None:
            if weights.shape[0] != 1:
                weights = weights.repeat_interleave(attn.heads, dim=0)
            attention_probs = attention_probs * weights[:, None]
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        if isinstance(attn.processor, LoRAAttnProcessor):
            hidden_states = attn.to_out[0](hidden_states) + lora_scale * attn.processor.to_out_lora(hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class FabricPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion and conditioning the results using feedback images.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`EulerAncestralDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: EulerAncestralDiscreteScheduler,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )

            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def initialize_prompts(self, prompts: List[str], device):
        # Breaking into individual prompts feels memory efficient
        prompt_embed_list = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
            )

            attention_mask = (
                prompt_tokens.attention_mask.to(device)
                if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
                )
                else None
            )

            prompt_embd = self.text_encoder(
                input_ids=prompt_tokens.input_ids.to(device),
                attention_mask=attention_mask,
            ).last_hidden_state

            prompt_embed_list.append(prompt_embd)

        all_prompt_embed = torch.cat(prompt_embed_list, dim=0)
        return all_prompt_embed

    def get_unet_hidden_states(self, z_all, t, prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):

                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)

                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)

        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return cached_hidden_states

    def unet_forward_with_cached_hidden_states(
        self,
        z_all,
        t,
        prompt_embd,
        cached_pos_hiddens: Optional[List[torch.Tensor]] = None,
        cached_neg_hiddens: Optional[List[torch.Tensor]] = None,
        pos_weights=(0.8, 0.8),
        neg_weights=(0.5, 0.5),
    ):
        if cached_pos_hiddens is None and cached_neg_hiddens is None:
            return self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        local_pos_weights = torch.linspace(*pos_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
        local_neg_weights = torch.linspace(*neg_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
        for block, pos_weight, neg_weight in zip(
            self.unet.down_blocks + [self.unet.mid_block] + self.unet.up_blocks,
            local_pos_weights + [pos_weights[1]] + local_pos_weights[::-1],
            local_neg_weights + [neg_weights[1]] + local_neg_weights[::-1],
        ):
            for module in block.modules():
                if isinstance(module, BasicTransformerBlock):

                    def new_forward(
                        self,
                        hidden_states,
                        pos_weight=pos_weight,
                        neg_weight=neg_weight,
                        **kwargs,
                    ):
                        cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0)
                        batch_size, d_model = cond_hiddens.shape[:2]
                        device, dtype = hidden_states.device, hidden_states.dtype

                        weights = torch.ones(batch_size, d_model, device=device, dtype=dtype)

                        out_pos = self.old_forward(hidden_states)
                        out_neg = self.old_forward(hidden_states)

                        if cached_pos_hiddens is not None:
                            cached_pos_hs = cached_pos_hiddens.pop(0).to(hidden_states.device)
                            cond_pos_hs = torch.cat([cond_hiddens, cached_pos_hs], dim=1)
                            pos_weights = weights.clone().repeat(1, 1 + cached_pos_hs.shape[1] // d_model)
                            pos_weights[:, d_model:] = pos_weight
                            attn_with_weights = CrossAttnProcessor()
                            out_pos = attn_with_weights(
                                self,
                                cond_hiddens,
                                encoder_hidden_states=cond_pos_hs,
                                weights=pos_weights,
                            )
                        else:
                            out_pos = self.old_forward(cond_hiddens)

                        if cached_neg_hiddens is not None:
                            cached_neg_hs = cached_neg_hiddens.pop(0).to(hidden_states.device)
                            uncond_neg_hs = torch.cat([uncond_hiddens, cached_neg_hs], dim=1)
                            neg_weights = weights.clone().repeat(1, 1 + cached_neg_hs.shape[1] // d_model)
                            neg_weights[:, d_model:] = neg_weight
                            attn_with_weights = CrossAttnProcessor()
                            out_neg = attn_with_weights(
                                self,
                                uncond_hiddens,
                                encoder_hidden_states=uncond_neg_hs,
                                weights=neg_weights,
                            )
                        else:
                            out_neg = self.old_forward(uncond_hiddens)

                        out = torch.cat([out_pos, out_neg], dim=0)
                        return out

                    module.attn1.old_forward = module.attn1.forward
                    module.attn1.forward = new_forward.__get__(module.attn1)

        out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return out

    def preprocess_feedback_images(self, images, vae, device, dtype) -> torch.tensor:
        images_t = [self.image_to_tensor(img, dtype) for img in images]
        images_t = torch.stack(images_t).to(device)
        latents = vae.config.scaling_factor * vae.encode(images_t).latent_dist.sample()
        return latents

    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        # we always cast to float32 as this does not cause significant overhead and is compatible             with bfloat16
        return image

    def check_inputs(
        self,
        prompt,
        negative_prompt=None,
        liked=None,
        disliked=None,
    ):
        if prompt is None:
            raise ValueError("Provide `prompt`. Cannot leave both `prompt` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if liked is not None and not isinstance(liked, list):
            raise ValueError(f"`liked` has to be of type `list` but is {type(liked)}")

        if disliked is not None and not isinstance(disliked, list):
            raise ValueError(f"`disliked` has to be of type `list` but is {type(disliked)}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = "lowres, bad anatomy, bad hands, cropped, worst quality",
        liked: Optional[Union[List[str], List[Image.Image]]] = [],
        disliked: Optional[Union[List[str], List[Image.Image]]] = [],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        num_images: int = 4,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 20,
        feedback_start_ratio: float = 0.33,
        feedback_end_ratio: float = 0.66,
        min_weight: float = 0.05,
        max_weight: float = 0.8,
        neg_scale: float = 0.5,
        pos_bottleneck_scale: float = 1.0,
        neg_bottleneck_scale: float = 1.0,
        output_type: Optional[str] = "pil",
        latents: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation. Generate a trajectory of images with binary
        feedback. The feedback can be given as a list of liked and disliked images.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            liked (`List[Image.Image]` or `List[str]`, *optional*):
                Liked enables feedback through images, encourages images with liked features.
            disliked (`List[Image.Image]` or `List[str]`, *optional*):
                Disliked enables feedback through images, discourages images with disliked features.
            generator (`torch.Generator` or `List[torch.Generator]` or `int`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html),
                can be int. to make generation deterministic.
            num_images (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.

        Examples:
            >>> from diffusers import FabricPipeline >>> import torch >>> model_id =
            "dreamlike-art/dreamlike-photoreal-2.0" >>> pipe = FabricPipeline(model_id, torch_dtype = torch.float16)
            >>> pipe = pipe.to("cuda") >>> prompt = "a giant standing in a fantasy landscape best quality" >>> liked =
            [] >>> disliked = [] >>> image = pipe(prompt, num_images=4, liked=liked,disliked=disliked).images

        Returns:
            [`~pipelines.fabric.FabricPipelineOutput`] or `tuple`: When returning a tuple, the first element is a list
            with the generated images, and the second element is a list of `bool`s denoting whether the corresponding
            generated image likely represents "not-safe-for-work" (nsfw) content, according to the `safety_checker`.

        """

        self.check_inputs(prompt, negative_prompt, liked, disliked)

        device = self._execution_device
        dtype = self.unet.dtype

        latent_noise = torch.randn(num_images, 4, 64, 64, device=device, dtype=dtype)

        positive_latents = (
            self.preprocess_feedback_images(liked, self.vae, device, dtype)
            if liked and len(liked) > 0
            else torch.tensor([], device=device, dtype=dtype)
        )
        negative_latents = (
            self.preprocess_feedback_images(disliked, self.vae, device, dtype)
            if disliked and len(disliked) > 0
            else torch.tensor([], device=device, dtype=dtype)
        )

        if isinstance(prompt, str) and prompt is not None:
            batch_size = 1
        elif isinstance(prompt, list) and prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = None

        prompt = [prompt] * num_images

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * num_images
        elif isinstance(negative_prompt, list):
            negative_prompt = negative_prompt
        else:
            assert len(negative_prompt) == num_images

        (cond_prompt_embs, uncond_prompt_embs, null_prompt_emb) = self.initialize_prompts(
            prompt + negative_prompt + [""], device
        ).split([num_images, num_images, batch_size * num_images])

        batched_prompt_embd = torch.cat([cond_prompt_embs, uncond_prompt_embs], dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latent_noise = latent_noise * self.scheduler.init_noise_sigma

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        ref_start_idx = round(len(timesteps) * feedback_start_ratio)
        ref_end_idx = round(len(timesteps) * feedback_end_ratio)

        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                sigma = self.scheduler.sigma_t[t] if hasattr(self.scheduler, "sigma_t") else 0
                if hasattr(self.scheduler, "sigmas"):
                    sigma = self.scheduler.sigmas[i]

                alpha_hat = 1 / (sigma**2 + 1)

                z_single = self.scheduler.scale_model_input(latent_noise, t)
                z_all = torch.cat([z_single] * 2, dim=0)
                z_ref = torch.cat([positive_latents, negative_latents], dim=0)

                if i >= ref_start_idx and i <= ref_end_idx:
                    weight_factor = max_weight
                else:
                    weight_factor = min_weight

                pos_ws = (weight_factor, weight_factor * pos_bottleneck_scale)
                neg_ws = (weight_factor * neg_scale, weight_factor * neg_scale * neg_bottleneck_scale)

                if z_ref.size(0) > 0 and weight_factor > 0:
                    noise = torch.randn_like(z_ref)
                    if isinstance(self.scheduler, EulerAncestralDiscreteScheduler):
                        z_ref_noised = (alpha_hat**0.5 * z_ref + (1 - alpha_hat) ** 0.5 * noise).type(dtype)
                    else:
                        z_ref_noised = self.scheduler.add_noise(z_ref, noise, t)

                    ref_prompt_embd = torch.cat(
                        [null_prompt_emb] * (len(positive_latents) + len(negative_latents)), dim=0
                    )
                    cached_hidden_states = self.get_unet_hidden_states(z_ref_noised, t, ref_prompt_embd)

                    n_pos, n_neg = positive_latents.shape[0], negative_latents.shape[0]
                    cached_pos_hs, cached_neg_hs = [], []
                    for hs in cached_hidden_states:
                        cached_pos, cached_neg = hs.split([n_pos, n_neg], dim=0)
                        cached_pos = cached_pos.view(1, -1, *cached_pos.shape[2:]).expand(num_images, -1, -1)
                        cached_neg = cached_neg.view(1, -1, *cached_neg.shape[2:]).expand(num_images, -1, -1)
                        cached_pos_hs.append(cached_pos)
                        cached_neg_hs.append(cached_neg)

                    if n_pos == 0:
                        cached_pos_hs = None
                    if n_neg == 0:
                        cached_neg_hs = None
                else:
                    cached_pos_hs, cached_neg_hs = None, None

                unet_out = self.unet_forward_with_cached_hidden_states(
                    z_all,
                    t,
                    prompt_embd=batched_prompt_embd,
                    cached_pos_hiddens=cached_pos_hs,
                    cached_neg_hiddens=cached_neg_hs,
                    pos_weights=pos_ws,
                    neg_weights=neg_ws,
                ).sample

                noise_cond, noise_uncond = unet_out.chunk(2)
                guidance = noise_cond - noise_uncond
                noise_pred = noise_uncond + guidance_scale * guidance
                latent_noise = self.scheduler.step(noise_pred, t, latent_noise).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    pbar.update()

        y = self.decode_latents(latent_noise)
        imgs = self.image_processor.postprocess(y, output_type=output_type)

        if not return_dict:
            return imgs

        return FabricPipelineOutput(imgs, False)

    @staticmethod
    def image_to_tensor(image: Union[str, Image.Image], dtype):
        """
        Convert latent PIL image to a torch tensor for further processing.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.type(dtype)
