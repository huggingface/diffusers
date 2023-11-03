# Copyright 2023 FABRIC authors and the HuggingFace Team. All rights reserved.
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

import torch
from diffuser.utils.torch_utils import randn_tensor
from packaging import version
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import EulerAncestralDiscreteScheduler, KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import DiffusionPipeline
        >>> import torch

        >>> model_id = "dreamlike-art/dreamlike-photoreal-2.0"
        >>> pipe = DiffusionPipeline(model_id, torch_dtype=torch.float16, custom_pipeline="pipeline_fabric")
        >>> pipe = pipe.to("cuda")
        >>> prompt = "a giant standing in a fantasy landscape best quality"
        >>> liked = []  # list of images for positive feedback
        >>> disliked = []  # list of images for negative feedback
        >>> image = pipe(prompt, num_images=4, liked=liked, disliked=disliked).images[0]
        ```
"""


class FabricCrossAttnProcessor:
    def __init__(self):
        self.attntion_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        weights=None,
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
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerAncestralDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
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
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
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

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

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

            # textual inversion: procecss multi-vector tokens if necessary
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

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

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
                            attn_with_weights = FabricCrossAttnProcessor()
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
                            attn_with_weights = FabricCrossAttnProcessor()
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

    def preprocess_feedback_images(self, images, vae, dim, device, dtype, generator) -> torch.tensor:
        images_t = [self.image_to_tensor(img, dim, dtype) for img in images]
        images_t = torch.stack(images_t).to(device)
        latents = vae.config.scaling_factor * vae.encode(images_t).latent_dist.sample(generator)

        return torch.cat([latents], dim=0)

    def check_inputs(
        self,
        prompt,
        negative_prompt=None,
        liked=None,
        disliked=None,
        height=None,
        width=None,
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

        if height is not None and not isinstance(height, int):
            raise ValueError(f"`height` has to be of type `int` but is {type(height)}")

        if width is not None and not isinstance(width, int):
            raise ValueError(f"`width` has to be of type `int` but is {type(width)}")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = "lowres, bad anatomy, bad hands, cropped, worst quality",
        liked: Optional[Union[List[str], List[Image.Image]]] = [],
        disliked: Optional[Union[List[str], List[Image.Image]]] = [],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        height: int = 512,
        width: int = 512,
        return_dict: bool = True,
        num_images: int = 4,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 20,
        output_type: Optional[str] = "pil",
        feedback_start_ratio: float = 0.33,
        feedback_end_ratio: float = 0.66,
        min_weight: float = 0.05,
        max_weight: float = 0.8,
        neg_scale: float = 0.5,
        pos_bottleneck_scale: float = 1.0,
        neg_bottleneck_scale: float = 1.0,
        latents: Optional[torch.FloatTensor] = None,
    ):
        r"""
        The call function to the pipeline for generation. Generate a trajectory of images with binary feedback. The
        feedback can be given as a list of liked and disliked images.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            liked (`List[Image.Image]` or `List[str]`, *optional*):
                Encourages images with liked features.
            disliked (`List[Image.Image]` or `List[str]`, *optional*):
                Discourages images with disliked features.
            generator (`torch.Generator` or `List[torch.Generator]` or `int`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) or an `int` to
                make generation deterministic.
            height (`int`, *optional*, defaults to 512):
                Height of the generated image.
            width (`int`, *optional*, defaults to 512):
                Width of the generated image.
            num_images (`int`, *optional*, defaults to 4):
                The number of images to generate per prompt.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            feedback_start_ratio (`float`, *optional*, defaults to `.33`):
                Start point for providing feedback (between 0 and 1).
            feedback_end_ratio (`float`, *optional*, defaults to `.66`):
                End point for providing feedback (between 0 and 1).
            min_weight (`float`, *optional*, defaults to `.05`):
                Minimum weight for feedback.
            max_weight (`float`, *optional*, defults tp `1.0`):
                Maximum weight for feedback.
            neg_scale (`float`, *optional*, defaults to `.5`):
                Scale factor for negative feedback.

        Examples:

        Returns:
            [`~pipelines.fabric.FabricPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.

        """

        self.check_inputs(prompt, negative_prompt, liked, disliked)

        device = self._execution_device
        dtype = self.unet.dtype

        if isinstance(prompt, str) and prompt is not None:
            batch_size = 1
        elif isinstance(prompt, list) and prompt is not None:
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if isinstance(negative_prompt, str):
            negative_prompt = negative_prompt
        elif isinstance(negative_prompt, list):
            negative_prompt = negative_prompt
        else:
            assert len(negative_prompt) == batch_size

        shape = (
            batch_size * num_images,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latent_noise = randn_tensor(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        positive_latents = (
            self.preprocess_feedback_images(liked, self.vae, (height, width), device, dtype, generator)
            if liked and len(liked) > 0
            else torch.tensor(
                [],
                device=device,
                dtype=dtype,
            )
        )
        negative_latents = (
            self.preprocess_feedback_images(disliked, self.vae, (height, width), device, dtype, generator)
            if disliked and len(disliked) > 0
            else torch.tensor(
                [],
                device=device,
                dtype=dtype,
            )
        )

        do_classifier_free_guidance = guidance_scale > 0.1

        (prompt_neg_embs, prompt_pos_embs) = self._encode_prompt(
            prompt,
            device,
            num_images,
            do_classifier_free_guidance,
            negative_prompt,
        ).split([num_images * batch_size, num_images * batch_size])

        batched_prompt_embd = torch.cat([prompt_pos_embs, prompt_neg_embs], dim=0)

        null_tokens = self.tokenizer(
            [""],
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
        )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = null_tokens.attention_mask.to(device)
        else:
            attention_mask = None

        null_prompt_emb = self.text_encoder(
            input_ids=null_tokens.input_ids.to(device),
            attention_mask=attention_mask,
        ).last_hidden_state

        null_prompt_emb = null_prompt_emb.to(device=device, dtype=dtype)

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
                )[0]

                noise_cond, noise_uncond = unet_out.chunk(2)
                guidance = noise_cond - noise_uncond
                noise_pred = noise_uncond + guidance_scale * guidance
                latent_noise = self.scheduler.step(noise_pred, t, latent_noise)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    pbar.update()

        y = self.vae.decode(latent_noise / self.vae.config.scaling_factor, return_dict=False)[0]
        imgs = self.image_processor.postprocess(
            y,
            output_type=output_type,
        )

        if not return_dict:
            return imgs

        return StableDiffusionPipelineOutput(imgs, False)

    def image_to_tensor(self, image: Union[str, Image.Image], dim: tuple, dtype):
        """
        Convert latent PIL image to a torch tensor for further processing.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.image_processor.preprocess(image, height=dim[0], width=dim[1])[0]
        return image.type(dtype)
