import math
import numbers
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AsymmetricAutoencoderKL, ImageProjection
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
    retrieve_timesteps,
)
from diffusers.utils import deprecate


class RASGAttnProcessor:
    def __init__(self, mask, token_idx, scale_factor):
        self.attention_scores = None  # Stores the last output of the similarity matrix here. Each layer will get its own RASGAttnProcessor assigned
        self.mask = mask
        self.token_idx = token_idx
        self.scale_factor = scale_factor
        self.mask_resoltuion = mask.shape[-1] * mask.shape[-2]  # 64 x 64 if the image is 512x512

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        # Same as the default AttnProcessor up untill the part where similarity matrix gets saved
        downscale_factor = self.mask_resoltuion // hidden_states.shape[1]
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Automatically recognize the resolution and save the attention similarity values
        # We need to use the values before the softmax function, hence the rewritten get_attention_scores function.
        if downscale_factor == self.scale_factor**2:
            self.attention_scores = get_attention_scores(attn, query, key, attention_mask)
            attention_probs = self.attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)  # Original code

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PAIntAAttnProcessor:
    def __init__(self, transformer_block, mask, token_idx, do_classifier_free_guidance, scale_factors):
        self.transformer_block = transformer_block  # Stores the parent transformer block.
        self.mask = mask
        self.scale_factors = scale_factors
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.token_idx = token_idx
        self.shape = mask.shape[2:]
        self.mask_resoltuion = mask.shape[-1] * mask.shape[-2]  # 64 x 64
        self.default_processor = AttnProcessor()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        # Automatically recognize the resolution of the current attention layer and resize the masks accordingly
        downscale_factor = self.mask_resoltuion // hidden_states.shape[1]

        mask = None
        for factor in self.scale_factors:
            if downscale_factor == factor**2:
                shape = (self.shape[0] // factor, self.shape[1] // factor)
                mask = F.interpolate(self.mask, shape, mode="bicubic")  # B, 1, H, W
                break
        if mask is None:
            return self.default_processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

        # STARTS HERE
        residual = hidden_states
        # Save the input hidden_states for later use
        input_hidden_states = hidden_states

        # ================================================== #
        # =============== SELF ATTENTION 1 ================= #
        # ================================================== #

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # self_attention_probs = attn.get_attention_scores(query, key, attention_mask) # We can't use post-softmax attention scores in this case
        self_attention_scores = get_attention_scores(
            attn, query, key, attention_mask
        )  # The custom function returns pre-softmax probabilities
        self_attention_probs = self_attention_scores.softmax(
            dim=-1
        )  # Manually compute the probabilities here, the scores will be reused in the second part of PAIntA
        self_attention_probs = self_attention_probs.to(query.dtype)

        hidden_states = torch.bmm(self_attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # x = x + self.attn1(self.norm1(x))

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:  # So many residuals everywhere
            hidden_states = hidden_states + residual

        self_attention_output_hidden_states = hidden_states / attn.rescale_output_factor

        # ================================================== #
        # ============ BasicTransformerBlock =============== #
        # ================================================== #
        # We use a hack by running the code from the BasicTransformerBlock that is between Self and Cross attentions here
        # The other option would've been modifying the BasicTransformerBlock and adding this functionality here.
        # I assumed that changing the BasicTransformerBlock would have been a bigger deal and decided to use this hack isntead.

        # The SelfAttention block recieves the normalized latents from the BasicTransformerBlock,
        # But the residual of the output is the non-normalized version.
        # Therefore we unnormalize the input hidden state here
        unnormalized_input_hidden_states = (
            input_hidden_states + self.transformer_block.norm1.bias
        ) * self.transformer_block.norm1.weight

        # TODO: return if neccessary
        # if self.use_ada_layer_norm_zero:
        #     attn_output = gate_msa.unsqueeze(1) * attn_output
        # elif self.use_ada_layer_norm_single:
        #     attn_output = gate_msa * attn_output

        transformer_hidden_states = self_attention_output_hidden_states + unnormalized_input_hidden_states
        if transformer_hidden_states.ndim == 4:
            transformer_hidden_states = transformer_hidden_states.squeeze(1)

        # TODO: return if neccessary
        # 2.5 GLIGEN Control
        # if gligen_kwargs is not None:
        #     transformer_hidden_states = self.fuser(transformer_hidden_states, gligen_kwargs["objs"])
        # NOTE: we experimented with using GLIGEN and HDPainter together, the results were not that great

        # 3. Cross-Attention
        if self.transformer_block.use_ada_layer_norm:
            # transformer_norm_hidden_states = self.transformer_block.norm2(transformer_hidden_states, timestep)
            raise NotImplementedError()
        elif self.transformer_block.use_ada_layer_norm_zero or self.transformer_block.use_layer_norm:
            transformer_norm_hidden_states = self.transformer_block.norm2(transformer_hidden_states)
        elif self.transformer_block.use_ada_layer_norm_single:
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            transformer_norm_hidden_states = transformer_hidden_states
        elif self.transformer_block.use_ada_layer_norm_continuous:
            # transformer_norm_hidden_states = self.transformer_block.norm2(transformer_hidden_states, added_cond_kwargs["pooled_text_emb"])
            raise NotImplementedError()
        else:
            raise ValueError("Incorrect norm")

        if self.transformer_block.pos_embed is not None and self.transformer_block.use_ada_layer_norm_single is False:
            transformer_norm_hidden_states = self.transformer_block.pos_embed(transformer_norm_hidden_states)

        # ================================================== #
        # ================= CROSS ATTENTION ================ #
        # ================================================== #

        # We do an initial pass of the CrossAttention up to obtaining the similarity matrix here.
        # The similarity matrix is used to obtain scaling coefficients for the attention matrix of the self attention
        # We reuse the previously computed self-attention matrix, and only repeat the steps after the softmax

        cross_attention_input_hidden_states = (
            transformer_norm_hidden_states  # Renaming the variable for the sake of readability
        )

        # TODO: check if classifier_free_guidance is being used before splitting here
        if self.do_classifier_free_guidance:
            # Our scaling coefficients depend only on the conditional part, so we split the inputs
            (
                _cross_attention_input_hidden_states_unconditional,
                cross_attention_input_hidden_states_conditional,
            ) = cross_attention_input_hidden_states.chunk(2)

            # Same split for the encoder_hidden_states i.e. the tokens
            # Since the SelfAttention processors don't get the encoder states as input, we inject them into the processor in the begining.
            _encoder_hidden_states_unconditional, encoder_hidden_states_conditional = self.encoder_hidden_states.chunk(
                2
            )
        else:
            cross_attention_input_hidden_states_conditional = cross_attention_input_hidden_states
            encoder_hidden_states_conditional = self.encoder_hidden_states.chunk(2)

        # Rename the variables for the sake of readability
        # The part below is the beginning of the __call__ function of the following CrossAttention layer
        cross_attention_hidden_states = cross_attention_input_hidden_states_conditional
        cross_attention_encoder_hidden_states = encoder_hidden_states_conditional

        attn2 = self.transformer_block.attn2

        if attn2.spatial_norm is not None:
            cross_attention_hidden_states = attn2.spatial_norm(cross_attention_hidden_states, temb)

        input_ndim = cross_attention_hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = cross_attention_hidden_states.shape
            cross_attention_hidden_states = cross_attention_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        (
            batch_size,
            sequence_length,
            _,
        ) = cross_attention_hidden_states.shape  # It is definitely a cross attention, so no need for an if block
        # TODO: change the attention_mask here
        attention_mask = attn2.prepare_attention_mask(
            None, sequence_length, batch_size
        )  # I assume the attention mask is the same...

        if attn2.group_norm is not None:
            cross_attention_hidden_states = attn2.group_norm(cross_attention_hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query2 = attn2.to_q(cross_attention_hidden_states)

        if attn2.norm_cross:
            cross_attention_encoder_hidden_states = attn2.norm_encoder_hidden_states(
                cross_attention_encoder_hidden_states
            )

        key2 = attn2.to_k(cross_attention_encoder_hidden_states)
        query2 = attn2.head_to_batch_dim(query2)
        key2 = attn2.head_to_batch_dim(key2)

        cross_attention_probs = attn2.get_attention_scores(query2, key2, attention_mask)

        # CrossAttention ends here, the remaining part is not used

        # ================================================== #
        # ================ SELF ATTENTION 2 ================ #
        # ================================================== #
        # DEJA VU!

        mask = (mask > 0.5).to(self_attention_output_hidden_states.dtype)
        m = mask.to(self_attention_output_hidden_states.device)
        # m = rearrange(m, 'b c h w -> b (h w) c').contiguous()
        m = m.permute(0, 2, 3, 1).reshape((m.shape[0], -1, m.shape[1])).contiguous()  # B HW 1
        m = torch.matmul(m, m.permute(0, 2, 1)) + (1 - m)

        # # Compute scaling coefficients for the similarity matrix
        # # Select the cross attention values for the correct tokens only!
        # cross_attention_probs = cross_attention_probs.mean(dim = 0)
        # cross_attention_probs = cross_attention_probs[:, self.token_idx].sum(dim=1)

        # cross_attention_probs = cross_attention_probs.reshape(shape)
        # gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).to(self_attention_output_hidden_states.device)
        # cross_attention_probs = gaussian_smoothing(cross_attention_probs.unsqueeze(0))[0] # optional smoothing
        # cross_attention_probs = cross_attention_probs.reshape(-1)
        # cross_attention_probs = ((cross_attention_probs - torch.median(cross_attention_probs.ravel())) / torch.max(cross_attention_probs.ravel())).clip(0, 1)

        # c = (1 - m) * cross_attention_probs.reshape(1, 1, -1) + m # PAIntA scaling coefficients

        # Compute scaling coefficients for the similarity matrix
        # Select the cross attention values for the correct tokens only!

        batch_size, dims, channels = cross_attention_probs.shape
        batch_size = batch_size // attn.heads
        cross_attention_probs = cross_attention_probs.reshape((batch_size, attn.heads, dims, channels))  # B, D, HW, T

        cross_attention_probs = cross_attention_probs.mean(dim=1)  # B, HW, T
        cross_attention_probs = cross_attention_probs[..., self.token_idx].sum(dim=-1)  # B, HW
        cross_attention_probs = cross_attention_probs.reshape((batch_size,) + shape)  # , B, H, W

        gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).to(
            self_attention_output_hidden_states.device
        )
        cross_attention_probs = gaussian_smoothing(cross_attention_probs[:, None])[:, 0]  # optional smoothing B, H, W

        # Median normalization
        cross_attention_probs = cross_attention_probs.reshape(batch_size, -1)  # B, HW
        cross_attention_probs = (
            cross_attention_probs - cross_attention_probs.median(dim=-1, keepdim=True).values
        ) / cross_attention_probs.max(dim=-1, keepdim=True).values
        cross_attention_probs = cross_attention_probs.clip(0, 1)

        c = (1 - m) * cross_attention_probs.reshape(batch_size, 1, -1) + m
        c = c.repeat_interleave(attn.heads, 0)  # BD, HW
        if self.do_classifier_free_guidance:
            c = torch.cat([c, c])  # 2BD, HW

        # Rescaling the original self-attention matrix
        self_attention_scores_rescaled = self_attention_scores * c
        self_attention_probs_rescaled = self_attention_scores_rescaled.softmax(dim=-1)

        # Continuing the self attention normally using the new matrix
        hidden_states = torch.bmm(self_attention_probs_rescaled, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + input_hidden_states

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StableDiffusionHDPainterPipeline(StableDiffusionInpaintPipeline):
    def get_tokenized_prompt(self, prompt):
        out = self.tokenizer(prompt)
        return [self.tokenizer.decode(x) for x in out["input_ids"]]

    def init_attn_processors(
        self,
        mask,
        token_idx,
        use_painta=True,
        use_rasg=True,
        painta_scale_factors=[2, 4],  # 64x64 -> [16x16, 32x32]
        rasg_scale_factor=4,  # 64x64 -> 16x16
        self_attention_layer_name="attn1",
        cross_attention_layer_name="attn2",
        list_of_painta_layer_names=None,
        list_of_rasg_layer_names=None,
    ):
        default_processor = AttnProcessor()
        width, height = mask.shape[-2:]
        width, height = width // self.vae_scale_factor, height // self.vae_scale_factor

        painta_scale_factors = [x * self.vae_scale_factor for x in painta_scale_factors]
        rasg_scale_factor = self.vae_scale_factor * rasg_scale_factor

        attn_processors = {}
        for x in self.unet.attn_processors:
            if (list_of_painta_layer_names is None and self_attention_layer_name in x) or (
                list_of_painta_layer_names is not None and x in list_of_painta_layer_names
            ):
                if use_painta:
                    transformer_block = self.unet.get_submodule(x.replace(".attn1.processor", ""))
                    attn_processors[x] = PAIntAAttnProcessor(
                        transformer_block, mask, token_idx, self.do_classifier_free_guidance, painta_scale_factors
                    )
                else:
                    attn_processors[x] = default_processor
            elif (list_of_rasg_layer_names is None and cross_attention_layer_name in x) or (
                list_of_rasg_layer_names is not None and x in list_of_rasg_layer_names
            ):
                if use_rasg:
                    attn_processors[x] = RASGAttnProcessor(mask, token_idx, rasg_scale_factor)
                else:
                    attn_processors[x] = default_processor

        self.unet.set_attn_processor(attn_processors)
        # import json
        # with open('/home/hayk.manukyan/repos/diffusers/debug.txt', 'a')  as f:
        #     json.dump({x:str(y) for x,y in self.unet.attn_processors.items()}, f, indent=4)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.Tensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        positive_prompt: Optional[str] = "",
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.01,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_painta=True,
        use_rasg=True,
        self_attention_layer_name=".attn1",
        cross_attention_layer_name=".attn2",
        painta_scale_factors=[2, 4],  # 16 x 16 and 32 x 32
        rasg_scale_factor=4,  # 16x16 by default
        list_of_painta_layer_names=None,
        list_of_rasg_layer_names=None,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        #
        prompt_no_positives = prompt
        if isinstance(prompt, list):
            prompt = [x + positive_prompt for x in prompt]
        else:
            prompt = prompt + positive_prompt

        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            mask_image,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            padding_mask_crop,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # assert batch_size == 1, "Does not work with batch size > 1 currently"

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image

        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 7.5 Setting up HD-Painter

        # Get the indices of the tokens to be modified by both RASG and PAIntA
        token_idx = list(range(1, self.get_tokenized_prompt(prompt_no_positives).index("<|endoftext|>"))) + [
            self.get_tokenized_prompt(prompt).index("<|endoftext|>")
        ]

        # Setting up the attention processors
        self.init_attn_processors(
            mask_condition,
            token_idx,
            use_painta,
            use_rasg,
            painta_scale_factors=painta_scale_factors,
            rasg_scale_factor=rasg_scale_factor,
            self_attention_layer_name=self_attention_layer_name,
            cross_attention_layer_name=cross_attention_layer_name,
            list_of_painta_layer_names=list_of_painta_layer_names,
            list_of_rasg_layer_names=list_of_rasg_layer_names,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if use_rasg:
            extra_step_kwargs["generator"] = None

        # 9.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 9.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        painta_active = True

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if t < 500 and painta_active:
                    self.init_attn_processors(
                        mask_condition,
                        token_idx,
                        False,
                        use_rasg,
                        painta_scale_factors=painta_scale_factors,
                        rasg_scale_factor=rasg_scale_factor,
                        self_attention_layer_name=self_attention_layer_name,
                        cross_attention_layer_name=cross_attention_layer_name,
                        list_of_painta_layer_names=list_of_painta_layer_names,
                        list_of_rasg_layer_names=list_of_rasg_layer_names,
                    )
                    painta_active = False

                with torch.enable_grad():
                    self.unet.zero_grad()
                    latents = latents.detach()
                    latents.requires_grad = True

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if num_channels_unet == 9:
                        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                    self.scheduler.latents = latents
                    self.encoder_hidden_states = prompt_embeds
                    for attn_processor in self.unet.attn_processors.values():
                        attn_processor.encoder_hidden_states = prompt_embeds

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if use_rasg:
                        # Perform RASG
                        _, _, height, width = mask_condition.shape  # 512 x 512
                        scale_factor = self.vae_scale_factor * rasg_scale_factor  # 8 * 4 = 32

                        # TODO: Fix for > 1 batch_size
                        rasg_mask = F.interpolate(
                            mask_condition, (height // scale_factor, width // scale_factor), mode="bicubic"
                        )[0, 0]  # mode is nearest by default, B, H, W

                        # Aggregate the saved attention maps
                        attn_map = []
                        for processor in self.unet.attn_processors.values():
                            if hasattr(processor, "attention_scores") and processor.attention_scores is not None:
                                if self.do_classifier_free_guidance:
                                    attn_map.append(processor.attention_scores.chunk(2)[1])  # (B/2) x H, 256, 77
                                else:
                                    attn_map.append(processor.attention_scores)  # B x H, 256, 77 ?

                        attn_map = (
                            torch.cat(attn_map)
                            .mean(0)
                            .permute(1, 0)
                            .reshape((-1, height // scale_factor, width // scale_factor))
                        )  # 77, 16, 16

                        # Compute the attention score
                        attn_score = -sum(
                            [
                                F.binary_cross_entropy_with_logits(x - 1.0, rasg_mask.to(device))
                                for x in attn_map[token_idx]
                            ]
                        )

                        # Backward the score and compute the gradients
                        attn_score.backward()

                        # Normalzie the gradients and compute the noise component
                        variance_noise = latents.grad.detach()
                        # print("VARIANCE SHAPE", variance_noise.shape)
                        variance_noise -= torch.mean(variance_noise, [1, 2, 3], keepdim=True)
                        variance_noise /= torch.std(variance_noise, [1, 2, 3], keepdim=True)
                    else:
                        variance_noise = None

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False, variance_noise=variance_noise
                )[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    mask = callback_outputs.pop("mask", mask)
                    masked_image_latents = callback_outputs.pop("masked_image_latents", masked_image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False, generator=generator, **condition_kwargs
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if padding_mask_crop is not None:
            image = [self.image_processor.apply_overlay(mask_image, original_image, i, crops_coords) for i in image]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


# ============= Utility Functions ============== #


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups, padding="same")


def get_attention_scores(
    self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=self.scale,
    )
    del baddbmm_input

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    return attention_scores
