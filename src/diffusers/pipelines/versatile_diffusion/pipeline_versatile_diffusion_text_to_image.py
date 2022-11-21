# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
import torch.utils.checkpoint

import PIL
from transformers import CLIPProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from ...models import AutoencoderKL, UNet2DConditionModel, VQModel
from ...models.attention import Transformer2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


class VersatileDiffusionTextToImagePipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        bert ([`LDMBertModel`]):
            Text-encoder model based on [BERT](https://huggingface.co/docs/transformers/model_doc/bert) architecture.
        tokenizer (`transformers.BertTokenizer`):
            Tokenizer of class
            [BertTokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    tokenizer: CLIPTokenizer
    image_processor: CLIPProcessor
    text_encoder: CLIPTextModel
    image_encoder: CLIPVisionModel
    image_unet: UNet2DConditionModel
    text_unet: UNet2DConditionModel
    vae: Union[VQModel, AutoencoderKL]
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        image_processor: CLIPProcessor,
        text_encoder: CLIPTextModel,
        image_unet: UNet2DConditionModel,
        text_unet: UNet2DConditionModel,
        vae: Union[VQModel, AutoencoderKL],
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            image_processor=image_processor,
            text_encoder=text_encoder,
            image_unet=image_unet,
            text_unet=text_unet,
            vae=vae,
            scheduler=scheduler,
        )
        for name, module in self.image_unet.named_modules():
            if isinstance(module, Transformer2DModel):
                parent_name, index = name.rsplit(".", 1)
                index = int(index)
                self.image_unet.get_submodule(parent_name)[index], self.text_unet.get_submodule(parent_name)[index] = (
                    self.text_unet.get_submodule(parent_name)[index],
                    self.image_unet.get_submodule(parent_name)[index],
                )

    def _encode_prompt(self, prompt, do_classifier_free_guidance):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return embeds

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if do_classifier_free_guidance:
            uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))
            uncond_embeddings = normalize_embeddings(uncond_embeddings)

        # get prompt text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))
        text_embeddings = normalize_embeddings(text_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _encode_image_prompt(self, prompt, do_classifier_free_guidance):
        r"""
        Encodes the image prompt into image encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
            embeds = self.image_encoder.visual_projection(embeds)
            embeds_pooled = embeds[:, 0:1]
            embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
            return embeds

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if do_classifier_free_guidance:
            dummy_images = [np.zeros((512, 512, 3))] * batch_size
            dummy_images = self.image_processor(images=dummy_images, return_tensors="pt")
            uncond_embeddings = self.image_encoder(dummy_images.pixel_values.to(self.device))
            uncond_embeddings = normalize_embeddings(uncond_embeddings)

        # get prompt text embeddings
        image_input = self.image_processor(images=prompt, return_tensors="pt")
        image_embeddings = self.image_encoder(image_input.pixel_values.to(self.device))
        image_embeddings = normalize_embeddings(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and image embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([uncond_embeddings, image_embeddings])

        return image_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 1.0,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt` at
                the, usually at the expense of lower image quality.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        do_classifier_free_guidance = guidance_scale > 1.0

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        condition_embeddings = self._encode_prompt(prompt, do_classifier_free_guidance)

        latents = torch.randn(
            (batch_size, self.image_unet.in_channels, height // 8, width // 8), generator=generator, device=self.device
        )

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            if not do_classifier_free_guidance:
                latents_input = latents
            else:
                latents_input = torch.cat([latents] * 2)

            # predict the noise residual
            noise_pred = self.image_unet(latents_input, t, encoder_hidden_states=condition_embeddings).sample
            # perform guidance
            if guidance_scale != 1.0:
                noise_pred_uncond, noise_prediction_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


# class UNetMultiDimConditionModel(ModelMixin, ConfigMixin):
#     r"""
#     UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
#     and returns sample shaped output.
#
#     This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
#     implements for all the models (such as downloading or saving, etc.)
#
#     Parameters:
#         sample_size (`int`, *optional*): The size of the input sample.
#         in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
#         out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
#         center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
#         flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
#             Whether to flip the sin to cos in the time embedding.
#         freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
#         down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
#             The tuple of downsample blocks to use.
#         up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
#             The tuple of upsample blocks to use.
#         block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
#             The tuple of output channels for each block.
#         layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
#         downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
#         mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
#         act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
#         norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
#         norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
#         cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
#         attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
#     """
#
#     _supports_gradient_checkpointing = True
#
#     @register_to_config
#     def __init__(
#         self,
#         sample_size: Optional[int] = None,
#         in_channels: int = 4,
#         out_channels: int = 4,
#         center_input_sample: bool = False,
#         flip_sin_to_cos: bool = True,
#         freq_shift: int = 0,
#         down_block_types: Tuple[str] = (
#             "CrossAttnDownBlockMultiDim",
#             "CrossAttnDownBlockMultiDim",
#             "CrossAttnDownBlockMultiDim",
#             "DownBlockMultiDim",
#         ),
#         up_block_types: Tuple[str] = (
#                 "UpBlockMultiDim",
#                 "CrossAttnUpBlockMultiDim",
#                 "CrossAttnUpBlockMultiDim",
#                 "CrossAttnUpBlockMultiDim"
#         ),
#         block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
#         block_second_dim: Tuple[int] = (4, 4, 4, 4),
#         layers_per_block: int = 2,
#         downsample_padding: int = 1,
#         mid_block_scale_factor: float = 1,
#         act_fn: str = "silu",
#         norm_num_groups: int = 32,
#         norm_eps: float = 1e-5,
#         cross_attention_dim: int = 1280,
#         attention_head_dim: int = 8,
#     ):
#         super().__init__()
#
#         self.sample_size = sample_size
#         time_embed_dim = block_out_channels[0] * 4
#
#         # input
#         self.conv_in = LinearMultiDim([in_channels, 1, 1], block_out_channels[0], kernel_size=3, padding=(1, 1))
#
#         # time
#         self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
#         timestep_input_dim = block_out_channels[0]
#
#         self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
#
#         self.down_blocks = nn.ModuleList([])
#         self.mid_block = None
#         self.up_blocks = nn.ModuleList([])
#
#         # down
#         output_channel = block_out_channels[0]
#         for i, down_block_type in enumerate(down_block_types):
#             input_channel = output_channel
#             output_channel = block_out_channels[i]
#             is_final_block = i == len(block_out_channels) - 1
#
#             down_block = self.get_down_block(
#                 down_block_type,
#                 num_layers=layers_per_block,
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 temb_channels=time_embed_dim,
#                 add_downsample=not is_final_block,
#                 resnet_eps=norm_eps,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 cross_attention_dim=cross_attention_dim,
#                 attn_num_head_channels=attention_head_dim,
#                 downsample_padding=downsample_padding,
#             )
#             self.down_blocks.append(down_block)
#
#         # mid
#         self.mid_block = UNetMidBlockMultiDimCrossAttn(
#             in_channels=block_out_channels[-1],
#             temb_channels=time_embed_dim,
#             resnet_eps=norm_eps,
#             resnet_act_fn=act_fn,
#             output_scale_factor=mid_block_scale_factor,
#             resnet_time_scale_shift="default",
#             cross_attention_dim=cross_attention_dim,
#             attn_num_head_channels=attention_head_dim,
#             resnet_groups=norm_num_groups,
#         )
#
#         # count how many layers upsample the images
#         self.num_upsamplers = 0
#
#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             is_final_block = i == len(block_out_channels) - 1
#
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]
#             input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
#
#             # add upsample block for all BUT final layer
#             if not is_final_block:
#                 add_upsample = True
#                 self.num_upsamplers += 1
#             else:
#                 add_upsample = False
#
#             up_block = self.get_up_block(
#                 up_block_type,
#                 num_layers=layers_per_block + 1,
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 prev_output_channel=prev_output_channel,
#                 temb_channels=time_embed_dim,
#                 add_upsample=add_upsample,
#                 resnet_eps=norm_eps,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 cross_attention_dim=cross_attention_dim,
#                 attn_num_head_channels=attention_head_dim,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel
#
#         # out
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
#
#     def get_down_block(
#             down_block_type,
#             num_layers,
#             in_channels,
#             out_channels,
#             temb_channels,
#             add_downsample,
#             resnet_eps,
#             resnet_act_fn,
#             attn_num_head_channels,
#             resnet_groups=None,
#             cross_attention_dim=None,
#             downsample_padding=None,
#     ):
#         down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
#         if down_block_type == "DownBlockMultiDim":
#             return DownBlockMultiDim(
#                 num_layers=num_layers,
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 temb_channels=temb_channels,
#                 add_downsample=add_downsample,
#                 resnet_eps=resnet_eps,
#                 resnet_act_fn=resnet_act_fn,
#                 resnet_groups=resnet_groups,
#                 downsample_padding=downsample_padding,
#             )
#         elif down_block_type == "CrossAttnDownBlockMultiDim":
#             if cross_attention_dim is None:
#                 raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
#             return CrossAttnDownBlockMultiDim(
#                 num_layers=num_layers,
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 temb_channels=temb_channels,
#                 add_downsample=add_downsample,
#                 resnet_eps=resnet_eps,
#                 resnet_act_fn=resnet_act_fn,
#                 resnet_groups=resnet_groups,
#                 downsample_padding=downsample_padding,
#                 cross_attention_dim=cross_attention_dim,
#                 attn_num_head_channels=attn_num_head_channels,
#             )
#
#     def set_attention_slice(self, slice_size):
#         if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
#             raise ValueError(
#                 f"Make sure slice_size {slice_size} is a divisor of "
#                 f"the number of heads used in cross_attention {self.config.attention_head_dim}"
#             )
#         if slice_size is not None and slice_size > self.config.attention_head_dim:
#             raise ValueError(
#                 f"Chunk_size {slice_size} has to be smaller or equal to "
#                 f"the number of heads used in cross_attention {self.config.attention_head_dim}"
#             )
#
#         for block in self.down_blocks:
#             if hasattr(block, "attentions") and block.attentions is not None:
#                 block.set_attention_slice(slice_size)
#
#         self.mid_block.set_attention_slice(slice_size)
#
#         for block in self.up_blocks:
#             if hasattr(block, "attentions") and block.attentions is not None:
#                 block.set_attention_slice(slice_size)
#
#     def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
#         for block in self.down_blocks:
#             if hasattr(block, "attentions") and block.attentions is not None:
#                 block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)
#
#         self.mid_block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)
#
#         for block in self.up_blocks:
#             if hasattr(block, "attentions") and block.attentions is not None:
#                 block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)
#
#     def _set_gradient_checkpointing(self, module, value=False):
#         if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
#             module.gradient_checkpointing = value
#
#     def forward(
#         self,
#         sample: torch.FloatTensor,
#         timestep: Union[torch.Tensor, float, int],
#         encoder_hidden_states: torch.Tensor,
#         return_dict: bool = True,
#     ) -> Union[UNet2DConditionOutput, Tuple]:
#         r"""
#         Args:
#             sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
#             timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
#             encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
#
#         Returns:
#             [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
#             [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
#             returning a tuple, the first element is the sample tensor.
#         """
#         # By default samples have to be AT least a multiple of the overall upsampling factor.
#         # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
#         # However, the upsampling interpolation output size can be forced to fit any upsampling size
#         # on the fly if necessary.
#         default_overall_up_factor = 2**self.num_upsamplers
#
#         # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
#         forward_upsample_size = False
#         upsample_size = None
#
#         if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
#             logger.info("Forward upsample size to force interpolation output size.")
#             forward_upsample_size = True
#
#         # 0. center input if necessary
#         if self.config.center_input_sample:
#             sample = 2 * sample - 1.0
#
#         # 1. time
#         timesteps = timestep
#         if not torch.is_tensor(timesteps):
#             # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
#             timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
#         elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
#             timesteps = timesteps[None].to(sample.device)
#
#         # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#         timesteps = timesteps.expand(sample.shape[0])
#
#         t_emb = self.time_proj(timesteps)
#
#         # timesteps does not contain any weights and will always return f32 tensors
#         # but time_embedding might actually be running in fp16. so we need to cast here.
#         # there might be better ways to encapsulate this.
#         t_emb = t_emb.to(dtype=self.dtype)
#         emb = self.time_embedding(t_emb)
#
#         # 2. pre-process
#         sample = self.conv_in(sample)
#
#         # 3. down
#         down_block_res_samples = (sample,)
#         for downsample_block in self.down_blocks:
#             if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
#                 sample, res_samples = downsample_block(
#                     hidden_states=sample,
#                     temb=emb,
#                     encoder_hidden_states=encoder_hidden_states,
#                 )
#             else:
#                 sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
#
#             down_block_res_samples += res_samples
#
#         # 4. mid
#         sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
#
#         # 5. up
#         for i, upsample_block in enumerate(self.up_blocks):
#             is_final_block = i == len(self.up_blocks) - 1
#
#             res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
#             down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
#
#             # if we have not reached the final block and need to forward the
#             # upsample size, we do it here
#             if not is_final_block and forward_upsample_size:
#                 upsample_size = down_block_res_samples[-1].shape[2:]
#
#             if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
#                 sample = upsample_block(
#                     hidden_states=sample,
#                     temb=emb,
#                     res_hidden_states_tuple=res_samples,
#                     encoder_hidden_states=encoder_hidden_states,
#                     upsample_size=upsample_size,
#                 )
#             else:
#                 sample = upsample_block(
#                     hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
#                 )
#         # 6. post-process
#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)
#
#         if not return_dict:
#             return (sample,)
#
#         return UNet2DConditionOutput(sample=sample)
#
#
# class LinearMultiDim(nn.Linear):
#     def __init__(self, in_features, out_features, *args, **kwargs):
#         in_features = [in_features] if isinstance(in_features, int) else list(in_features)
#         out_features = [out_features] if isinstance(out_features, int) else list(out_features)
#         self.in_features_multidim = in_features
#         self.out_features_multidim = out_features
#         super().__init__(
#             np.array(in_features).prod(),
#             np.array(out_features).prod(),
#             *args, **kwargs)
#
#     def forward(self, x):
#         shape = x.shape
#         n = len(self.in_features_multidim)
#         x = x.view(*shape[0:-n], self.in_features)
#         y = super().forward(x)
#         y = y.view(*shape[0:-n], *self.out_features_multidim)
#         return y
