# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import gc
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.utils.model_zoo
from einops import rearrange, repeat
from gmflow.gmflow import GMFlow
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img, sample_coords, mode="bilinear", padding_mode="zeros", return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


class Dilate:
    def __init__(self, kernel_size=7, channels=1, device="cpu"):
        self.kernel_size = kernel_size
        self.channels = channels
        gaussian_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        self.mean = (self.kernel_size - 1) // 2
        gaussian_kernel = gaussian_kernel.to(device)
        self.gaussian_filter = gaussian_kernel

    def __call__(self, x):
        x = F.pad(x, (self.mean, self.mean, self.mean, self.mean), "replicate")
        return torch.clamp(F.conv2d(x, self.gaussian_filter, bias=None), 0, 1)


def flow_warp(feature, flow, mask=False, mode="bilinear", padding_mode="zeros"):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]
    grid = grid.to(feature.dtype)
    return bilinear_sample(feature, grid, mode=mode, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow
    # (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def numpy2tensor(img):
    x0 = torch.from_numpy(img.copy()).float().cuda() / 255.0 * 2.0 - 1.0
    x0 = torch.stack([x0], dim=0)
    # einops.rearrange(x0, 'b h w c -> b c h w').clone()
    return x0.permute(0, 3, 1, 2)


def calc_mean_std(feat, eps=1e-5, chunk=1):
    size = feat.size()
    assert len(size) == 4
    if chunk == 2:
        feat = torch.cat(feat.chunk(2), dim=3)
    N, C = size[:2]
    feat_var = feat.view(N // chunk, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N // chunk, C, -1).mean(dim=2).view(N // chunk, C, 1, 1)
    return feat_mean.repeat(chunk, 1, 1, 1), feat_std.repeat(chunk, 1, 1, 1)


def adaptive_instance_normalization(content_feat, style_feat, chunk=1):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat, chunk)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def optimize_feature(
    sample, flows, occs, correlation_matrix=[], intra_weight=1e2, iters=20, unet_chunk_size=2, optimize_temporal=True
):
    """
    FRESO-guided latent feature optimization
    * optimize spatial correspondence (match correlation_matrix)
    * optimize temporal correspondence (match warped_image)
    """
    if (flows is None or occs is None or (not optimize_temporal)) and (
        intra_weight == 0 or len(correlation_matrix) == 0
    ):
        return sample
    # flows=[fwd_flows, bwd_flows]: (N-1)*2*H1*W1
    # occs=[fwd_occs, bwd_occs]: (N-1)*H1*W1
    # sample: 2N*C*H*W
    torch.cuda.empty_cache()
    video_length = sample.shape[0] // unet_chunk_size
    latent = rearrange(sample.to(torch.float32), "(b f) c h w -> b f c h w", f=video_length)

    cs = torch.nn.Parameter((latent.detach().clone()))
    optimizer = torch.optim.Adam([cs], lr=0.2)

    # unify resolution
    if flows is not None and occs is not None:
        scale = sample.shape[2] * 1.0 / flows[0].shape[2]
        kernel = int(1 / scale)
        bwd_flow_ = F.interpolate(flows[1] * scale, scale_factor=scale, mode="bilinear").repeat(
            unet_chunk_size, 1, 1, 1
        )
        bwd_occ_ = F.max_pool2d(occs[1].unsqueeze(1), kernel_size=kernel).repeat(
            unet_chunk_size, 1, 1, 1
        )  # 2(N-1)*1*H1*W1
        fwd_flow_ = F.interpolate(flows[0] * scale, scale_factor=scale, mode="bilinear").repeat(
            unet_chunk_size, 1, 1, 1
        )
        fwd_occ_ = F.max_pool2d(occs[0].unsqueeze(1), kernel_size=kernel).repeat(
            unet_chunk_size, 1, 1, 1
        )  # 2(N-1)*1*H1*W1
        # match frame 0,1,2,3 and frame 1,2,3,0
        reshuffle_list = list(range(1, video_length)) + [0]

    # attention_probs is the GRAM matrix of the normalized feature
    attention_probs = None
    for tmp in correlation_matrix:
        if sample.shape[2] * sample.shape[3] == tmp.shape[1]:
            attention_probs = tmp  # 2N*HW*HW
            break

    n_iter = [0]
    while n_iter[0] < iters:

        def closure():
            optimizer.zero_grad()

            loss = 0

            # temporal consistency loss
            if optimize_temporal and flows is not None and occs is not None:
                c1 = rearrange(cs[:, :], "b f c h w -> (b f) c h w")
                c2 = rearrange(cs[:, reshuffle_list], "b f c h w -> (b f) c h w")
                warped_image1 = flow_warp(c1, bwd_flow_)
                warped_image2 = flow_warp(c2, fwd_flow_)
                loss = (
                    abs((c2 - warped_image1) * (1 - bwd_occ_)) + abs((c1 - warped_image2) * (1 - fwd_occ_))
                ).mean() * 2

            # spatial consistency loss
            if attention_probs is not None and intra_weight > 0:
                cs_vector = rearrange(cs, "b f c h w -> (b f) (h w) c")
                # attention_scores = torch.bmm(cs_vector, cs_vector.transpose(-1, -2))
                # cs_attention_probs = attention_scores.softmax(dim=-1)
                cs_vector = cs_vector / ((cs_vector**2).sum(dim=2, keepdims=True) ** 0.5)
                cs_attention_probs = torch.bmm(cs_vector, cs_vector.transpose(-1, -2))
                tmp = F.l1_loss(cs_attention_probs, attention_probs) * intra_weight
                loss = tmp + loss

            loss.backward()
            n_iter[0] += 1

            return loss

        optimizer.step(closure)

    torch.cuda.empty_cache()
    return adaptive_instance_normalization(rearrange(cs.data.to(sample.dtype), "b f c h w -> (b f) c h w"), sample)


@torch.no_grad()
def warp_tensor(sample, flows, occs, saliency, unet_chunk_size):
    """
    Warp images or features based on optical flow
    Fuse the warped imges or features based on occusion masks and saliency map
    """
    scale = sample.shape[2] * 1.0 / flows[0].shape[2]
    kernel = int(1 / scale)
    bwd_flow_ = F.interpolate(flows[1] * scale, scale_factor=scale, mode="bilinear")
    bwd_occ_ = F.max_pool2d(occs[1].unsqueeze(1), kernel_size=kernel)  # (N-1)*1*H1*W1
    if scale == 1:
        bwd_occ_ = Dilate(kernel_size=13, device=sample.device)(bwd_occ_)
    fwd_flow_ = F.interpolate(flows[0] * scale, scale_factor=scale, mode="bilinear")
    fwd_occ_ = F.max_pool2d(occs[0].unsqueeze(1), kernel_size=kernel)  # (N-1)*1*H1*W1
    if scale == 1:
        fwd_occ_ = Dilate(kernel_size=13, device=sample.device)(fwd_occ_)
    scale2 = sample.shape[2] * 1.0 / saliency.shape[2]
    saliency = F.interpolate(saliency, scale_factor=scale2, mode="bilinear")
    latent = sample.to(torch.float32)
    video_length = sample.shape[0] // unet_chunk_size
    warp_saliency = flow_warp(saliency, bwd_flow_)
    warp_saliency_ = flow_warp(saliency[0:1], fwd_flow_[video_length - 1 : video_length])

    for j in range(unet_chunk_size):
        for ii in range(video_length - 1):
            i = video_length * j + ii
            warped_image = flow_warp(latent[i : i + 1], bwd_flow_[ii : ii + 1])
            mask = (1 - bwd_occ_[ii : ii + 1]) * saliency[ii + 1 : ii + 2] * warp_saliency[ii : ii + 1]
            latent[i + 1 : i + 2] = latent[i + 1 : i + 2] * (1 - mask) + warped_image * mask
        i = video_length * j
        ii = video_length - 1
        warped_image = flow_warp(latent[i : i + 1], fwd_flow_[ii : ii + 1])
        mask = (1 - fwd_occ_[ii : ii + 1]) * saliency[ii : ii + 1] * warp_saliency_
        latent[ii + i : ii + i + 1] = latent[ii + i : ii + i + 1] * (1 - mask) + warped_image * mask

    return latent.to(sample.dtype)


def my_forward(
    self,
    steps=[],
    layers=[0, 1, 2, 3],
    flows=None,
    occs=None,
    correlation_matrix=[],
    intra_weight=1e2,
    iters=20,
    optimize_temporal=True,
    saliency=None,
):
    """
    Hacked pipe.unet.forward()
    copied from https://github.com/huggingface/diffusers/blob/v0.19.3/src/diffusers/models/unet_2d_condition.py#L700
    if you are using a new version of diffusers, please copy the source code and modify it accordingly (find [HACK] in the code)
    * restore and return the decoder features
    * optimize the decoder features
    * perform background smoothing
    """

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)
            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        """
        [HACK] restore the decoder features in up_samples
        """
        up_samples = ()
        # down_samples = ()
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            """
            [HACK] restore the decoder features in up_samples
            [HACK] optimize the decoder features
            [HACK] perform background smoothing
            """
            if i in layers:
                up_samples += (sample,)
            if timestep in steps and i in layers:
                sample = optimize_feature(
                    sample, flows, occs, correlation_matrix, intra_weight, iters, optimize_temporal=optimize_temporal
                )
                if saliency is not None:
                    sample = warp_tensor(sample, flows, occs, saliency, 2)

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        """
        [HACK] return the output feature as well as the decoder features
        """
        if not return_dict:
            return (sample,) + up_samples

        return UNet2DConditionOutput(sample=sample)

    return forward


@torch.no_grad()
def get_single_mapping_ind(bwd_flow, bwd_occ, imgs, scale=1.0):
    """
    FLATTEN: Optical fLow-guided attention (Temoporal-guided attention)
    Find the correspondence between every pixels in a pair of frames

    [input]
    bwd_flow: 1*2*H*W
    bwd_occ: 1*H*W      i.e., f2 = warp(f1, bwd_flow) * bwd_occ
    imgs: 2*3*H*W       i.e., [f1,f2]

    [output]
    mapping_ind: pixel index correspondence
    unlinkedmask: indicate whether a pixel has no correspondence
    i.e., f2 = f1[mapping_ind] * unlinkedmask
    """
    flows = F.interpolate(bwd_flow, scale_factor=1.0 / scale, mode="bilinear")[0][[1, 0]] / scale  # 2*H*W
    _, H, W = flows.shape
    masks = torch.logical_not(F.interpolate(bwd_occ[None], scale_factor=1.0 / scale, mode="bilinear") > 0.5)[
        0
    ]  # 1*H*W
    frames = F.interpolate(imgs, scale_factor=1.0 / scale, mode="bilinear").view(2, 3, -1)  # 2*3*HW
    grid = torch.stack(torch.meshgrid([torch.arange(H), torch.arange(W)]), dim=0).to(flows.device)  # 2*H*W
    warp_grid = torch.round(grid + flows)
    mask = torch.logical_and(
        torch.logical_and(
            torch.logical_and(torch.logical_and(warp_grid[0] >= 0, warp_grid[0] < H), warp_grid[1] >= 0),
            warp_grid[1] < W,
        ),
        masks[0],
    ).view(-1)  # HW
    warp_grid = warp_grid.view(2, -1)  # 2*HW
    warp_ind = (warp_grid[0] * W + warp_grid[1]).to(torch.long)  # HW
    mapping_ind = torch.zeros_like(warp_ind) - 1  # HW

    for f0ind, f1ind in enumerate(warp_ind):
        if mask[f0ind]:
            if mapping_ind[f1ind] == -1:
                mapping_ind[f1ind] = f0ind
            else:
                targetv = frames[0, :, f1ind]
                pref0ind = mapping_ind[f1ind]
                prev = frames[1, :, pref0ind]
                v = frames[1, :, f0ind]
                if ((prev - targetv) ** 2).mean() > ((v - targetv) ** 2).mean():
                    mask[pref0ind] = False
                    mapping_ind[f1ind] = f0ind
                else:
                    mask[f0ind] = False

    unusedind = torch.arange(len(mask)).to(mask.device)[~mask]
    unlinkedmask = mapping_ind == -1
    mapping_ind[unlinkedmask] = unusedind
    return mapping_ind, unlinkedmask


@torch.no_grad()
def get_mapping_ind(bwd_flows, bwd_occs, imgs, scale=1.0):
    """
    FLATTEN: Optical fLow-guided attention (Temoporal-guided attention)
    Find pixel correspondence between every consecutive frames in a batch

    [input]
    bwd_flow: (N-1)*2*H*W
    bwd_occ: (N-1)*H*W
    imgs: N*3*H*W

    [output]
    fwd_mappings: N*1*HW
    bwd_mappings: N*1*HW
    flattn_mask: HW*1*N*N
    i.e., imgs[i,:,fwd_mappings[i]] corresponds to imgs[0]
    i.e., imgs[i,:,fwd_mappings[i]][:,bwd_mappings[i]] restore the original imgs[i]
    """
    N, H, W = imgs.shape[0], int(imgs.shape[2] // scale), int(imgs.shape[3] // scale)
    iterattn_mask = torch.ones(H * W, N, N, dtype=torch.bool).to(imgs.device)
    for i in range(len(imgs) - 1):
        one_mask = torch.ones(N, N, dtype=torch.bool).to(imgs.device)
        one_mask[: i + 1, i + 1 :] = False
        one_mask[i + 1 :, : i + 1] = False
        mapping_ind, unlinkedmask = get_single_mapping_ind(
            bwd_flows[i : i + 1], bwd_occs[i : i + 1], imgs[i : i + 2], scale
        )
        if i == 0:
            fwd_mapping = [torch.arange(len(mapping_ind)).to(mapping_ind.device)]
            bwd_mapping = [torch.arange(len(mapping_ind)).to(mapping_ind.device)]
        iterattn_mask[unlinkedmask[fwd_mapping[-1]]] = torch.logical_and(
            iterattn_mask[unlinkedmask[fwd_mapping[-1]]], one_mask
        )
        fwd_mapping += [mapping_ind[fwd_mapping[-1]]]
        bwd_mapping += [torch.sort(fwd_mapping[-1])[1]]
    fwd_mappings = torch.stack(fwd_mapping, dim=0).unsqueeze(1)
    bwd_mappings = torch.stack(bwd_mapping, dim=0).unsqueeze(1)
    return fwd_mappings, bwd_mappings, iterattn_mask.unsqueeze(1)


def apply_FRESCO_opt(
    pipe,
    steps=[],
    layers=[0, 1, 2, 3],
    flows=None,
    occs=None,
    correlation_matrix=[],
    intra_weight=1e2,
    iters=20,
    optimize_temporal=True,
    saliency=None,
):
    """
    Apply FRESCO-based optimization to a StableDiffusionPipeline
    """
    pipe.unet.forward = my_forward(
        pipe.unet, steps, layers, flows, occs, correlation_matrix, intra_weight, iters, optimize_temporal, saliency
    )


@torch.no_grad()
def get_intraframe_paras(pipe, imgs, frescoProc, prompt_embeds, do_classifier_free_guidance=True, generator=None):
    """
    Get parameters for spatial-guided attention and optimization
    * perform one step denoising
    * collect attention feature, stored in frescoProc.controller.stored_attn['decoder_attn']
    * compute the gram matrix of the normalized feature for spatial consistency loss
    """

    noise_scheduler = pipe.scheduler
    timestep = noise_scheduler.timesteps[-1]
    device = pipe._execution_device
    B, C, H, W = imgs.shape

    frescoProc.controller.disable_controller()
    apply_FRESCO_opt(pipe)
    frescoProc.controller.clear_store()
    frescoProc.controller.enable_store()

    latents = pipe.prepare_latents(
        imgs.to(pipe.unet.dtype), timestep, B, 1, prompt_embeds.dtype, device, generator=generator, repeat_noise=False
    )

    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    model_output = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=None,
        return_dict=False,
    )

    frescoProc.controller.disable_store()

    # gram matrix of the normalized feature for spatial consistency loss
    correlation_matrix = []
    for tmp in model_output[1:]:
        latent_vector = rearrange(tmp, "b c h w -> b (h w) c")
        latent_vector = latent_vector / ((latent_vector**2).sum(dim=2, keepdims=True) ** 0.5)
        attention_probs = torch.bmm(latent_vector, latent_vector.transpose(-1, -2))
        correlation_matrix += [attention_probs.detach().clone().to(torch.float32)]
        del attention_probs, latent_vector, tmp
    del model_output

    clear_cache()

    return correlation_matrix


@torch.no_grad()
def get_flow_and_interframe_paras(flow_model, imgs):
    """
    Get parameters for temporal-guided attention and optimization
    * predict optical flow and occlusion mask
    * compute pixel index correspondence for FLATTEN
    """
    images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs], dim=0).cuda()
    imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)

    reshuffle_list = list(range(1, len(images))) + [0]

    results_dict = flow_model(
        images,
        images[reshuffle_list],
        attn_splits_list=[2],
        corr_radius_list=[-1],
        prop_radius_list=[-1],
        pred_bidir_flow=True,
    )
    flow_pr = results_dict["flow_preds"][-1]  # [2*B, 2, H, W]
    fwd_flows, bwd_flows = flow_pr.chunk(2)  # [B, 2, H, W]
    fwd_occs, bwd_occs = forward_backward_consistency_check(fwd_flows, bwd_flows)  # [B, H, W]

    warped_image1 = flow_warp(images, bwd_flows)
    bwd_occs = torch.clamp(
        bwd_occs + (abs(images[reshuffle_list] - warped_image1).mean(dim=1) > 255 * 0.25).float(), 0, 1
    )

    warped_image2 = flow_warp(images[reshuffle_list], fwd_flows)
    fwd_occs = torch.clamp(fwd_occs + (abs(images - warped_image2).mean(dim=1) > 255 * 0.25).float(), 0, 1)

    attn_mask = []
    for scale in [8.0, 16.0, 32.0]:
        bwd_occs_ = F.interpolate(bwd_occs[:-1].unsqueeze(1), scale_factor=1.0 / scale, mode="bilinear")
        attn_mask += [
            torch.cat((bwd_occs_[0:1].reshape(1, -1) > -1, bwd_occs_.reshape(bwd_occs_.shape[0], -1) > 0.5), dim=0)
        ]

    fwd_mappings = []
    bwd_mappings = []
    interattn_masks = []
    for scale in [8.0, 16.0]:
        fwd_mapping, bwd_mapping, interattn_mask = get_mapping_ind(bwd_flows, bwd_occs, imgs_torch, scale=scale)
        fwd_mappings += [fwd_mapping]
        bwd_mappings += [bwd_mapping]
        interattn_masks += [interattn_mask]

    interattn_paras = {}
    interattn_paras["fwd_mappings"] = fwd_mappings
    interattn_paras["bwd_mappings"] = bwd_mappings
    interattn_paras["interattn_masks"] = interattn_masks

    clear_cache()

    return [fwd_flows, bwd_flows], [fwd_occs, bwd_occs], attn_mask, interattn_paras


class AttentionControl:
    """
    Control FRESCO-based attention
    * enable/diable spatial-guided attention
    * enable/diable temporal-guided attention
    * enable/diable cross-frame attention
    * collect intermediate attention feature (for spatial-guided attention)
    """

    def __init__(self):
        self.stored_attn = self.get_empty_store()
        self.store = False
        self.index = 0
        self.attn_mask = None
        self.interattn_paras = None
        self.use_interattn = False
        self.use_cfattn = False
        self.use_intraattn = False
        self.intraattn_bias = 0
        self.intraattn_scale_factor = 0.2
        self.interattn_scale_factor = 0.2

    @staticmethod
    def get_empty_store():
        return {
            "decoder_attn": [],
        }

    def clear_store(self):
        del self.stored_attn
        torch.cuda.empty_cache()
        gc.collect()
        self.stored_attn = self.get_empty_store()
        self.disable_intraattn()

    # store attention feature of the input frame for spatial-guided attention
    def enable_store(self):
        self.store = True

    def disable_store(self):
        self.store = False

    # spatial-guided attention
    def enable_intraattn(self):
        self.index = 0
        self.use_intraattn = True
        self.disable_store()
        if len(self.stored_attn["decoder_attn"]) == 0:
            self.use_intraattn = False

    def disable_intraattn(self):
        self.index = 0
        self.use_intraattn = False
        self.disable_store()

    def disable_cfattn(self):
        self.use_cfattn = False

    # cross frame attention
    def enable_cfattn(self, attn_mask=None):
        if attn_mask:
            if self.attn_mask:
                del self.attn_mask
                torch.cuda.empty_cache()
            self.attn_mask = attn_mask
            self.use_cfattn = True
        else:
            if self.attn_mask:
                self.use_cfattn = True
            else:
                print("Warning: no valid cross-frame attention parameters available!")
                self.disable_cfattn()

    def disable_interattn(self):
        self.use_interattn = False

    # temporal-guided attention
    def enable_interattn(self, interattn_paras=None):
        if interattn_paras:
            if self.interattn_paras:
                del self.interattn_paras
                torch.cuda.empty_cache()
            self.interattn_paras = interattn_paras
            self.use_interattn = True
        else:
            if self.interattn_paras:
                self.use_interattn = True
            else:
                print("Warning: no valid temporal-guided attention parameters available!")
                self.disable_interattn()

    def disable_controller(self):
        self.disable_intraattn()
        self.disable_interattn()
        self.disable_cfattn()

    def enable_controller(self, interattn_paras=None, attn_mask=None):
        self.enable_intraattn()
        self.enable_interattn(interattn_paras)
        self.enable_cfattn(attn_mask)

    def forward(self, context):
        if self.store:
            self.stored_attn["decoder_attn"].append(context.detach())
        if self.use_intraattn and len(self.stored_attn["decoder_attn"]) > 0:
            tmp = self.stored_attn["decoder_attn"][self.index]
            self.index = self.index + 1
            if self.index >= len(self.stored_attn["decoder_attn"]):
                self.index = 0
                self.disable_store()
            return tmp
        return context

    def __call__(self, context):
        context = self.forward(context)
        return context


class FRESCOAttnProcessor2_0:
    """
    Hack self attention to FRESCO-based attention
    * adding spatial-guided attention
    * adding temporal-guided attention
    * adding cross-frame attention

    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    Usage
    frescoProc = FRESCOAttnProcessor2_0(2, attn_mask)
    attnProc = AttnProcessor2_0()

    attn_processor_dict = {}
    for k in pipe.unet.attn_processors.keys():
        if k.startswith("up_blocks.2") or k.startswith("up_blocks.3"):
            attn_processor_dict[k] = frescoProc
        else:
            attn_processor_dict[k] = attnProc
    pipe.unet.set_attn_processor(attn_processor_dict)
    """

    def __init__(self, unet_chunk_size=2, controller=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.unet_chunk_size = unet_chunk_size
        self.controller = controller

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            if self.controller and self.controller.store:
                self.controller(hidden_states.detach().clone())
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # BC * HW * 8D
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_raw, key_raw = None, None
        if self.controller and self.controller.use_interattn and (not crossattn):
            query_raw, key_raw = query.clone(), key.clone()

        inner_dim = key.shape[-1]  # 8D
        head_dim = inner_dim // attn.heads  # D

        """for efficient cross-frame attention"""
        if self.controller and self.controller.use_cfattn and (not crossattn):
            video_length = key.size()[0] // self.unet_chunk_size
            former_frame_index = [0] * video_length
            attn_mask = None
            if self.controller.attn_mask is not None:
                for m in self.controller.attn_mask:
                    if m.shape[1] == key.shape[1]:
                        attn_mask = m
            # BC * HW * 8D --> B * C * HW * 8D
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            # B * C * HW * 8D --> B * C * HW * 8D
            if attn_mask is None:
                key = key[:, former_frame_index]
            else:
                key = repeat(key[:, attn_mask], "b d c -> b f d c", f=video_length)
            # B * C * HW * 8D --> BC * HW * 8D
            key = rearrange(key, "b f d c -> (b f) d c").detach()
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            if attn_mask is None:
                value = value[:, former_frame_index]
            else:
                value = repeat(value[:, attn_mask], "b d c -> b f d c", f=video_length)
            value = rearrange(value, "b f d c -> (b f) d c").detach()

        # BC * HW * 8D --> BC * HW * 8 * D --> BC * 8 * HW * D
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # BC * 8 * HW2 * D
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # BC * 8 * HW2 * D2
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        """for spatial-guided intra-frame attention"""
        if self.controller and self.controller.use_intraattn and (not crossattn):
            ref_hidden_states = self.controller(None)
            assert ref_hidden_states.shape == encoder_hidden_states.shape
            query_ = attn.to_q(ref_hidden_states)
            key_ = attn.to_k(ref_hidden_states)

            # BC * 8 * HW * D
            query_ = query_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            query = F.scaled_dot_product_attention(
                query_,
                key_ * self.controller.intraattn_scale_factor,
                query,
                attn_mask=torch.eye(query_.size(-2), key_.size(-2), dtype=query.dtype, device=query.device)
                * self.controller.intraattn_bias,
            ).detach()

            del query_, key_
            torch.cuda.empty_cache()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # output: BC * 8 * HW * D2
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        """for temporal-guided inter-frame attention (FLATTEN)"""
        if self.controller and self.controller.use_interattn and (not crossattn):
            del query, key, value
            torch.cuda.empty_cache()
            bwd_mapping = None
            fwd_mapping = None
            for i, f in enumerate(self.controller.interattn_paras["fwd_mappings"]):
                if f.shape[2] == hidden_states.shape[2]:
                    fwd_mapping = f
                    bwd_mapping = self.controller.interattn_paras["bwd_mappings"][i]
                    interattn_mask = self.controller.interattn_paras["interattn_masks"][i]
            video_length = key_raw.size()[0] // self.unet_chunk_size
            # BC * HW * 8D --> C * 8BD * HW
            key = rearrange(key_raw, "(b f) d c -> f (b c) d", f=video_length)
            query = rearrange(query_raw, "(b f) d c -> f (b c) d", f=video_length)
            # BC * 8 * HW * D --> C * 8BD * HW
            # key = rearrange(hidden_states, "(b f) h d c -> f (b h c) d", f=video_length) ########
            # query = rearrange(hidden_states, "(b f) h d c -> f (b h c) d", f=video_length) #######

            value = rearrange(hidden_states, "(b f) h d c -> f (b h c) d", f=video_length)
            key = torch.gather(key, 2, fwd_mapping.expand(-1, key.shape[1], -1))
            query = torch.gather(query, 2, fwd_mapping.expand(-1, query.shape[1], -1))
            value = torch.gather(value, 2, fwd_mapping.expand(-1, value.shape[1], -1))
            # C * 8BD * HW --> BHW, C, 8D
            key = rearrange(key, "f (b c) d -> (b d) f c", b=self.unet_chunk_size)
            query = rearrange(query, "f (b c) d -> (b d) f c", b=self.unet_chunk_size)
            value = rearrange(value, "f (b c) d -> (b d) f c", b=self.unet_chunk_size)
            # BHW * C * 8D --> BHW * C * 8 * D--> BHW * 8 * C * D
            query = query.view(-1, video_length, attn.heads, head_dim).transpose(1, 2).detach()
            key = key.view(-1, video_length, attn.heads, head_dim).transpose(1, 2).detach()
            value = value.view(-1, video_length, attn.heads, head_dim).transpose(1, 2).detach()
            hidden_states_ = F.scaled_dot_product_attention(
                query,
                key * self.controller.interattn_scale_factor,
                value,
                # .to(query.dtype)-1.0) * 1e6 -
                attn_mask=(interattn_mask.repeat(self.unet_chunk_size, 1, 1, 1)),
                # torch.eye(interattn_mask.shape[2]).to(query.device).to(query.dtype) * 1e4,
            )

            # BHW * 8 * C * D --> C * 8BD * HW
            hidden_states_ = rearrange(hidden_states_, "(b d) h f c -> f (b h c) d", b=self.unet_chunk_size)
            hidden_states_ = torch.gather(
                hidden_states_, 2, bwd_mapping.expand(-1, hidden_states_.shape[1], -1)
            ).detach()
            # C * 8BD * HW --> BC * 8 * HW * D
            hidden_states = rearrange(
                hidden_states_, "f (b h c) d -> (b f) h d c", b=self.unet_chunk_size, h=attn.heads
            )

        # BC * 8 * HW * D --> BC * HW * 8D
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

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


def apply_FRESCO_attn(pipe):
    """
    Apply FRESCO-guided attention to a StableDiffusionPipeline
    """
    frescoProc = FRESCOAttnProcessor2_0(2, AttentionControl())
    attnProc = AttnProcessor2_0()
    attn_processor_dict = {}
    for k in pipe.unet.attn_processors.keys():
        if k.startswith("up_blocks.2") or k.startswith("up_blocks.3"):
            attn_processor_dict[k] = frescoProc
        else:
            attn_processor_dict[k] = attnProc
    pipe.unet.set_attn_processor(attn_processor_dict)
    return frescoProc


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


class FrescoV2VPipeline(StableDiffusionControlNetImg2ImgPipeline):
    r"""
    Pipeline for video-to-video translation using Stable Diffusion with FRESCO Algorithm.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

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

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        frescoProc = FRESCOAttnProcessor2_0(2, AttentionControl())
        attnProc = AttnProcessor2_0()
        attn_processor_dict = {}
        for k in self.unet.attn_processors.keys():
            if k.startswith("up_blocks.2") or k.startswith("up_blocks.3"):
                attn_processor_dict[k] = frescoProc
            else:
                attn_processor_dict[k] = attnProc
        self.unet.set_attn_processor(attn_processor_dict)
        self.frescoProc = frescoProc

        flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to(self.device)

        checkpoint = torch.utils.model_zoo.load_url(
            "https://huggingface.co/Anonymous-sub/Rerender/resolve/main/models/gmflow_sintel-0c07dcb3.pth",
            map_location=lambda storage, loc: storage,
        )
        weights = checkpoint["model"] if "model" in checkpoint else checkpoint
        flow_model.load_state_dict(weights, strict=False)
        flow_model.eval()
        self.flow_model = flow_model

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
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
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
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
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

        if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

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
        image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        callback_on_step_end_tensor_inputs=None,
    ):
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

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image
    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
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
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.prepare_latents
    def prepare_latents(
        self, image, timestep, batch_size, num_images_per_prompt, dtype, device, repeat_noise, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        if repeat_noise:
            noise = randn_tensor((1, *shape[1:]), generator=generator, device=device, dtype=dtype)
            one_tuple = (1,) * (len(shape) - 1)
            noise = noise.repeat(batch_size, *one_tuple)
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        frames: Union[List[np.ndarray], torch.FloatTensor] = None,
        control_frames: Union[List[np.ndarray], torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        end_opt_step=15,
        num_intraattn_steps=1,
        step_interattn_end=350,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            frames (`List[np.ndarray]` or `torch.FloatTensor`): The input images to be used as the starting point for the image generation process.
            control_frames (`List[np.ndarray]` or `torch.FloatTensor`): The ControlNet input images condition to provide guidance to the `unet` for generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
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
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            end_opt_step:
                The feature optimization is activated from strength * num_inference_step to end_opt_step.
            num_intraattn_steps:
                Apply num_interattn_steps steps of spatial-guided attention.
            step_interattn_end:
                Apply temporal-guided attention in [step_interattn_end, 1000] steps

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_frames[0],
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        batch_size = len(frames)

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
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
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare image
        imgs_np = []
        for frame in frames:
            if isinstance(frame, PIL.Image.Image):
                imgs_np.append(np.asarray(frame))
            else:
                # np.ndarray
                imgs_np.append(frame)
        images_pt = self.image_processor.preprocess(frames).to(dtype=torch.float32)

        # 5. Prepare controlnet_conditioning_image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_frames,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_frames:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        self.flow_model.to(device)

        flows, occs, attn_mask, interattn_paras = get_flow_and_interframe_paras(self.flow_model, imgs_np)
        correlation_matrix = get_intraframe_paras(self, images_pt, self.frescoProc, prompt_embeds, generator)

        """
        Flexible settings for attention:
        * Turn off FRESCO-guided attention: frescoProc.controller.disable_controller()
        Then you can turn on one specific attention submodule
        * Turn on Cross-frame attention: frescoProc.controller.enable_cfattn(attn_mask)
        * Turn on Spatial-guided attention: frescoProc.controller.enable_intraattn()
        * Turn on Temporal-guided attention: frescoProc.controller.enable_interattn(interattn_paras)

        Flexible settings for optimization:
        * Turn off Spatial-guided optimization: set optimize_temporal = False in apply_FRESCO_opt()
        * Turn off Temporal-guided optimization: set correlation_matrix = [] in apply_FRESCO_opt()
        * Turn off FRESCO-guided optimization: disable_FRESCO_opt(pipe)

        Flexible settings for background smoothing:
        * Turn off background smoothing: set saliency = None in apply_FRESCO_opt()
        """

        self.frescoProc.controller.enable_controller(interattn_paras=interattn_paras, attn_mask=attn_mask)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        apply_FRESCO_opt(
            self,
            steps=timesteps[:end_opt_step],
            flows=flows,
            occs=occs,
            correlation_matrix=correlation_matrix,
            saliency=None,
            optimize_temporal=True,
        )

        clear_cache()

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            images_pt,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator=generator,
            repeat_noise=True,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i >= num_intraattn_steps:
                    self.frescoProc.controller.disable_intraattn()
                if t < step_interattn_end:
                    self.frescoProc.controller.disable_interattn()

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
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
