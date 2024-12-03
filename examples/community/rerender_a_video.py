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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from gmflow.gmflow import GMFlow
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


@torch.no_grad()
def get_warped_and_mask(flow_model, image1, image2, image3=None, pixel_consistency=False, device=None):
    if image3 is None:
        image3 = image1
    padder = InputPadder(image1.shape, padding_factor=8)
    image1, image2 = padder.pad(image1[None].to(device), image2[None].to(device))
    results_dict = flow_model(
        image1, image2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True
    )
    flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
    fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)  # [1, H, W] float
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(
            bwd_occ + (abs(image2 - warped_image1).mean(dim=1) > 255 * 0.25).float(), 0, 1
        ).unsqueeze(0)
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ, bwd_flow


blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text-to-video pipelines.

    Args:
        frames (`List[np.ndarray]` or `torch.Tensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `torch` tensor. The length of the list denotes the video length (the number of frames).
    """

    frames: Union[List[np.ndarray], torch.Tensor]


@torch.no_grad()
def find_flat_region(mask):
    device = mask.device
    kernel_x = torch.Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.Tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    mask_ = F.pad(mask.unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    grad_x = torch.nn.functional.conv2d(mask_, kernel_x)
    grad_y = torch.nn.functional.conv2d(mask_, kernel_y)
    return ((abs(grad_x) + abs(grad_y)) == 0).float()[0]


class AttnState:
    STORE = 0
    LOAD = 1
    LOAD_AND_STORE_PREV = 2

    def __init__(self):
        self.reset()

    @property
    def state(self):
        return self.__state

    @property
    def timestep(self):
        return self.__timestep

    def set_timestep(self, t):
        self.__timestep = t

    def reset(self):
        self.__state = AttnState.STORE
        self.__timestep = 0

    def to_load(self):
        self.__state = AttnState.LOAD

    def to_load_and_store_prev(self):
        self.__state = AttnState.LOAD_AND_STORE_PREV


class CrossFrameAttnProcessor(AttnProcessor):
    """
    Cross frame attention processor. Each frame attends the first frame and previous frame.

    Args:
        attn_state: Whether the model is processing the first frame or an intermediate frame
    """

    def __init__(self, attn_state: AttnState):
        super().__init__()
        self.attn_state = attn_state
        self.first_maps = {}
        self.prev_maps = {}

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # Is self attention
        if encoder_hidden_states is None:
            t = self.attn_state.timestep
            if self.attn_state.state == AttnState.STORE:
                self.first_maps[t] = hidden_states.detach()
                self.prev_maps[t] = hidden_states.detach()
                res = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                if self.attn_state.state == AttnState.LOAD_AND_STORE_PREV:
                    tmp = hidden_states.detach()
                cross_map = torch.cat((self.first_maps[t], self.prev_maps[t]), dim=1)
                res = super().__call__(attn, hidden_states, cross_map, attention_mask, temb)
                if self.attn_state.state == AttnState.LOAD_AND_STORE_PREV:
                    self.prev_maps[t] = tmp
        else:
            res = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        return res


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


class RerenderAVideoPipeline(StableDiffusionControlNetImg2ImgPipeline):
    r"""
    Pipeline for video-to-video translation using Stable Diffusion with Rerender Algorithm.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

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
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

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
        image_encoder=None,
        requires_safety_checker: bool = True,
        device=None,
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
        self.to(device)

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
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.attn_state = AttnState()
        attn_processor_dict = {}
        for k in unet.attn_processors.keys():
            if k.startswith("up"):
                attn_processor_dict[k] = CrossFrameAttnProcessor(self.attn_state)
            else:
                attn_processor_dict[k] = AttnProcessor()

        self.unet.set_attn_processor(attn_processor_dict)

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

    # Modified from src/diffusers/pipelines/controlnet/pipeline_controlnet.StableDiffusionControlNetImg2ImgPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
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

        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )

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

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.prepare_latents
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
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
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

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
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        frames: Union[List[np.ndarray], torch.Tensor] = None,
        control_frames: Union[List[np.ndarray], torch.Tensor] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        warp_start: Union[float, List[float]] = 0.0,
        warp_end: Union[float, List[float]] = 0.3,
        mask_start: Union[float, List[float]] = 0.5,
        mask_end: Union[float, List[float]] = 0.8,
        smooth_boundary: bool = True,
        mask_strength: Union[float, List[float]] = 0.5,
        inner_strength: Union[float, List[float]] = 0.9,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            frames (`List[np.ndarray]` or `torch.Tensor`): The input images to be used as the starting point for the image generation process.
            control_frames (`List[np.ndarray]` or `torch.Tensor`): The ControlNet input images condition to provide guidance to the `unet` for generation.
            strength ('float'): SDEdit strength.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting
                than for [`~StableDiffusionControlNetPipeline.__call__`].
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.
            warp_start (`float`): Shape-aware fusion start timestep.
            warp_end (`float`): Shape-aware fusion end timestep.
            mask_start (`float`): Pixel-aware fusion start timestep.
            mask_end (`float`):Pixel-aware fusion end timestep.
            smooth_boundary (`bool`): Smooth fusion boundary. Set `True` to prevent artifacts at boundary.
            mask_strength (`float`): Pixel-aware fusion strength.
            inner_strength (`float`): Pixel-aware fusion detail level.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
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
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        # Currently we only support 1 prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            assert False
        else:
            assert False
        num_images_per_prompt = 1

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

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
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Process the first frame
        height, width = None, None
        output_frames = []
        self.attn_state.reset()

        # 4.1 prepare frames
        image = self.image_processor.preprocess(frames[0]).to(dtype=torch.float32)
        first_image = image[0]  # C, H, W

        # 4.2 Prepare controlnet_conditioning_image
        # Currently we only support single control
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_frames[0],
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=1,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        else:
            assert False

        # 4.3 Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, cur_num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size)

        # 4.4 Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 4.5 Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 4.6 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        first_x0_list = []

        # 4.7 Denoising loop
        num_warmup_steps = len(timesteps) - cur_num_inference_steps * self.scheduler.order
        with self.progress_bar(total=cur_num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.attn_state.set_timestep(t.item())

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
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

                if guess_mode and do_classifier_free_guidance:
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
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_x0 = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                first_x0 = pred_x0.detach()
                first_x0_list.append(first_x0)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        first_result = image
        prev_result = image
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        output_frames.append(image[0])

        # 5. Process each frame
        for idx in range(1, len(frames)):
            image = frames[idx]
            prev_image = frames[idx - 1]
            control_image = control_frames[idx]
            # 5.1 prepare frames
            image = self.image_processor.preprocess(image).to(dtype=torch.float32)
            prev_image = self.image_processor.preprocess(prev_image).to(dtype=torch.float32)

            warped_0, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(
                self.flow_model, first_image, image[0], first_result, False, self.device
            )
            blend_mask_0 = blur(F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
            blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

            warped_pre, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(
                self.flow_model, prev_image[0], image[0], prev_result, False, self.device
            )
            blend_mask_pre = blur(F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
            blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

            warp_mask = 1 - F.max_pool2d(blend_mask_0, kernel_size=8)
            warp_flow = F.interpolate(bwd_flow_0 / 8.0, scale_factor=1.0 / 8, mode="bilinear")

            # 5.2 Prepare controlnet_conditioning_image
            # Currently we only support single control
            if isinstance(controlnet, ControlNetModel):
                control_image = self.prepare_control_image(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
            else:
                assert False

            # 5.3 Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps, cur_num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(batch_size)

            skip_t = int(num_inference_steps * (1 - strength))
            warp_start_t = int(warp_start * num_inference_steps)
            warp_end_t = int(warp_end * num_inference_steps)
            mask_start_t = int(mask_start * num_inference_steps)
            mask_end_t = int(mask_end * num_inference_steps)

            # 5.4 Prepare latent variables
            init_latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

            # 5.5 Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 5.6 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

            # 5.7 Denoising loop
            num_warmup_steps = len(timesteps) - cur_num_inference_steps * self.scheduler.order

            def denoising_loop(latents, mask=None, xtrg=None, noise_rescale=None):
                dir_xt = 0
                latents_dtype = latents.dtype
                with self.progress_bar(total=cur_num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        self.attn_state.set_timestep(t.item())
                        if i + skip_t >= mask_start_t and i + skip_t <= mask_end_t and xtrg is not None:
                            rescale = torch.maximum(1.0 - mask, (1 - mask**2) ** 0.5 * inner_strength)
                            if noise_rescale is not None:
                                rescale = (1.0 - mask) * (1 - noise_rescale) + rescale * noise_rescale
                            noise = randn_tensor(xtrg.shape, generator=generator, device=device, dtype=xtrg.dtype)
                            latents_ref = self.scheduler.add_noise(xtrg, noise, t)
                            latents = latents_ref * mask + (1.0 - mask) * (latents - dir_xt) + rescale * dir_xt
                            latents = latents.to(latents_dtype)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # controlnet(s) inference
                        if guess_mode and do_classifier_free_guidance:
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

                        if guess_mode and do_classifier_free_guidance:
                            # Inferred ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [
                                torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
                            ]
                            mid_block_res_sample = torch.cat(
                                [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
                            )

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # Get pred_x0 from scheduler
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        pred_x0 = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                        if i + skip_t >= warp_start_t and i + skip_t <= warp_end_t:
                            # warp x_0
                            pred_x0 = (
                                flow_warp(first_x0_list[i], warp_flow, mode="nearest") * warp_mask
                                + (1 - warp_mask) * pred_x0
                            )

                            # get x_t from x_0
                            latents = self.scheduler.add_noise(pred_x0, noise_pred, t).to(latents_dtype)

                        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                        if i == len(timesteps) - 1:
                            alpha_t_prev = 1.0
                        else:
                            alpha_t_prev = self.scheduler.alphas_cumprod[prev_t]

                        dir_xt = (1.0 - alpha_t_prev) ** 0.5 * noise_pred

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[
                            0
                        ]

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                        ):
                            progress_bar.update()
                            if callback is not None and i % callback_steps == 0:
                                callback(i, t, latents)

                    return latents

            if mask_start_t <= mask_end_t:
                self.attn_state.to_load()
            else:
                self.attn_state.to_load_and_store_prev()
            latents = denoising_loop(init_latents)

            if mask_start_t <= mask_end_t:
                direct_result = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                blend_results = (1 - blend_mask_pre) * warped_pre + blend_mask_pre * direct_result
                blend_results = (1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

                bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
                blend_mask = blur(F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
                blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

                blend_results = blend_results.to(latents.dtype)
                xtrg = self.vae.encode(blend_results).latent_dist.sample(generator)
                xtrg = self.vae.config.scaling_factor * xtrg
                blend_results_rec = self.vae.decode(xtrg / self.vae.config.scaling_factor, return_dict=False)[0]
                xtrg_rec = self.vae.encode(blend_results_rec).latent_dist.sample(generator)
                xtrg_rec = self.vae.config.scaling_factor * xtrg_rec
                xtrg_ = xtrg + (xtrg - xtrg_rec)
                blend_results_rec_new = self.vae.decode(xtrg_ / self.vae.config.scaling_factor, return_dict=False)[0]
                tmp = (abs(blend_results_rec_new - blend_results).mean(dim=1, keepdims=True) > 0.25).float()

                mask_x = F.max_pool2d(
                    (F.interpolate(tmp, scale_factor=1 / 8.0, mode="bilinear") > 0).float(),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

                mask = 1 - F.max_pool2d(1 - blend_mask, kernel_size=8)  # * (1-mask_x)

                if smooth_boundary:
                    noise_rescale = find_flat_region(mask)
                else:
                    noise_rescale = torch.ones_like(mask)

                xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask
                xtrg = xtrg.to(latents.dtype)

                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps, cur_num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

                self.attn_state.to_load_and_store_prev()
                latents = denoising_loop(init_latents, mask * mask_strength, xtrg, noise_rescale)

            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            else:
                image = latents

            prev_result = image

            do_denormalize = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            output_frames.append(image[0])

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return output_frames

        return TextToVideoSDPipelineOutput(frames=output_frames)


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
