# Copyright 2025 Jingyang Zhang and The HuggingFace Team. All rights reserved.
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

import abc
import inspect
import math
import numbers
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, FusedAttnProcessor2_0
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class GaussianSmoothing(nn.Module):
    """
    Copied from official repo: https://github.com/showlab/BoxDiff/blob/master/utils/gaussian_smoothing.py
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed separately for each channel
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
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


class AttendExciteCrossAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=1)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    # @property
    # def num_uncond_att_layers(self):
    #     return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.global_store[key]] for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        """
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        """
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
        self.num_uncond_att_layers = 0


def aggregate_attention(
    attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int
) -> torch.Tensor:
    """Aggregates the attention across the different layers and heads at the specified resolution."""
    out = []
    attention_maps = attention_store.get_average_attention()

    # for k, v in attention_maps.items():
    #     for vv in v:
    #         print(vv.shape)
    # exit()

    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out


def register_attention_control(model, controller):
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        # cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            # hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            # block_id = int(name[len("up_blocks.")])
            # hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            # block_id = int(name[len("down_blocks.")])
            # hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(attnstore=controller, place_in_unet=place_in_unet)
    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionBoxDiffPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with BoxDiff.

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
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if scheduler is not None and getattr(scheduler.config, "steps_offset", 1) != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if scheduler is not None and getattr(scheduler.config, "clip_sample", False) is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

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

        is_unet_version_less_0_9_0 = (
            unet is not None
            and hasattr(unet.config, "_diffusers_version")
            and version.parse(version.parse(unet.config._diffusers_version).base_version) < version.parse("0.9.0.dev0")
        )
        is_unet_sample_size_less_64 = (
            unet is not None and hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- stable-diffusion-v1-5/stable-diffusion-v1-5"
                " \n- stable-diffusion-v1-5/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
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
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        depr_message = f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`."
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`."
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        depr_message = f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`."
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        depr_message = f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`."
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            depr_message,
        )
        self.vae.disable_tiling()

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

        if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return text_inputs, prompt_embeds, negative_prompt_embeds

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

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
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
        height,
        width,
        boxdiff_phrases,
        boxdiff_boxes,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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

        if boxdiff_phrases is not None or boxdiff_boxes is not None:
            if not (boxdiff_phrases is not None and boxdiff_boxes is not None):
                raise ValueError("Either both `boxdiff_phrases` and `boxdiff_boxes` must be passed or none of them.")

            if not isinstance(boxdiff_phrases, list) or not isinstance(boxdiff_boxes, list):
                raise ValueError("`boxdiff_phrases` and `boxdiff_boxes` must be lists.")

            if len(boxdiff_phrases) != len(boxdiff_boxes):
                raise ValueError(
                    "`boxdiff_phrases` and `boxdiff_boxes` must have the same length,"
                    f" got: `boxdiff_phrases` {len(boxdiff_phrases)} != `boxdiff_boxes`"
                    f" {len(boxdiff_boxes)}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://huggingface.co/papers/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.fuse_qkv_projections
    def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING]
        > This API is 🧪 experimental.

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_unet = False
        self.fusing_vae = False

        if unet:
            self.fusing_unet = True
            self.unet.fuse_qkv_projections()
            self.unet.set_attn_processor(FusedAttnProcessor2_0())

        if vae:
            if not isinstance(self.vae, AutoencoderKL):
                raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")

            self.fusing_vae = True
            self.vae.fuse_qkv_projections()
            self.vae.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.unfuse_qkv_projections
    def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        > [!WARNING]
        > This API is 🧪 experimental.

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if unet:
            if not self.fusing_unet:
                logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
            else:
                self.unet.unfuse_qkv_projections()
                self.fusing_unet = False

        if vae:
            if not self.fusing_vae:
                logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices_to_alter: List[int],
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
        bboxes: List[int] = None,
        L: int = 1,
        P: float = 0.2,
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token "1:last_idx"
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        cnt = 0
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]

            # TODO
            # box = [max(round(b / (512 / image.shape[0])), 0) for b in bboxes[cnt]]
            # x1, y1, x2, y2 = box
            H, W = image.shape
            x1 = min(max(round(bboxes[cnt][0] * W), 0), W)
            y1 = min(max(round(bboxes[cnt][1] * H), 0), H)
            x2 = min(max(round(bboxes[cnt][2] * W), 0), W)
            y2 = min(max(round(bboxes[cnt][3] * H), 0), H)
            box = [x1, y1, x2, y2]
            cnt += 1

            # coordinates to masks
            obj_mask = torch.zeros_like(image)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(image.device)
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            k = (obj_mask.sum() * P).long()
            max_indices_list_fg.append((image * obj_mask).reshape(-1).topk(k)[0].mean())

            # Outer-Box constraint
            k = (bg_mask.sum() * P).long()
            max_indices_list_bg.append((image * bg_mask).reshape(-1).topk(k)[0].mean())

            # Corner Constraint
            gt_proj_x = torch.max(obj_mask, dim=0)[0]
            gt_proj_y = torch.max(obj_mask, dim=1)[0]
            corner_mask_x = torch.zeros_like(gt_proj_x)
            corner_mask_y = torch.zeros_like(gt_proj_y)

            # create gt according to the number config.L
            N = gt_proj_x.shape[0]
            corner_mask_x[max(box[0] - L, 0) : min(box[0] + L + 1, N)] = 1.0
            corner_mask_x[max(box[2] - L, 0) : min(box[2] + L + 1, N)] = 1.0
            corner_mask_y[max(box[1] - L, 0) : min(box[1] + L + 1, N)] = 1.0
            corner_mask_y[max(box[3] - L, 0) : min(box[3] + L + 1, N)] = 1.0
            dist_x.append((F.l1_loss(image.max(dim=0)[0], gt_proj_x, reduction="none") * corner_mask_x).mean())
            dist_y.append((F.l1_loss(image.max(dim=1)[0], gt_proj_y, reduction="none") * corner_mask_y).mean())

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

    def _aggregate_and_get_max_attention_per_token(
        self,
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        attention_res: int = 16,
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
        bboxes: List[int] = None,
        L: int = 1,
        P: float = 0.2,
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )
        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bboxes=bboxes,
            L=L,
            P=P,
        )
        return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y

    @staticmethod
    def _compute_loss(
        max_attention_per_index_fg: List[torch.Tensor],
        max_attention_per_index_bg: List[torch.Tensor],
        dist_x: List[torch.Tensor],
        dist_y: List[torch.Tensor],
        return_losses: bool = False,
    ) -> torch.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""
        losses_fg = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[int],
        loss_fg: torch.Tensor,
        threshold: float,
        text_embeddings: torch.Tensor,
        text_input,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 16,
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        max_refinement_steps: int = 20,
        normalize_eot: bool = False,
        bboxes: List[int] = None,
        L: int = 1,
        P: float = 0.2,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1.0 - threshold)

        while loss_fg > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            # noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            (
                max_attention_per_index_fg,
                max_attention_per_index_bg,
                dist_x,
                dist_y,
            ) = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot,
                bboxes=bboxes,
                L=L,
                P=P,
            )

            loss_fg, losses_fg = self._compute_loss(
                max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True
            )

            if loss_fg != 0:
                latents = self._update_latent(latents, loss_fg, step_size)

            # with torch.no_grad():
            #     noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
            #     noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            # try:
            #     low_token = np.argmax([l.item() if not isinstance(l, int) else l for l in losses_fg])
            # except Exception as e:
            #     print(e)  # catch edge case :)
            #     low_token = np.argmax(losses_fg)

            # low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            # print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index_fg[low_token]}')

            if iteration >= max_refinement_steps:
                # print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                #       f'Finished with a max attention of {max_attention_per_index_fg[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        # noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        (
            max_attention_per_index_fg,
            max_attention_per_index_bg,
            dist_x,
            dist_y,
        ) = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bboxes=bboxes,
            L=L,
            P=P,
        )
        loss_fg, losses_fg = self._compute_loss(
            max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, return_losses=True
        )
        # print(f"\t Finished with loss of: {loss_fg}")
        return loss_fg, latents, max_attention_per_index_fg

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        boxdiff_phrases: List[str] = None,
        boxdiff_boxes: List[List[float]] = None,  # TODO
        boxdiff_kwargs: Optional[Dict[str, Any]] = {
            "attention_res": 16,
            "P": 0.2,
            "L": 1,
            "max_iter_to_alter": 25,
            "loss_thresholds": {0: 0.05, 10: 0.5, 20: 0.8},
            "scale_factor": 20,
            "scale_range": (1.0, 0.5),
            "smooth_attentions": True,
            "sigma": 0.5,
            "kernel_size": 3,
            "refine": False,
            "normalize_eot": True,
        },
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.

            boxdiff_attention_res (`int`, *optional*, defaults to 16):
                The resolution of the attention maps used for computing the BoxDiff loss.
            boxdiff_P (`float`, *optional*, defaults to 0.2):

            boxdiff_L (`int`, *optional*, defaults to 1):
                The number of pixels around the corner to be selected in BoxDiff loss.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
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

        # -1. Register attention control (for BoxDiff)
        attention_store = AttentionStore()
        register_attention_control(self, attention_store)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            boxdiff_phrases,
            boxdiff_boxes,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self.prompt = prompt

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
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

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        text_inputs, prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
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

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 6.3 Prepare BoxDiff inputs
        # a) Indices to alter
        input_ids = self.tokenizer(prompt)["input_ids"]
        decoded = [self.tokenizer.decode([t]) for t in input_ids]
        indices_to_alter = []
        bboxes = []
        for phrase, box in zip(boxdiff_phrases, boxdiff_boxes):
            # it could happen that phrase does not correspond a single token?
            if phrase not in decoded:
                continue
            indices_to_alter.append(decoded.index(phrase))
            bboxes.append(box)

        # b) A bunch of hyperparameters
        attention_res = boxdiff_kwargs.get("attention_res", 16)
        smooth_attentions = boxdiff_kwargs.get("smooth_attentions", True)
        sigma = boxdiff_kwargs.get("sigma", 0.5)
        kernel_size = boxdiff_kwargs.get("kernel_size", 3)
        L = boxdiff_kwargs.get("L", 1)
        P = boxdiff_kwargs.get("P", 0.2)
        thresholds = boxdiff_kwargs.get("loss_thresholds", {0: 0.05, 10: 0.5, 20: 0.8})
        max_iter_to_alter = boxdiff_kwargs.get("max_iter_to_alter", len(self.scheduler.timesteps) + 1)
        scale_factor = boxdiff_kwargs.get("scale_factor", 20)
        refine = boxdiff_kwargs.get("refine", False)
        normalize_eot = boxdiff_kwargs.get("normalize_eot", True)

        scale_range = boxdiff_kwargs.get("scale_range", (1.0, 0.5))
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # BoxDiff optimization
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    self.unet.zero_grad()

                    # Get max activation value for each subject token
                    (
                        max_attention_per_index_fg,
                        max_attention_per_index_bg,
                        dist_x,
                        dist_y,
                    ) = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=normalize_eot,
                        bboxes=bboxes,
                        L=L,
                        P=P,
                    )

                    loss_fg, loss = self._compute_loss(
                        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y
                    )

                    # Refinement from attend-and-excite (not necessary)
                    if refine and i in thresholds.keys() and loss_fg > 1.0 - thresholds[i]:
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        loss_fg, latents, max_attention_per_index_fg = self._perform_iterative_refinement_step(
                            latents=latents,
                            indices_to_alter=indices_to_alter,
                            loss_fg=loss_fg,
                            threshold=thresholds[i],
                            text_embeddings=prompt_embeds,
                            text_input=text_inputs,
                            attention_store=attention_store,
                            step_size=scale_factor * np.sqrt(scale_range[i]),
                            t=t,
                            attention_res=attention_res,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                            normalize_eot=normalize_eot,
                            bboxes=bboxes,
                            L=L,
                            P=P,
                        )

                    # Perform gradient update
                    if i < max_iter_to_alter:
                        _, loss = self._compute_loss(
                            max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y
                        )
                        if loss != 0:
                            latents = self._update_latent(
                                latents=latents, loss=loss, step_size=scale_factor * np.sqrt(scale_range[i])
                            )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

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

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

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
