# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.

import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained("A/B", torch_dtype=torch.float16, variant="fp16",
        ...                                           custom_pipeline="matryoshka",)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        >>> image
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
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
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
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
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class NestedUNetConfig:  # UNetConfig
    inner_config = field(
        default={"nesting": True},  # UNetConfig(nesting=True),
        metadata={"help": "inner unet used as middle blocks"},
    )
    skip_mid_blocks: bool = field(default=True)
    skip_cond_emb: bool = field(default=True)
    skip_inner_unet_input: bool = field(
        default=False,
        metadata={"help": "If enabled, the inner unet only received the downsampled image, no features."},
    )
    skip_normalization: bool = field(
        default=False,
    )
    initialize_inner_with_pretrained: str = field(
        default=None,
        metadata={
            "help": (
                "Initialize the inner unet with pretrained vision model ",
                "Provide the vision_model_path",
            )
        },
    )
    freeze_inner_unet: bool = field(default=False)
    interp_conditioning: bool = field(
        default=False,
    )


@dataclass
class Nested2UNetConfig(NestedUNetConfig):
    inner_config: NestedUNetConfig = field(
        default=NestedUNetConfig(nesting=True, initialize_inner_with_pretrained=None)
    )


@dataclass
class Nested3UNetConfig(Nested2UNetConfig):
    inner_config: Nested2UNetConfig = field(
        default=Nested2UNetConfig(nesting=True, initialize_inner_with_pretrained=None)
    )


@dataclass
class Nested4UNetConfig(Nested3UNetConfig):
    inner_config: Nested3UNetConfig = field(
        default=Nested3UNetConfig(nesting=True, initialize_inner_with_pretrained=None)
    )


class NestedUNet(  # UNet,
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    """
    Pipeline for text-to-image generation using Stable Diffusion.

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
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
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
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(..., in_channels=3, out_channels=3)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
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

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
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
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        unet.config.inner_config.conditioning_feature_dim = unet.config.conditioning_feature_dim
        if getattr(unet.config.inner_config, "inner_config", None) is None:
            self.inner_unet = UNet2DConditionModel(
                in_channels=3,
                out_channels=3,
                block_out_channels=(256, 512, 768),
                ff_act_fn="gelu",
                cross_attention_dim=2048,
                resnet_time_scale_shift="scale_shift",
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            )
            self.inner_unet  # Arrange inner attributes
        else:
            self.inner_unet = NestedUNet(3, 3, unet.config.inner_config)

        if not unet.config.skip_inner_unet_input:
            self.in_adapter = nn.Conv2d(
                unet.config.resolution_channels[-1],
                unet.config.inner_config.resolution_channels[0],
                kernel_size=3,
                padding=1,
                bias=True,
            )
        else:
            self.in_adapter = None
        self.out_adapter = nn.Conv2d(
            unet.config.inner_config.resolution_channels[0],
            unet.config.resolution_channels[-1],
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.is_temporal = [unet.config.temporal_mode and (not unet.config.temporal_spatial_ds)]
        if hasattr(self.inner_unet, "is_temporal"):
            self.is_temporal += self.inner_unet.is_temporal

        nest_ratio = int(2 ** (len(unet.config.resolution_channels) - 1))
        if self.is_temporal[0]:
            nest_ratio = int(np.sqrt(nest_ratio))
        if self.inner_unet.config.nesting and self.inner_unet.model_type == "nested_unet":
            self.nest_ratio = [nest_ratio * self.inner_unet.nest_ratio[0]] + self.inner_unet.nest_ratio
        else:
            self.nest_ratio = [nest_ratio]

        if self.config.interp_conditioning:
            self.interp_layer1 = nn.Linear(self.temporal_dim // 4, self.temporal_dim)
            self.interp_layer2 = nn.Linear(self.temporal_dim, self.temporal_dim)

    @property
    def model_type(self):
        return "nested_unet"

    def forward_conditioning(self, *args, **kwargs):
        return self.inner_unet.forward_conditioning(*args, **kwargs)

    def forward_denoising(self, x_t, times, cond_emb=None, conditioning=None, cond_mask=None, micros={}):
        # 1. time embedding
        temb = self.create_temporal_embedding(times)
        if cond_emb is not None:
            temb = temb + cond_emb
        if self.conditions is not None:
            temb = temb + self.forward_micro_conditioning(times, micros)

        # 2. input layer (normalize the input)
        if self._config.nesting:
            x_t, x_feat = x_t
        bsz = [x.size(0) for x in x_t]
        bh, bl = bsz[0], bsz[1]
        x_t_low, x_t = x_t[1:], x_t[0]
        x = self.forward_input_layer(x_t, normalize=(not self.config.skip_normalization))
        if self._config.nesting:
            x = x + x_feat

        # 3. downsample blocks in the outer layers
        x, skip_activations = self.forward_downsample(
            x,
            temb[:bh],
            conditioning[:bh],
            cond_mask[:bh] if cond_mask is not None else cond_mask,
        )

        # 4. run inner unet
        x_inner = self.in_adapter(x) if self.in_adapter is not None else None
        x_inner = (
            torch.cat([x_inner, x_inner.new_zeros(bl - bh, *x_inner.size()[1:])], 0) if bh < bl else x_inner
        )  # pad zeros for low-resolutions
        x_low, x_inner = self.inner_unet.forward_denoising(
            (x_t_low, x_inner), times, cond_emb, conditioning, cond_mask, micros
        )
        x_inner = self.out_adapter(x_inner)
        x = x + x_inner[:bh] if bh < bl else x + x_inner

        # 5. upsample blocks in the outer layers
        x = self.forward_upsample(
            x,
            temb[:bh],
            conditioning[:bh],
            cond_mask[:bh] if cond_mask is not None else cond_mask,
            skip_activations,
        )

        # 6. output layer
        x_out = self.forward_output_layer(x)

        # 7. outpupt both low and high-res output
        if isinstance(x_low, list):
            out = [x_out] + x_low
        else:
            out = [x_out, x_low]
        if self._config.nesting:
            return out, x
        return out
