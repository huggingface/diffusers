import inspect
import math
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention_processor import Attention, AttnProcessor
from ...models.lora import adjust_lora_scale_text_encoder
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ...schedulers import DDIMScheduler, DPMSolverMultistepScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import LEditsPPPipelineStableDiffusion
        >>> from diffusers.utils import load_image

        >>> pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/cherry_blossom.png"
        >>> image = load_image(img_url).convert("RGB")

        >>> _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.1)

        >>> edited_image = pipe(
        ...     editing_prompt=["cherry blossom"], edit_guidance_scale=10.0, edit_threshold=0.75
        ... ).images[0]
        ```
"""


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.AttentionStore
class LeditsAttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP=False):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        if attn.shape[1] <= self.max_size:
            bs = 1 + int(PnP) + editing_prompts
            skip = 2 if PnP else 1  # skip PnP & unconditional
            attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
            source_batch_size = int(attn.shape[1] // bs)
            self.forward(attn[:, skip * source_batch_size :], is_cross, place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        self.step_store[key].append(attn)

    def between_steps(self, store_step=True):
        if store_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            self.cur_step += 1
        self.step_store = self.get_empty_store()

    def get_attention(self, step: int):
        if self.average:
            attention = {
                key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
            }
        else:
            assert step is not None
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(
        self, attention_maps, prompts, res: Union[int, Tuple[int]], from_where: List[str], is_cross: bool, select: int
    ):
        out = [[] for x in range(self.batch_size)]
        if isinstance(res, int):
            num_pixels = res**2
            resolution = (res, res)
        else:
            num_pixels = res[0] * res[1]
            resolution = res[:2]

        for location in from_where:
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                for batch, item in enumerate(bs_item):
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(prompts), -1, *resolution, item.shape[-1])[select]
                        out[batch].append(cross_maps)

        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # average over heads
        out = out.sum(1) / out.shape[1]
        return out

    def __init__(self, average: bool, batch_size=1, max_resolution=16, max_size: int = None):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average
        self.batch_size = batch_size
        if max_size is None:
            self.max_size = max_resolution**2
        elif max_size is not None and max_resolution is None:
            self.max_size = max_size
        else:
            raise ValueError("Only allowed to set one of max_resolution or max_size")


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing
class LeditsGaussianSmoothing:
    def __init__(self, device):
        kernel_size = [3, 3]
        sigma = [0.5, 0.5]

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        self.weight = kernel.to(device)

    def __call__(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv2d(input, weight=self.weight.to(input.dtype))


class LEDITSCrossAttnProcessor:
    def __init__(self, attention_store, place_in_unet, pnp, editing_prompts):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.editing_prompts = editing_prompts
        self.pnp = pnp

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        temb=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

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

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(
            attention_probs,
            is_cross=True,
            place_in_unet=self.place_in_unet,
            editing_prompts=self.editing_prompts,
            PnP=self.pnp,
        )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
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


class LEditsPPPipelineStableDiffusion(
    DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionLoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    """
    Pipeline for textual image editing using LEDits++ with Stable Diffusion.

    This model inherits from [`DiffusionPipeline`] and builds on the [`StableDiffusionPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]. If any other scheduler is passed it will
            automatically be set to [`DPMSolverMultistepScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if not isinstance(scheduler, DDIMScheduler) and not isinstance(scheduler, DPMSolverMultistepScheduler):
            scheduler = DPMSolverMultistepScheduler.from_config(
                scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2
            )
            logger.warning(
                "This pipeline only supports DDIMScheduler and DPMSolverMultistepScheduler. "
                "The scheduler has been changed to DPMSolverMultistepScheduler."
            )

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
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.inversion_steps = None

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
    def prepare_extra_step_kwargs(self, eta, generator=None):
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

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        negative_prompt=None,
        editing_prompt_embeddings=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if editing_prompt_embeddings is not None and negative_prompt_embeds is not None:
            if editing_prompt_embeddings.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`editing_prompt_embeddings` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `editing_prompt_embeddings` {editing_prompt_embeddings.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        # shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        # if latents.shape != shape:
        #    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_unet(self, attention_store, PnP: bool = False):
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            if "attn2" in name and place_in_unet != "mid":
                attn_procs[name] = LEDITSCrossAttnProcessor(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    pnp=PnP,
                    editing_prompts=self.enabled_editing_prompts,
                )
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)

    def encode_prompt(
        self,
        device,
        num_images_per_prompt,
        enable_edit_guidance,
        negative_prompt=None,
        editing_prompt=None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        editing_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            enable_edit_guidance (`bool`):
                whether to perform any editing or reconstruct the input image instead
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            editing_prompt (`str` or `List[str]`, *optional*):
                Editing prompt(s) to be encoded. If not defined, one has to pass `editing_prompt_embeds` instead.
            editing_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
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

        batch_size = self.batch_size
        num_edit_tokens = None

        if negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but exoected"
                    f"{batch_size} based on the input images. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
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

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = negative_prompt_embeds.dtype

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if enable_edit_guidance:
            if editing_prompt_embeds is None:
                # textual inversion: procecss multi-vector tokens if necessary
                # if isinstance(self, TextualInversionLoaderMixin):
                #    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
                if isinstance(editing_prompt, str):
                    editing_prompt = [editing_prompt]

                max_length = negative_prompt_embeds.shape[1]
                text_inputs = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_length=True,
                )

                num_edit_tokens = text_inputs.length - 2  # not counting startoftext and endoftext
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="longest",
                    return_tensors="pt",
                ).input_ids

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

                if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                if clip_skip is None:
                    editing_prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                    editing_prompt_embeds = editing_prompt_embeds[0]
                else:
                    editing_prompt_embeds = self.text_encoder(
                        text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                    )
                    # Access the `hidden_states` first, that contains a tuple of
                    # all the hidden states from the encoder layers. Then index into
                    # the tuple to access the hidden states from the desired layer.
                    editing_prompt_embeds = editing_prompt_embeds[-1][-(clip_skip + 1)]
                    # We also need to apply the final LayerNorm here to not mess with the
                    # representations. The `last_hidden_states` that we typically use for
                    # obtaining the final prompt representations passes through the LayerNorm
                    # layer.
                    editing_prompt_embeds = self.text_encoder.text_model.final_layer_norm(editing_prompt_embeds)

            editing_prompt_embeds = editing_prompt_embeds.to(dtype=negative_prompt_embeds.dtype, device=device)

            bs_embed_edit, seq_len, _ = editing_prompt_embeds.shape
            editing_prompt_embeds = editing_prompt_embeds.to(dtype=negative_prompt_embeds.dtype, device=device)
            editing_prompt_embeds = editing_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            editing_prompt_embeds = editing_prompt_embeds.view(bs_embed_edit * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return editing_prompt_embeds, negative_prompt_embeds, num_edit_tokens

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 0,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        user_mask: Optional[torch.Tensor] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
        use_cross_attn_mask: bool = False,
        use_intersect_mask: bool = True,
        attn_store_steps: Optional[List[int]] = [],
        store_averaged_over_steps: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for editing. The
        [`~pipelines.ledits_pp.LEditsPPPipelineStableDiffusion.invert`] method has to be called beforehand. Edits will
        always be performed for the last inverted image(s).

        Args:
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] instead of a plain
                tuple.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. The image is reconstructed by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeds (`torch.Tensor>`, *optional*):
                Pre-computed embeddings to use for guiding the image generation. Guidance direction of embedding should
                be specified via `reverse_editing_direction`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for guiding the image generation. If provided as list values should correspond to
                `editing_prompt`. `edit_guidance_scale` is defined as `s_e` of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which guidance will not be applied.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Masking threshold of guidance. Threshold should be proportional to the image region that is modified.
                'edit_threshold' is defined as 'λ' of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            user_mask (`torch.Tensor`, *optional*):
                User-provided mask for even better control over the editing process. This is helpful when LEDITS++'s
                implicit masks do not meet user preferences.
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.
            use_cross_attn_mask (`bool`, defaults to `False`):
                Whether cross-attention masks are used. Cross-attention masks are always used when use_intersect_mask
                is set to true. Cross-attention masks are defined as 'M^1' of equation 12 of [LEDITS++
                paper](https://arxiv.org/pdf/2311.16711.pdf).
            use_intersect_mask (`bool`, defaults to `True`):
                Whether the masking term is calculated as intersection of cross-attention masks and masks derived from
                the noise estimate. Cross-attention mask are defined as 'M^1' and masks derived from the noise estimate
                are defined as 'M^2' of equation 12 of [LEDITS++ paper](https://arxiv.org/pdf/2311.16711.pdf).
            attn_store_steps (`List[int]`, *optional*):
                Steps for which the attention maps are stored in the AttentionStore. Just for visualization purposes.
            store_averaged_over_steps (`bool`, defaults to `True`):
                Whether the attention maps for the 'attn_store_steps' are stored averaged over the diffusion steps. If
                False, attention maps for each step are stores separately. Just for visualization purposes.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
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
            [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            content, according to the `safety_checker`.
        """

        if self.inversion_steps is None:
            raise ValueError(
                "You need to invert an input image first before calling the pipeline. The `invert` method has to be called beforehand. Edits will always be performed for the last inverted image(s)."
            )

        eta = self.eta
        num_images_per_prompt = 1
        latents = self.init_latents

        zs = self.zs
        self.scheduler.set_timesteps(len(self.scheduler.timesteps))

        if use_intersect_mask:
            use_cross_attn_mask = True

        if use_cross_attn_mask:
            self.smoothing = LeditsGaussianSmoothing(self.device)

        if user_mask is not None:
            user_mask = user_mask.to(self.device)

        org_prompt = ""

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            negative_prompt,
            editing_prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        batch_size = self.batch_size

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeds is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = editing_prompt_embeds.shape[0]
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        edit_concepts, uncond_embeddings, num_edit_tokens = self.encode_prompt(
            editing_prompt=editing_prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            enable_edit_guidance=enable_edit_guidance,
            negative_prompt=negative_prompt,
            editing_prompt_embeds=editing_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if enable_edit_guidance:
            text_embeddings = torch.cat([uncond_embeddings, edit_concepts])
            self.text_cross_attention_maps = [editing_prompt] if isinstance(editing_prompt, str) else editing_prompt
        else:
            text_embeddings = torch.cat([uncond_embeddings])

        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.inversion_steps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}

        if use_cross_attn_mask:
            self.attention_store = LeditsAttentionStore(
                average=store_averaged_over_steps,
                batch_size=batch_size,
                max_size=(latents.shape[-2] / 4.0) * (latents.shape[-1] / 4.0),
                max_resolution=None,
            )
            self.prepare_unet(self.attention_store, PnP=False)
            resolution = latents.shape[-2:]
            att_res = (int(resolution[0] / 4), int(resolution[1] / 4))

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            None,
            None,
            text_embeddings.dtype,
            self.device,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        self.sem_guidance = None
        self.activation_mask = None

        # 7. Denoising loop
        num_warmup_steps = 0
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance

                if enable_edit_guidance:
                    latent_model_input = torch.cat([latents] * (1 + self.enabled_editing_prompts))
                else:
                    latent_model_input = latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                text_embed_input = text_embeddings

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample

                noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)  # [b,4, 64, 64]
                noise_pred_uncond = noise_pred_out[0]
                noise_pred_edit_concepts = noise_pred_out[1:]

                noise_guidance_edit = torch.zeros(
                    noise_pred_uncond.shape,
                    device=self.device,
                    dtype=noise_pred_uncond.dtype,
                )

                if sem_guidance is not None and len(sem_guidance) > i:
                    noise_guidance_edit += sem_guidance[i].to(self.device)

                elif enable_edit_guidance:
                    if self.activation_mask is None:
                        self.activation_mask = torch.zeros(
                            (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                        )

                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_uncond.shape))

                    for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                        if isinstance(edit_warmup_steps, list):
                            edit_warmup_steps_c = edit_warmup_steps[c]
                        else:
                            edit_warmup_steps_c = edit_warmup_steps
                        if i < edit_warmup_steps_c:
                            continue

                        if isinstance(edit_guidance_scale, list):
                            edit_guidance_scale_c = edit_guidance_scale[c]
                        else:
                            edit_guidance_scale_c = edit_guidance_scale

                        if isinstance(edit_threshold, list):
                            edit_threshold_c = edit_threshold[c]
                        else:
                            edit_threshold_c = edit_threshold
                        if isinstance(reverse_editing_direction, list):
                            reverse_editing_direction_c = reverse_editing_direction[c]
                        else:
                            reverse_editing_direction_c = reverse_editing_direction

                        if isinstance(edit_cooldown_steps, list):
                            edit_cooldown_steps_c = edit_cooldown_steps[c]
                        elif edit_cooldown_steps is None:
                            edit_cooldown_steps_c = i + 1
                        else:
                            edit_cooldown_steps_c = edit_cooldown_steps

                        if i >= edit_cooldown_steps_c:
                            continue

                        noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond

                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1

                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                        if user_mask is not None:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * user_mask

                        if use_cross_attn_mask:
                            out = self.attention_store.aggregate_attention(
                                attention_maps=self.attention_store.step_store,
                                prompts=self.text_cross_attention_maps,
                                res=att_res,
                                from_where=["up", "down"],
                                is_cross=True,
                                select=self.text_cross_attention_maps.index(editing_prompt[c]),
                            )
                            attn_map = out[:, :, :, 1 : 1 + num_edit_tokens[c]]  # 0 -> startoftext

                            # average over all tokens
                            if attn_map.shape[3] != num_edit_tokens[c]:
                                raise ValueError(
                                    f"Incorrect shape of attention_map. Expected size {num_edit_tokens[c]}, but found {attn_map.shape[3]}!"
                                )

                            attn_map = torch.sum(attn_map, dim=3)

                            # gaussian_smoothing
                            attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
                            attn_map = self.smoothing(attn_map).squeeze(1)

                            # torch.quantile function expects float32
                            if attn_map.dtype == torch.float32:
                                tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
                            else:
                                tmp = torch.quantile(
                                    attn_map.flatten(start_dim=1).to(torch.float32), edit_threshold_c, dim=1
                                ).to(attn_map.dtype)
                            attn_mask = torch.where(
                                attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1, *att_res), 1.0, 0.0
                            )

                            # resolution must match latent space dimension
                            attn_mask = F.interpolate(
                                attn_mask.unsqueeze(1),
                                noise_guidance_edit_tmp.shape[-2:],  # 64,64
                            ).repeat(1, 4, 1, 1)
                            self.activation_mask[i, c] = attn_mask.detach().cpu()
                            if not use_intersect_mask:
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                        if use_intersect_mask:
                            if t <= 800:
                                noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                                noise_guidance_edit_tmp_quantile = torch.sum(
                                    noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
                                )
                                noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(
                                    1, self.unet.config.in_channels, 1, 1
                                )

                                # torch.quantile function expects float32
                                if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                    tmp = torch.quantile(
                                        noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                        edit_threshold_c,
                                        dim=2,
                                        keepdim=False,
                                    )
                                else:
                                    tmp = torch.quantile(
                                        noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                        edit_threshold_c,
                                        dim=2,
                                        keepdim=False,
                                    ).to(noise_guidance_edit_tmp_quantile.dtype)

                                intersect_mask = (
                                    torch.where(
                                        noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                        torch.ones_like(noise_guidance_edit_tmp),
                                        torch.zeros_like(noise_guidance_edit_tmp),
                                    )
                                    * attn_mask
                                )

                                self.activation_mask[i, c] = intersect_mask.detach().cpu()

                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask

                            else:
                                # print(f"only attention mask for step {i}")
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                        elif not use_cross_attn_mask:
                            # calculate quantile
                            noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                            noise_guidance_edit_tmp_quantile = torch.sum(
                                noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
                            )
                            noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp_quantile.dtype)

                            self.activation_mask[i, c] = (
                                torch.where(
                                    noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                    torch.ones_like(noise_guidance_edit_tmp),
                                    torch.zeros_like(noise_guidance_edit_tmp),
                                )
                                .detach()
                                .cpu()
                            )

                            noise_guidance_edit_tmp = torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                noise_guidance_edit_tmp,
                                torch.zeros_like(noise_guidance_edit_tmp),
                            )

                        noise_guidance_edit += noise_guidance_edit_tmp

                    self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                noise_pred = noise_pred_uncond + noise_guidance_edit

                if enable_edit_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_edit_concepts.mean(dim=0, keepdim=False),
                        guidance_rescale=self.guidance_rescale,
                    )

                idx = t_to_idx[int(t)]
                latents = self.scheduler.step(
                    noise_pred, t, latents, variance_noise=zs[idx], **extra_step_kwargs
                ).prev_sample

                # step callback
                if use_cross_attn_mask:
                    store_step = i in attn_store_steps
                    self.attention_store.between_steps(store_step)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    # prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, text_embeddings.dtype)
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

        return LEditsPPDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad()
    def invert(
        self,
        image: PipelineImageInput,
        source_prompt: str = "",
        source_guidance_scale: float = 3.5,
        num_inversion_steps: int = 30,
        skip: float = 0.15,
        generator: Optional[torch.Generator] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resize_mode: Optional[str] = "default",
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ):
        r"""
        The function to the pipeline for image inversion as described by the [LEDITS++
        Paper](https://arxiv.org/abs/2301.12247). If the scheduler is set to [`~schedulers.DDIMScheduler`] the
        inversion proposed by [edit-friendly DPDM](https://arxiv.org/abs/2304.06140) will be performed instead.

        Args:
            image (`PipelineImageInput`):
                Input for the image(s) that are to be edited. Multiple input images have to default to the same aspect
                ratio.
            source_prompt (`str`, defaults to `""`):
                Prompt describing the input image that will be used for guidance during inversion. Guidance is disabled
                if the `source_prompt` is `""`.
            source_guidance_scale (`float`, defaults to `3.5`):
                Strength of guidance during inversion.
            num_inversion_steps (`int`, defaults to `30`):
                Number of total performed inversion steps after discarding the initial `skip` steps.
            skip (`float`, defaults to `0.15`):
                Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values
                will lead to stronger changes to the input image. `skip` has to be between `0` and `1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make inversion
                deterministic.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
                height.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
                the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
                resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
                center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
                image to fit within the specified width and height, maintaining the aspect ratio, and then center the
                image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.
            crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
                The crop coordinates for each image in the batch. If `None`, will not crop the image.

        Returns:
            [`~pipelines.ledits_pp.LEditsPPInversionPipelineOutput`]: Output will contain the resized input image(s)
            and respective VAE reconstruction(s).
        """
        # Reset attn processor, we do not want to store attn maps during inversion
        self.unet.set_attn_processor(AttnProcessor())

        self.eta = 1.0

        self.scheduler.config.timestep_spacing = "leading"
        self.scheduler.set_timesteps(int(num_inversion_steps * (1 + skip)))
        self.inversion_steps = self.scheduler.timesteps[-num_inversion_steps:]
        timesteps = self.inversion_steps

        # 1. encode image
        x0, resized = self.encode_image(
            image,
            dtype=self.text_encoder.dtype,
            height=height,
            width=width,
            resize_mode=resize_mode,
            crops_coords=crops_coords,
        )
        self.batch_size = x0.shape[0]

        # autoencoder reconstruction
        image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image_rec = self.image_processor.postprocess(image_rec, output_type="pil")

        # 2. get embeddings
        do_classifier_free_guidance = source_guidance_scale > 1.0

        lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None

        uncond_embedding, text_embeddings, _ = self.encode_prompt(
            num_images_per_prompt=1,
            device=self.device,
            negative_prompt=None,
            enable_edit_guidance=do_classifier_free_guidance,
            editing_prompt=source_prompt,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

        # 3. find zs and xts
        variance_noise_shape = (num_inversion_steps, *x0.shape)

        # intermediate latents
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=uncond_embedding.dtype)

        for t in reversed(timesteps):
            idx = num_inversion_steps - t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, torch.Tensor([t]))
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        self.scheduler.set_timesteps(len(self.scheduler.timesteps))
        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=uncond_embedding.dtype)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for t in timesteps:
                idx = num_inversion_steps - t_to_idx[int(t)] - 1
                # 1. predict noise residual
                xt = xts[idx + 1]

                noise_pred = self.unet(xt, timestep=t, encoder_hidden_states=uncond_embedding).sample

                if not source_prompt == "":
                    noise_pred_cond = self.unet(xt, timestep=t, encoder_hidden_states=text_embeddings).sample
                    noise_pred = noise_pred + source_guidance_scale * (noise_pred_cond - noise_pred)

                xtm1 = xts[idx]
                z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, self.eta)
                zs[idx] = z

                # correction to avoid error accumulation
                xts[idx] = xtm1_corrected

                progress_bar.update()

        self.init_latents = xts[-1].expand(self.batch_size, -1, -1, -1)
        zs = zs.flip(0)
        self.zs = zs

        return LEditsPPInversionPipelineOutput(images=resized, vae_reconstruction_images=image_rec)

    @torch.no_grad()
    def encode_image(self, image, dtype=None, height=None, width=None, resize_mode="default", crops_coords=None):
        image = self.image_processor.preprocess(
            image=image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        resized = self.image_processor.postprocess(image=image, output_type="pil")

        if max(image.shape[-2:]) > self.vae.config["sample_size"] * 1.5:
            logger.warning(
                "Your input images far exceed the default resolution of the underlying diffusion model. "
                "The output images may contain severe artifacts! "
                "Consider down-sampling the input using the `height` and `width` parameters"
            )
        image = image.to(dtype)

        x0 = self.vae.encode(image.to(self.device)).latent_dist.mode()
        x0 = x0.to(dtype)
        x0 = self.vae.config.scaling_factor * x0
        return x0, resized


def compute_noise_ddim(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. Clip "predicted x_0"
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

    # modifed so that updated xtm1 is returned as well (to avoid error accumulation)
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    if variance > 0.0:
        noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)
    else:
        noise = torch.tensor([0.0]).to(latents.device)

    return noise, mu_xt + (eta * variance**0.5) * noise


def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    def first_order_update(model_output, sample):  # timestep, prev_timestep, sample):
        sigma_t, sigma_s = scheduler.sigmas[scheduler.step_index + 1], scheduler.sigmas[scheduler.step_index]
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s

        mu_xt = (sigma_t / sigma_s * torch.exp(-h)) * sample + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output

        mu_xt = scheduler.dpm_solver_first_order_update(
            model_output=model_output, sample=sample, noise=torch.zeros_like(sample)
        )

        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
        if sigma > 0.0:
            noise = (prev_latents - mu_xt) / sigma
        else:
            noise = torch.tensor([0.0]).to(sample.device)

        prev_sample = mu_xt + sigma * noise
        return noise, prev_sample

    def second_order_update(model_output_list, sample):  # timestep_list, prev_timestep, sample):
        sigma_t, sigma_s0, sigma_s1 = (
            scheduler.sigmas[scheduler.step_index + 1],
            scheduler.sigmas[scheduler.step_index],
            scheduler.sigmas[scheduler.step_index - 1],
        )

        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = scheduler._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        mu_xt = (
            (sigma_t / sigma_s0 * torch.exp(-h)) * sample
            + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
            + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
        )

        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
        if sigma > 0.0:
            noise = (prev_latents - mu_xt) / sigma
        else:
            noise = torch.tensor([0.0]).to(sample.device)

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    model_output = scheduler.convert_model_output(model_output=noise_pred, sample=latents)
    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    if scheduler.lower_order_nums < 1:
        noise, prev_sample = first_order_update(model_output, latents)
    else:
        noise, prev_sample = second_order_update(scheduler.model_outputs, latents)

    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1

    # upon completion increase step index by one
    scheduler._step_index += 1

    return noise, prev_sample


def compute_noise(scheduler, *args):
    if isinstance(scheduler, DDIMScheduler):
        return compute_noise_ddim(scheduler, *args)
    elif (
        isinstance(scheduler, DPMSolverMultistepScheduler)
        and scheduler.config.algorithm_type == "sde-dpmsolver++"
        and scheduler.config.solver_order == 2
    ):
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        raise NotImplementedError
