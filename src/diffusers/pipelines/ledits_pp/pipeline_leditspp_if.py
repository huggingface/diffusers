import html
import inspect
import math
import re
import urllib.parse as ul
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ...image_processor import PipelineImageInput
from ...loaders import LoraLoaderMixin
from ...models import UNet2DConditionModel
from ...models.attention_processor import Attention, AttnAddedKVProcessor
from ...schedulers import DDIMScheduler, DPMSolverMultistepScheduler
from ...utils import (
    BACKENDS_MAPPING,
    is_accelerate_available,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..deepfloyd_if.safety_checker import IFSafetyChecker
from ..deepfloyd_if.watermark import IFWatermarker
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import LEditsPPPipelineIF, IFSuperResolutionPipeline

        >>> pipe = LEditsPPPipelineIF.from_pretrained(
        ...     "DeepFloyd/IF-I-XL-v1.0"
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")

        >>> img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg"
        >>> image = download_image(img_url)

        >>> _ = pipe.invert(
        ...     image = image,
        ...     num_inversion_steps=50,
        ...     skip=0.3
        ... )

        >>> edited_image = pipe(
        ...     editing_prompt=["tennis ball","tomato"],
        ...     reverse_editing_direction=[True,False],
        ...     edit_guidance_scale=[5.0,10.0],
        ...     edit_threshold=[0.9,0.85],
        ).images[0]

        >>> super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
        ...     "DeepFloyd/IF-II-L-v1.0"
        ... )
        >>> super_res_1_pipe.enable_model_cpu_offload()

        >>> image = super_res_1_pipe(
        ...     image=edited_image, prompt="", noise_level=0
        ... ).images[0]
        ```
"""


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LeditsAttentionStore
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


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LeditsGaussianSmoothing
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


class CrossAttnProcessor:
    def __init__(self, attention_store, place_in_unet, PnP, editing_prompts):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.editing_prompts = editing_prompts
        self.PnP = PnP

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross:
            self.attnstore(
                attention_probs,
                is_cross=True,
                place_in_unet=self.place_in_unet,
                editing_prompts=self.editing_prompts,
                PnP=self.PnP,
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.load_images
def load_images(
    images: PipelineImageInput, sizes=(512, 768), left=0, right=0, top=0, bottom=0, device=None, dtype=None
):
    def pre_process(im, sizes, left=0, right=0, top=0, bottom=0):
        if isinstance(im, str):
            image = np.array(Image.open(im).convert("RGB"))[:, :, :3]
        elif isinstance(im, Image.Image):
            image = np.array((im).convert("RGB"))[:, :, :3]
        else:
            image = im
        org_size = image.shape

        h, w, c = image.shape
        left = min(left, w - 1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top : h - bottom, left : w - right]

        ar = max(*image.shape[:2]) / min(*image.shape[:2])
        if ar > 1.25:
            h_max = image.shape[0] > image.shape[1]
            if h_max:
                resized = Image.fromarray(image).resize((sizes[0], sizes[1]))
            else:
                resized = Image.fromarray(image).resize((sizes[1], sizes[0]))
        else:
            resized = Image.fromarray(image).resize((sizes[0], sizes[0]))
        image = np.array(resized)
        if image.shape != org_size:
            print(
                f"Input image has been resized to {image.shape[1]}x{image.shape[0]}px (from {org_size[1]}x{org_size[0]}px)"
            )
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image, resized

    tmps = []
    resized_imgs = []
    if isinstance(images, list):
        for item in images:
            prep, resized = pre_process(item, sizes, left, right, top, bottom)
            if len(tmps) > 0 and prep.shape != tmps[0].shape:
                raise ValueError(
                    f"Mixed image resolution not supported in batch processing. Target resolution set to {tmps[0].shape[2]}x{tmps[0].shape[1]}px,"
                    f"but found image with resolution {prep.shape[2]}x{prep.shape[1]}px"
                )
            tmps.append(prep)
            resized_imgs.append(resized)
    else:
        prep, resized = pre_process(images, sizes, left, right, top, bottom)
        tmps.append(prep)
        resized_imgs.append(resized)
    image = torch.stack(tmps) / 127.5 - 1

    image = image.to(device=device, dtype=dtype)
    return image, resized_imgs


class LEditsPPPipelineIF(DiffusionPipeline, LoraLoaderMixin):
    """
    Pipeline for textual image editing using LEDits++ with Deepfloyd-IF.

    This model inherits from [`DiffusionPipeline`] and builds on the [`IFPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`LEditsPPPipelineIF.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    Args:
        tokenizer ([`~transformers.T5Tokenizer`]):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/t5#transformers.T5Tokenizer).
        text_encoder ([`~transformers.T5EncoderModel`]):
            Frozen text-encoder of class
            [T5EncoderModel](https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/t5#transformers.T5EncoderModel).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]. If any other scheduler is passed it will automatically
            be set to [`DPMSolverMultistepScheduler`].
        safety_checker ([`IFSafetyChecker`], *optional*):
            Classification module that estimates whether generated images could be considered offensive or harmful.
        feature_extractor ([`~transformers.CLIPImageProcessor`], *optional*):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        watermarker ([`IFWatermarker`], *optional*):
            Model that adds a watermark to generated images.

    """

    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    unet: UNet2DConditionModel
    scheduler: DPMSolverMultistepScheduler

    feature_extractor: Optional[CLIPImageProcessor]
    safety_checker: Optional[IFSafetyChecker]

    watermarker: Optional[IFWatermarker]

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]
    model_cpu_offload_seq = "text_encoder->unet"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if not isinstance(scheduler, DDIMScheduler) or not isinstance(scheduler, DPMSolverMultistepScheduler):
            conf = scheduler.config
            conf["clip_sample"] = False
            conf["thresholding"] = False
            conf["variance_type"] = "fixed"

            scheduler = DPMSolverMultistepScheduler.from_config(conf, algorithm_type="sde-dpmsolver++", solver_order=2)
            logger.warning(
                "This pipeline only supports DDIMScheduler and DPMSolverMultistepScheduler. "
                "The scheduler has been changed to DPMSolverMultistepScheduler."
            )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the IF license and do not expose unfiltered"
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

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            watermarker=watermarker,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def remove_all_hooks(self):
        if is_accelerate_available():
            from accelerate.hooks import remove_hook_from_module
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        for model in [self.text_encoder, self.unet, self.safety_checker]:
            if model is not None:
                remove_hook_from_module(model, recurse=True)

        self.unet_offload_hook = None
        self.text_encoder_offload_hook = None
        self.final_offload_hook = None

    @torch.no_grad()
    def encode_prompt(
        self,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        enable_edit_guidance: bool = True,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        edit_prompt_embeds: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            enable_edit_guidance (`bool`, *optional*, defaults to `True`):
                Whether to guide towards editing prompt(s) or not.
            editing_prompt (`str` or `List[str]`, *optional*):
                Editing prompt(s) to be encoded. If not defined and 'enable_edit_guidance' is True, one has to pass
                `editing_prompt_embeds` instead.
            edit_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided and 'enable_edit_guidance' is True, editing_prompt_embeds will be generated
                from `editing_prompt` input argument.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
        """
        if device is None:
            device = self._execution_device

        batch_size = self.batch_size

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        # get unconditional embeddings for classifier free guidance
        if negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but batch size {batch_size} is expected."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        num_edit_tokens = 0
        if enable_edit_guidance:
            if edit_prompt_embeds is None:
                edit_tokens: List[str]
                if isinstance(editing_prompt, str):
                    edit_tokens = [editing_prompt]
                else:
                    edit_tokens = editing_prompt
                edit_tokens = [x for item in edit_tokens for x in repeat(item, batch_size)]
                edit_tokens = self._text_preprocessing(edit_tokens, clean_caption=clean_caption)

                edit_input = self.tokenizer(
                    edit_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                    return_length=True,
                )
                num_edit_tokens = edit_input.length - 1  # not counting endoftext

                attention_mask = edit_input.attention_mask.to(device)

                edit_prompt_embeds = self.text_encoder(
                    edit_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                edit_prompt_embeds = edit_prompt_embeds[0]

            bs_embed_edit, seq_len_edit, _ = edit_prompt_embeds.shape

            edit_prompt_embeds = edit_prompt_embeds.to(dtype=dtype, device=device)
            edit_prompt_embeds = edit_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            edit_prompt_embeds = edit_prompt_embeds.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        return negative_prompt_embeds, edit_prompt_embeds, num_edit_tokens

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype),
            )
        else:
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()

        return image, nsfw_detected, watermark_detected

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

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        # TODO
        pass

    def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, device, intermediate_images):
        shape = (batch_size, num_channels, height, width)

        if intermediate_images.shape != shape:
            raise ValueError(f"Unexpected image shape, got {intermediate_images.shape}, expected {shape}")

        intermediate_images = intermediate_images.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        return intermediate_images

    def prepare_unet(self, attention_store):
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

            attn_procs[name] = CrossAttnProcessor(
                attention_store=attention_store,
                place_in_unet=place_in_unet,
                PnP=False,
                editing_prompts=self.enabled_editing_prompts,
            )

        self.unet.set_attn_processor(attn_procs)

    def _text_preprocessing(self, text, clean_caption=False):
        if clean_caption and not is_bs4_available():
            logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
            logger.warn("Setting `clean_caption` to False...")
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        edit_prompt_embeds: Optional[torch.FloatTensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 0,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        sem_guidance: Optional[List[torch.Tensor]] = None,
        use_cross_attn_mask: bool = False,
        use_intersect_mask: bool = True,
        user_mask: Optional[torch.FloatTensor] = None,
        attn_store_steps: Optional[List[int]] = [],
        store_averaged_over_steps: bool = True,
    ):
        r"""
        The call function to the pipeline for editing. The [`~pipelines.ledits_pp.LEditsPPPipelineStableDiffusion.invert`]
        method has to be called beforehand. Edits will always be performed for the last inverted image(s).

        Args:
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. The image is reconstructed by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via `reverse_editing_direction`.
            edit_prompt_embeds (`torch.Tensor>`, *optional*):
                Pre-computed embeddings to use for guiding the image generation. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for guiding the image generation. If provided as list values should correspond to `editing_prompt`.
                `edit_guidance_scale` is defined as `s_e` of equation 12 of
                [LEDITS++ Paper](https://arxiv.org/abs/2301.12247).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which guidance will not be applied.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Masking threshold of guidance. Threshold should be proportional to the image region that is modified.
                'edit_threshold' is defined as 'λ' of equation 12 of [LEDITS++ Paper](https://arxiv.org/abs/2301.12247).
            user_mask (`torch.FloatTensor`, *optional*):
                User-provided mask for even better control over the editing process. This is helpful when LEDITS++'s implicit
                masks do not meet user preferences.
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.
            use_cross_attn_mask (`bool`, defaults to `False`):
                Whether cross-attention masks are used. Cross-attention masks are always used when use_intersect_mask
                is set to true. Cross-attention masks are defined as 'M^1' of equation 12 of
                [LEDITS++ paper](https://arxiv.org/pdf/2311.16711.pdf).
            use_intersect_mask (`bool`, defaults to `True`):
                Whether the masking term is calculated as intersection of cross-attention masks and masks derived
                from the noise estimate. Cross-attention mask are defined as 'M^1' and masks derived from the noise
                estimate are defined as 'M^2' of equation 12 of [LEDITS++ paper](https://arxiv.org/pdf/2311.16711.pdf).
            attn_store_steps (`List[int]`, *optional*):
                Steps for which the attention maps are stored in the AttentionStore. Just for visualization purposes.
            store_averaged_over_steps (`bool`, defaults to `True`):
                Whether the attention maps for the 'attn_store_steps' are stored averaged over the diffusion steps.
                If False, attention maps for each step are stores separately. Just for visualization purposes.

        Examples:

        Returns:
            [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.ledits_pp.LEditsPPDiffusionPipelineOutput`] if `return_dict` is True,
            otherwise a `tuple. When returning a tuple, the first element is a list with the generated images, and the
            second element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        eta = self.eta
        num_images_per_prompt = 1
        intermediate_images = self.init_images
        device = self._execution_device

        zs = self.zs
        reset_dpm(self.scheduler)

        if use_intersect_mask:
            use_cross_attn_mask = True

        if use_cross_attn_mask:
            self.smoothing = LeditsGaussianSmoothing(device)

        if user_mask is not None:
            user_mask = user_mask.to(self.device)

        # 1. Check inputs. Raise error if not correct
        # TODO
        # self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif edit_prompt_embeds is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = int(edit_prompt_embeds.shape[0] / self.batch_size)
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

        # 3. Encode input prompt
        negative_prompt_embeds, edit_prompt_embeds, num_edit_tokens = self.encode_prompt(
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            enable_edit_guidance=enable_edit_guidance,
            editing_prompt=editing_prompt,
            edit_prompt_embeds=edit_prompt_embeds,
            clean_caption=clean_caption,
        )

        if enable_edit_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, edit_prompt_embeds])
            self.text_cross_attention_maps = [editing_prompt] if isinstance(editing_prompt, str) else editing_prompt
        else:
            prompt_embeds = negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps = self.scheduler.inversion_steps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}

        self.attention_store = LeditsAttentionStore(
            average=store_averaged_over_steps,
            batch_size=self.batch_size,
            max_size=(intermediate_images.shape[-2] / 4.0) * (intermediate_images.shape[-1] / 4.0),
            max_resolution=None,
        )
        self.prepare_unet(self.attention_store)
        resolution = intermediate_images.shape[-2:]
        att_res = (int(resolution[0] / 4), int(resolution[1] / 4))

        # 5. Prepare intermediate images
        intermediate_images = self.prepare_intermediate_images(
            self.batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            self.size[0],
            self.size[1],
            prompt_embeds.dtype,
            device,
            intermediate_images,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 7. Denoising loop
        self.sem_guidance = None
        self.activation_mask = None

        num_warmup_steps = 0
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([intermediate_images] * (1 + self.enabled_editing_prompts))
                model_input = self.scheduler.scale_model_input(model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)
                noise_pred_uncond = noise_pred_out[0]
                noise_pred_uncond, predicted_variance = noise_pred_uncond.split(model_input.shape[1], dim=1)

                noise_guidance_edit = torch.zeros(
                    noise_pred_uncond.shape,
                    device=device,
                    dtype=noise_pred_uncond.dtype,
                )

                if sem_guidance is not None and len(sem_guidance) > i:
                    noise_guidance_edit += sem_guidance[i].to(device)

                if enable_edit_guidance:
                    noise_pred_edit_concepts = noise_pred_out[1:]

                    if self.activation_mask is None:
                        self.activation_mask = torch.zeros(
                            (len(timesteps), self.enabled_editing_prompts, *noise_pred_uncond.shape)
                        )
                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_uncond.shape))

                    for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                        noise_pred_edit_concept, _ = noise_pred_edit_concept.split(model_input.shape[1], dim=1)

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
                            attn_map = out[:, :, :, : num_edit_tokens[c]]  # there is no startoftext

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
                            ).repeat(1, self.unet.config.in_channels, 1, 1)
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
                                noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                        elif not use_cross_attn_mask:
                            # calculate quantile
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
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                idx = t_to_idx[int(t)]
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, variance_noise=zs[idx], **extra_step_kwargs, return_dict=False
                )[0]

                # step callback
                store_step = i in attn_store_steps
                self.attention_store.between_steps(store_step)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images

        if output_type == "pil":
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)

            # 11. Apply watermark
            if self.watermarker is not None:
                image = self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pt":
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, nsfw_detected, watermark_detected)

        return LEditsPPDiffusionPipelineOutput(images=image, nsfw_content_detected=nsfw_detected)

    @torch.no_grad()
    def invert(
        self,
        image: PipelineImageInput,
        source_prompt: str = "",
        source_guidance_scale=3.5,
        negative_prompt: str = None,
        num_inversion_steps: int = 100,
        skip: float = 0.15,
        generator: Optional[torch.Generator] = None,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        The function to the pipeline for image inversion as described by the [LEDITS++ Paper](https://arxiv.org/abs/2301.12247).
        If the scheduler is set to [`~schedulers.DDIMScheduler`] the inversion proposed by [edit-friendly DPDM](https://arxiv.org/abs/2304.06140)
        will be performed instead.

         Args:
            image (`PipelineImageInput`):
                Input for the image(s) that are to be edited. Multiple input images have to default to the same aspect
                ratio.
            source_prompt (`str`, defaults to `""`):
                Prompt describing the input image that will be used for guidance during inversion. Guidance is disabled
                if the `source_prompt` is `""`.
            source_guidance_scale (`float`, defaults to `3.5`):
                Strength of guidance during inversion.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inversion_steps (`int`, defaults to `30`):
                Number of total performed inversion steps after discarding the initial `skip` steps.
            skip (`float`, defaults to `0.15`):
                Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values
                will lead to stronger changes to the input image. `skip` has to be between `0` and `1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                inversion deterministic.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided prompts before encoding.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Returns:
            [`~pipelines.ledits_pp.LEditsPPInversionPipelineOutput`]:
            Output will contain the resized input image(s) and respective VAE reconstruction(s).
        """
        # Reset attn processor, we do not want to store attn maps during inversion
        self.unet.set_attn_processor(AttnAddedKVProcessor())

        self.eta = 1.0

        self.scheduler.config.timestep_spacing = "leading"
        self.scheduler.set_timesteps(int(num_inversion_steps * (1 + skip)))
        self.scheduler.inversion_steps = self.scheduler.timesteps[-num_inversion_steps:]
        timesteps = self.scheduler.inversion_steps

        device = self._execution_device
        dtype = self.text_encoder.dtype
        num_images_per_prompt = 1

        reset_dpm(self.scheduler)

        # 1. prepare image
        x0, resized = load_images(
            image,
            sizes=(
                self.unet.sample_size,
                int(self.unet.sample_size * 1.5),
            ),
            device=device,
            dtype=dtype,
        )
        width = x0.shape[2]
        height = x0.shape[3]
        self.size = (height, width)

        self.batch_size = x0.shape[0]

        # 2. get embeddings
        do_classifier_free_guidance = source_guidance_scale > 1.0

        if isinstance(source_prompt, str):
            source_prompt = [source_prompt] * self.batch_size

        uncond_embedding, text_embeddings, _ = self.encode_prompt(
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            enable_edit_guidance=do_classifier_free_guidance,
            editing_prompt=source_prompt,
            clean_caption=clean_caption,
        )

        prompt_embeds = torch.cat([uncond_embedding, text_embeddings])

        # 3. find zs and xts
        variance_noise_shape = (num_inversion_steps, *x0.shape)

        # intermediate images
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=device, dtype=uncond_embedding.dtype)

        for t in reversed(timesteps):
            idx = num_inversion_steps - t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, torch.Tensor([t]))
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=device, dtype=uncond_embedding.dtype)

        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for t in timesteps:
                idx = num_inversion_steps - t_to_idx[int(t)] - 1

                # 1. predict noise residual
                xt = xts[idx + 1]
                model_input = torch.cat([xt] * 2)
                noise_pred = self.unet(
                    model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)

                noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * source_guidance_scale

                xtm1 = xts[idx]
                z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, self.eta)
                zs[idx] = z

                # correction to avoid error accumulation
                xts[idx] = xtm1_corrected

                progress_bar.update()

        self.init_images = xts[-1].expand(self.batch_size, -1, -1, -1)
        zs = zs.flip(0)
        self.zs = zs

        if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
            self.unet_offload_hook.offload()

        # Offload all models
        self.maybe_free_model_hooks()

        return LEditsPPInversionPipelineOutput(images=x0, vae_reconstruction_images=None)


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.reset_dpm
def reset_dpm(scheduler):
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        scheduler.model_outputs = [
            None,
        ] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0
        scheduler._step_index = None


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.compute_noise_ddim
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


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.compute_noise_sde_dpm_pp_2nd
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


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.compute_noise
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
