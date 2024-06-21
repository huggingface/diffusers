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

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    XFormersAttnProcessor,
)
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import DDIMScheduler, DPMSolverMultistepScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput


if is_invisible_watermark_available():
    from ..stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import PIL
        >>> import requests
        >>> from io import BytesIO

        >>> from diffusers import LEditsPPPipelineStableDiffusionXL

        >>> pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg"
        >>> image = download_image(img_url)

        >>> _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.2)

        >>> edited_image = pipe(
        ...     editing_prompt=["tennis ball", "tomato"],
        ...     reverse_editing_direction=[True, False],
        ...     edit_guidance_scale=[5.0, 10.0],
        ...     edit_threshold=[0.9, 0.85],
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


# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEDITSCrossAttnProcessor
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


class LEditsPPPipelineStableDiffusionXL(
    DiffusionPipeline,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    """
    Pipeline for textual image editing using LEDits++ with Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`] and builds on the [`StableDiffusionXLPipeline`]. Check the
    superclass documentation for the generic methods implemented for all pipelines (downloading, saving, running on a
    particular device, etc.).

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`LEditsPPPipelineStableDiffusionXL.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer ([`~transformers.CLIPTokenizer`]):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DPMSolverMultistepScheduler`] or [`DDIMScheduler`]. If any other scheduler is passed it will
            automatically be set to [`DPMSolverMultistepScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DPMSolverMultistepScheduler, DDIMScheduler],
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if not isinstance(scheduler, DDIMScheduler) and not isinstance(scheduler, DPMSolverMultistepScheduler):
            self.scheduler = DPMSolverMultistepScheduler.from_config(
                scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2
            )
            logger.warning(
                "This pipeline only supports DDIMScheduler and DPMSolverMultistepScheduler. "
                "The scheduler has been changed to DPMSolverMultistepScheduler."
            )

        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None
        self.inversion_steps = None

    def encode_prompt(
        self,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        enable_edit_guidance: bool = True,
        editing_prompt: Optional[str] = None,
        editing_prompt_embeds: Optional[torch.Tensor] = None,
        editing_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> object:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            enable_edit_guidance (`bool`):
                Whether to guide towards an editing prompt or not.
            editing_prompt (`str` or `List[str]`, *optional*):
                Editing prompt(s) to be encoded. If not defined and 'enable_edit_guidance' is True, one has to pass
                `editing_prompt_embeds` instead.
            editing_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided and 'enable_edit_guidance' is True, editing_prompt_embeds will be generated from
                `editing_prompt` input argument.
            editing_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated edit pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled editing_pooled_prompt_embeds will be generated from `editing_prompt`
                input argument.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        batch_size = self.batch_size

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        num_edit_tokens = 0

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt

        if negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]

            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but image inversion "
                    f" has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of the input images."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

            if zero_out_negative_prompt:
                negative_prompt_embeds = torch.zeros_like(negative_prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(negative_pooled_prompt_embeds)

        if enable_edit_guidance and editing_prompt_embeds is None:
            editing_prompt_2 = editing_prompt

            editing_prompts = [editing_prompt, editing_prompt_2]
            edit_prompt_embeds_list = []

            for editing_prompt, tokenizer, text_encoder in zip(editing_prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    editing_prompt = self.maybe_convert_prompt(editing_prompt, tokenizer)

                max_length = negative_prompt_embeds.shape[1]
                edit_concepts_input = tokenizer(
                    # [x for item in editing_prompt for x in repeat(item, batch_size)],
                    editing_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_length=True,
                )
                num_edit_tokens = edit_concepts_input.length - 2

                edit_concepts_embeds = text_encoder(
                    edit_concepts_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                editing_pooled_prompt_embeds = edit_concepts_embeds[0]
                if clip_skip is None:
                    edit_concepts_embeds = edit_concepts_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    edit_concepts_embeds = edit_concepts_embeds.hidden_states[-(clip_skip + 2)]

                edit_prompt_embeds_list.append(edit_concepts_embeds)

            edit_concepts_embeds = torch.concat(edit_prompt_embeds_list, dim=-1)
        elif not enable_edit_guidance:
            edit_concepts_embeds = None
            editing_pooled_prompt_embeds = None

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = negative_prompt_embeds.shape
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if enable_edit_guidance:
            bs_embed_edit, seq_len, _ = edit_concepts_embeds.shape
            edit_concepts_embeds = edit_concepts_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            edit_concepts_embeds = edit_concepts_embeds.repeat(1, num_images_per_prompt, 1)
            edit_concepts_embeds = edit_concepts_embeds.view(bs_embed_edit * num_images_per_prompt, seq_len, -1)

        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        if enable_edit_guidance:
            editing_pooled_prompt_embeds = editing_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed_edit * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            negative_prompt_embeds,
            edit_concepts_embeds,
            negative_pooled_prompt_embeds,
            editing_pooled_prompt_embeds,
            num_edit_tokens,
        )

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
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, device, latents):
        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
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
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion.prepare_unet
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        denoising_end: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeddings: Optional[torch.Tensor] = None,
        editing_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 0,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        sem_guidance: Optional[List[torch.Tensor]] = None,
        use_cross_attn_mask: bool = False,
        use_intersect_mask: bool = False,
        user_mask: Optional[torch.Tensor] = None,
        attn_store_steps: Optional[List[int]] = [],
        store_averaged_over_steps: bool = True,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for editing. The
        [`~pipelines.ledits_pp.LEditsPPPipelineStableDiffusionXL.invert`] method has to be called beforehand. Edits
        will always be performed for the last inverted image(s).

        Args:
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
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
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. The image is reconstructed by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-generated edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, editing_prompt_embeddings will be generated from `editing_prompt` input argument.
            editing_pooled_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-generated pooled edit text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, editing_prompt_embeddings will be generated from `editing_prompt` input
                argument.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for guiding the image generation. If provided as list values should correspond to
                `editing_prompt`. `edit_guidance_scale` is defined as `s_e` of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which guidance is not applied.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which guidance is no longer applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Masking threshold of guidance. Threshold should be proportional to the image region that is modified.
                'edit_threshold' is defined as 'λ' of equation 12 of [LEDITS++
                Paper](https://arxiv.org/abs/2301.12247).
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.
            use_cross_attn_mask:
                Whether cross-attention masks are used. Cross-attention masks are always used when use_intersect_mask
                is set to true. Cross-attention masks are defined as 'M^1' of equation 12 of [LEDITS++
                paper](https://arxiv.org/pdf/2311.16711.pdf).
            use_intersect_mask:
                Whether the masking term is calculated as intersection of cross-attention masks and masks derived from
                the noise estimate. Cross-attention mask are defined as 'M^1' and masks derived from the noise estimate
                are defined as 'M^2' of equation 12 of [LEDITS++ paper](https://arxiv.org/pdf/2311.16711.pdf).
            user_mask:
                User-provided mask for even better control over the editing process. This is helpful when LEDITS++'s
                implicit masks do not meet user preferences.
            attn_store_steps:
                Steps for which the attention maps are stored in the AttentionStore. Just for visualization purposes.
            store_averaged_over_steps:
                Whether the attention maps for the 'attn_store_steps' are stored averaged over the diffusion steps. If
                False, attention maps for each step are stores separately. Just for visualization purposes.
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
            returning a tuple, the first element is a list with the generated images.
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

        # TODO: Check inputs
        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #    callback_steps,
        #    negative_prompt,
        #    negative_prompt_2,
        #    prompt_embeds,
        #    negative_prompt_embeds,
        #    pooled_prompt_embeds,
        #    negative_pooled_prompt_embeds,
        # )
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        batch_size = self.batch_size

        device = self._execution_device

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            edit_prompt_embeds,
            negative_pooled_prompt_embeds,
            pooled_edit_embeds,
            num_edit_tokens,
        ) = self.encode_prompt(
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
            enable_edit_guidance=enable_edit_guidance,
            editing_prompt=editing_prompt,
            editing_prompt_embeds=editing_prompt_embeddings,
            editing_pooled_prompt_embeds=editing_pooled_prompt_embeds,
        )

        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.inversion_steps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}

        if use_cross_attn_mask:
            self.attention_store = LeditsAttentionStore(
                average=store_averaged_over_steps,
                batch_size=batch_size,
                max_size=(latents.shape[-2] / 4.0) * (latents.shape[-1] / 4.0),
                max_resolution=None,
            )
            self.prepare_unet(self.attention_store)
            resolution = latents.shape[-2:]
            att_res = (int(resolution[0] / 4), int(resolution[1] / 4))

        # 5. Prepare latent variables
        latents = self.prepare_latents(device=device, latents=latents)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(negative_pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        # 7. Prepare added time ids & embeddings
        add_text_embeds = negative_pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            self.size,
            crops_coords_top_left,
            self.size,
            dtype=negative_pooled_prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if enable_edit_guidance:
            prompt_embeds = torch.cat([prompt_embeds, edit_prompt_embeds], dim=0)
            add_text_embeds = torch.cat([add_text_embeds, pooled_edit_embeds], dim=0)
            edit_concepts_time_ids = add_time_ids.repeat(edit_prompt_embeds.shape[0], 1)
            add_time_ids = torch.cat([add_time_ids, edit_concepts_time_ids], dim=0)
            self.text_cross_attention_maps = [editing_prompt] if isinstance(editing_prompt, str) else editing_prompt

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            # TODO: fix image encoding
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        self.sem_guidance = None
        self.activation_mask = None

        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=self._num_timesteps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * (1 + self.enabled_editing_prompts))
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

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
                            (len(timesteps), self.enabled_editing_prompts, *noise_pred_edit_concepts[0].shape)
                        )
                    if self.sem_guidance is None:
                        self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_uncond.shape))

                    # noise_guidance_edit = torch.zeros_like(noise_guidance)
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

                # compute the previous noisy sample x_t -> x_t-1
                if enable_edit_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_edit_concepts.mean(dim=0, keepdim=False),
                        guidance_rescale=self.guidance_rescale,
                    )

                idx = t_to_idx[int(t)]
                latents = self.scheduler.step(
                    noise_pred, t, latents, variance_noise=zs[idx], **extra_step_kwargs, return_dict=False
                )[0]

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
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    # negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return LEditsPPDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    @torch.no_grad()
    # Modified from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion.encode_image
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
        image = image.to(self.device, dtype=dtype)
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            image = image.float()
            self.upcast_vae()

        x0 = self.vae.encode(image).latent_dist.mode()
        x0 = x0.to(dtype)
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        x0 = self.vae.config.scaling_factor * x0
        return x0, resized

    @torch.no_grad()
    def invert(
        self,
        image: PipelineImageInput,
        source_prompt: str = "",
        source_guidance_scale=3.5,
        negative_prompt: str = None,
        negative_prompt_2: str = None,
        num_inversion_steps: int = 50,
        skip: float = 0.15,
        generator: Optional[torch.Generator] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        num_zero_noise_steps: int = 3,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
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
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_inversion_steps (`int`, defaults to `50`):
                Number of total performed inversion steps after discarding the initial `skip` steps.
            skip (`float`, defaults to `0.15`):
                Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values
                will lead to stronger changes to the input image. `skip` has to be between `0` and `1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make inversion
                deterministic.
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            num_zero_noise_steps (`int`, defaults to `3`):
                Number of final diffusion steps that will not renoise the current image. If no steps are set to zero
                SD-XL in combination with [`DPMSolverMultistepScheduler`] will produce noise artifacts.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

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

        num_images_per_prompt = 1

        device = self._execution_device

        # 0. Ensure that only uncond embedding is used if prompt = ""
        if source_prompt == "":
            # noise pred should only be noise_pred_uncond
            source_guidance_scale = 0.0
            do_classifier_free_guidance = False
        else:
            do_classifier_free_guidance = source_guidance_scale > 1.0

        # 1. prepare image
        x0, resized = self.encode_image(image, dtype=self.text_encoder_2.dtype)
        width = x0.shape[2] * self.vae_scale_factor
        height = x0.shape[3] * self.vae_scale_factor
        self.size = (height, width)

        self.batch_size = x0.shape[0]

        # 2. get embeddings
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        if isinstance(source_prompt, str):
            source_prompt = [source_prompt] * self.batch_size

        (
            negative_prompt_embeds,
            prompt_embeds,
            negative_pooled_prompt_embeds,
            edit_pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            editing_prompt=source_prompt,
            lora_scale=text_encoder_lora_scale,
            enable_edit_guidance=do_classifier_free_guidance,
        )
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(negative_pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        # 3. Prepare added time ids & embeddings
        add_text_embeds = negative_pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            self.size,
            crops_coords_top_left,
            self.size,
            dtype=negative_prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([add_text_embeds, edit_pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        negative_prompt_embeds = negative_prompt_embeds.to(device)

        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(self.batch_size * num_images_per_prompt, 1)

        # autoencoder reconstruction
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            x0_tmp = x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image_rec = self.vae.decode(
                x0_tmp / self.vae.config.scaling_factor, return_dict=False, generator=generator
            )[0]
        elif self.vae.config.force_upcast:
            x0_tmp = x0.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            image_rec = self.vae.decode(
                x0_tmp / self.vae.config.scaling_factor, return_dict=False, generator=generator
            )[0]
        else:
            image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

        image_rec = self.image_processor.postprocess(image_rec, output_type="pil")

        # 5. find zs and xts
        variance_noise_shape = (num_inversion_steps, *x0.shape)

        # intermediate latents
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=negative_prompt_embeds.dtype)

        for t in reversed(timesteps):
            idx = num_inversion_steps - t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, t.unsqueeze(0))
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=negative_prompt_embeds.dtype)

        self.scheduler.set_timesteps(len(self.scheduler.timesteps))

        for t in self.progress_bar(timesteps):
            idx = num_inversion_steps - t_to_idx[int(t)] - 1
            # 1. predict noise residual
            xt = xts[idx + 1]

            latent_model_input = torch.cat([xt] * 2) if do_classifier_free_guidance else xt
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=negative_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # 2. perform guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2)
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred = noise_pred_uncond + source_guidance_scale * (noise_pred_text - noise_pred_uncond)

            xtm1 = xts[idx]
            z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, self.eta)
            zs[idx] = z

            # correction to avoid error accumulation
            xts[idx] = xtm1_corrected

        self.init_latents = xts[-1]
        zs = zs.flip(0)

        if num_zero_noise_steps > 0:
            zs[-num_zero_noise_steps:] = torch.zeros_like(zs[-num_zero_noise_steps:])
        self.zs = zs
        return LEditsPPInversionPipelineOutput(images=resized, vae_reconstruction_images=image_rec)


# Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.rescale_noise_cfg
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
