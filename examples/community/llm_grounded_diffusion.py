# Copyright 2024 Long Lian, the GLIGEN Authors, and The HuggingFace Team. All rights reserved.
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

# This is a single file implementation of LMD+. See README.md for examples.

import ast
import gc
import inspect
import math
import warnings
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import Attention, GatedSelfAttentionDense
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
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


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained(
        ...     "longlian/lmd_plus",
        ...     custom_pipeline="llm_grounded_diffusion",
        ...     custom_revision="main",
        ...     variant="fp16", torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> # Generate an image described by the prompt and
        >>> # insert objects described by text at the region defined by bounding boxes
        >>> prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage"
        >>> boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
        >>> phrases = ["a waterfall", "a modern high speed train"]

        >>> images = pipe(
        ...     prompt=prompt,
        ...     phrases=phrases,
        ...     boxes=boxes,
        ...     gligen_scheduled_sampling_beta=0.4,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ...     lmd_guidance_kwargs={}
        ... ).images

        >>> images[0].save("./lmd_plus_generation.jpg")

        >>> # Generate directly from a text prompt and an LLM response
        >>> prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage"
        >>> phrases, boxes, bg_prompt, neg_prompt = pipe.parse_llm_response(\"""
        [('a waterfall', [71, 105, 148, 258]), ('a modern high speed train', [255, 223, 181, 149])]
        Background prompt: A beautiful forest with fall foliage
        Negative prompt:
        \""")

        >> images = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=neg_prompt,
        ...     phrases=phrases,
        ...     boxes=boxes,
        ...     gligen_scheduled_sampling_beta=0.4,
        ...     output_type="pil",
        ...     num_inference_steps=50,
        ...     lmd_guidance_kwargs={}
        ... ).images

        >>> images[0].save("./lmd_plus_generation.jpg")

images[0]

        ```
"""

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# All keys in Stable Diffusion models: [('down', 0, 0, 0), ('down', 0, 1, 0), ('down', 1, 0, 0), ('down', 1, 1, 0), ('down', 2, 0, 0), ('down', 2, 1, 0), ('mid', 0, 0, 0), ('up', 1, 0, 0), ('up', 1, 1, 0), ('up', 1, 2, 0), ('up', 2, 0, 0), ('up', 2, 1, 0), ('up', 2, 2, 0), ('up', 3, 0, 0), ('up', 3, 1, 0), ('up', 3, 2, 0)]
# Note that the first up block is `UpBlock2D` rather than `CrossAttnUpBlock2D` and does not have attention. The last index is always 0 in our case since we have one `BasicTransformerBlock` in each `Transformer2DModel`.
DEFAULT_GUIDANCE_ATTN_KEYS = [
    ("mid", 0, 0, 0),
    ("up", 1, 0, 0),
    ("up", 1, 1, 0),
    ("up", 1, 2, 0),
]


def convert_attn_keys(key):
    """Convert the attention key from tuple format to the torch state format"""

    if key[0] == "mid":
        assert key[1] == 0, f"mid block only has one block but the index is {key[1]}"
        return f"{key[0]}_block.attentions.{key[2]}.transformer_blocks.{key[3]}.attn2.processor"

    return f"{key[0]}_blocks.{key[1]}.attentions.{key[2]}.transformer_blocks.{key[3]}.attn2.processor"


DEFAULT_GUIDANCE_ATTN_KEYS = [convert_attn_keys(key) for key in DEFAULT_GUIDANCE_ATTN_KEYS]


def scale_proportion(obj_box, H, W):
    # Separately rounding box_w and box_h to allow shift invariant box sizes. Otherwise box sizes may change when both coordinates being rounded end with ".5".
    x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
    box_w, box_h = round((obj_box[2] - obj_box[0]) * W), round((obj_box[3] - obj_box[1]) * H)
    x_max, y_max = x_min + box_w, y_min + box_h

    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, W), min(y_max, H)

    return x_min, y_min, x_max, y_max


# Adapted from the parent class `AttnProcessor2_0`
class AttnProcessorWithHook(AttnProcessor2_0):
    def __init__(
        self,
        attn_processor_key,
        hidden_size,
        cross_attention_dim,
        hook=None,
        fast_attn=True,
        enabled=True,
    ):
        super().__init__()
        self.attn_processor_key = attn_processor_key
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.hook = hook
        self.fast_attn = fast_attn
        self.enabled = enabled

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
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

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if (self.hook is not None and self.enabled) or not self.fast_attn:
            query_batch_dim = attn.head_to_batch_dim(query)
            key_batch_dim = attn.head_to_batch_dim(key)
            value_batch_dim = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query_batch_dim, key_batch_dim, attention_mask)

        if self.hook is not None and self.enabled:
            # Call the hook with query, key, value, and attention maps
            self.hook(
                self.attn_processor_key,
                query_batch_dim,
                key_batch_dim,
                value_batch_dim,
                attention_probs,
            )

        if self.fast_attn:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attention_mask is not None:
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LLMGroundedDiffusionPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for layout-grounded text-to-image generation using LLM-grounded Diffusion (LMD+): https://arxiv.org/pdf/2305.13655.pdf.

    This model inherits from [`StableDiffusionPipeline`] and aims at implementing the pipeline with minimal modifications. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    This is a simplified implementation that does not perform latent or attention transfer from single object generation to overall generation. The final image is generated directly with attention and adapters control.

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
        requires_safety_checker (bool):
            Whether a safety checker is needed for this pipeline.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    objects_text = "Objects: "
    bg_prompt_text = "Background prompt: "
    bg_prompt_text_no_trailing_space = bg_prompt_text.rstrip()
    neg_prompt_text = "Negative prompt: "
    neg_prompt_text_no_trailing_space = neg_prompt_text.rstrip()

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
        # This is copied from StableDiffusionPipeline, with hook initizations for LMD+.
        super().__init__()

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
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        # Initialize the attention hooks for LLM-grounded Diffusion
        self.register_attn_hooks(unet)
        self._saved_attn = None

    def attn_hook(self, name, query, key, value, attention_probs):
        if name in DEFAULT_GUIDANCE_ATTN_KEYS:
            self._saved_attn[name] = attention_probs

    @classmethod
    def convert_box(cls, box, height, width):
        # box: x, y, w, h (in 512 format) -> x_min, y_min, x_max, y_max
        x_min, y_min = box[0] / width, box[1] / height
        w_box, h_box = box[2] / width, box[3] / height

        x_max, y_max = x_min + w_box, y_min + h_box

        return x_min, y_min, x_max, y_max

    @classmethod
    def _parse_response_with_negative(cls, text):
        if not text:
            raise ValueError("LLM response is empty")

        if cls.objects_text in text:
            text = text.split(cls.objects_text)[1]

        text_split = text.split(cls.bg_prompt_text_no_trailing_space)
        if len(text_split) == 2:
            gen_boxes, text_rem = text_split
        else:
            raise ValueError(f"LLM response is incomplete: {text}")

        text_split = text_rem.split(cls.neg_prompt_text_no_trailing_space)

        if len(text_split) == 2:
            bg_prompt, neg_prompt = text_split
        else:
            raise ValueError(f"LLM response is incomplete: {text}")

        try:
            gen_boxes = ast.literal_eval(gen_boxes)
        except SyntaxError as e:
            # Sometimes the response is in plain text
            if "No objects" in gen_boxes or gen_boxes.strip() == "":
                gen_boxes = []
            else:
                raise e
        bg_prompt = bg_prompt.strip()
        neg_prompt = neg_prompt.strip()

        # LLM may return "None" to mean no negative prompt provided.
        if neg_prompt == "None":
            neg_prompt = ""

        return gen_boxes, bg_prompt, neg_prompt

    @classmethod
    def parse_llm_response(cls, response, canvas_height=512, canvas_width=512):
        # Infer from spec
        gen_boxes, bg_prompt, neg_prompt = cls._parse_response_with_negative(text=response)

        gen_boxes = sorted(gen_boxes, key=lambda gen_box: gen_box[0])

        phrases = [name for name, _ in gen_boxes]
        boxes = [cls.convert_box(box, height=canvas_height, width=canvas_width) for _, box in gen_boxes]

        return phrases, boxes, bg_prompt, neg_prompt

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        phrases,
        boxes,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        phrase_indices=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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
        elif prompt is None and phrase_indices is None:
            raise ValueError("If the prompt is None, the phrase_indices cannot be None")

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

        if len(phrases) != len(boxes):
            raise ValueError(
                "length of `phrases` and `boxes` has to be same, but"
                f" got: `phrases` {len(phrases)} != `boxes` {len(boxes)}"
            )

    def register_attn_hooks(self, unet):
        """Registering hooks to obtain the attention maps for guidance"""

        attn_procs = {}

        for name in unet.attn_processors.keys():
            # Only obtain the queries and keys from cross-attention
            if name.endswith("attn1.processor") or name.endswith("fuser.attn.processor"):
                # Keep the same attn_processors for self-attention (no hooks for self-attention)
                attn_procs[name] = unet.attn_processors[name]
                continue

            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            attn_procs[name] = AttnProcessorWithHook(
                attn_processor_key=name,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                hook=self.attn_hook,
                fast_attn=True,
                # Not enabled by default
                enabled=False,
            )

        unet.set_attn_processor(attn_procs)

    def enable_fuser(self, enabled=True):
        for module in self.unet.modules():
            if isinstance(module, GatedSelfAttentionDense):
                module.enabled = enabled

    def enable_attn_hook(self, enabled=True):
        for module in self.unet.attn_processors.values():
            if isinstance(module, AttnProcessorWithHook):
                module.enabled = enabled

    def get_token_map(self, prompt, padding="do_not_pad", verbose=False):
        """Get a list of mapping: prompt index to str (prompt in a list of token str)"""
        fg_prompt_tokens = self.tokenizer([prompt], padding=padding, max_length=77, return_tensors="np")
        input_ids = fg_prompt_tokens["input_ids"][0]

        token_map = []
        for ind, item in enumerate(input_ids.tolist()):
            token = self.tokenizer._convert_id_to_token(item)

            if verbose:
                logger.info(f"{ind}, {token} ({item})")

            token_map.append(token)

        return token_map

    def get_phrase_indices(
        self,
        prompt,
        phrases,
        token_map=None,
        add_suffix_if_not_found=False,
        verbose=False,
    ):
        for obj in phrases:
            # Suffix the prompt with object name for attention guidance if object is not in the prompt, using "|" to separate the prompt and the suffix
            if obj not in prompt:
                prompt += "| " + obj

        if token_map is None:
            # We allow using a pre-computed token map.
            token_map = self.get_token_map(prompt=prompt, padding="do_not_pad", verbose=verbose)
        token_map_str = " ".join(token_map)

        phrase_indices = []

        for obj in phrases:
            phrase_token_map = self.get_token_map(prompt=obj, padding="do_not_pad", verbose=verbose)
            # Remove <bos> and <eos> in substr
            phrase_token_map = phrase_token_map[1:-1]
            phrase_token_map_len = len(phrase_token_map)
            phrase_token_map_str = " ".join(phrase_token_map)

            if verbose:
                logger.info(
                    "Full str:",
                    token_map_str,
                    "Substr:",
                    phrase_token_map_str,
                    "Phrase:",
                    phrases,
                )

            # Count the number of token before substr
            # The substring comes with a trailing space that needs to be removed by minus one in the index.
            obj_first_index = len(token_map_str[: token_map_str.index(phrase_token_map_str) - 1].split(" "))

            obj_position = list(range(obj_first_index, obj_first_index + phrase_token_map_len))
            phrase_indices.append(obj_position)

        if add_suffix_if_not_found:
            return phrase_indices, prompt

        return phrase_indices

    def add_ca_loss_per_attn_map_to_loss(
        self,
        loss,
        attn_map,
        object_number,
        bboxes,
        phrase_indices,
        fg_top_p=0.2,
        bg_top_p=0.2,
        fg_weight=1.0,
        bg_weight=1.0,
    ):
        # b is the number of heads, not batch
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W), device="cuda")
            obj_boxes = bboxes[obj_idx]

            # We support two level (one box per phrase) and three level (multiple boxes per phrase)
            if not isinstance(obj_boxes[0], Iterable):
                obj_boxes = [obj_boxes]

            for obj_box in obj_boxes:
                # x_min, y_min, x_max, y_max = int(obj_box[0] * W), int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                x_min, y_min, x_max, y_max = scale_proportion(obj_box, H=H, W=W)
                mask[y_min:y_max, x_min:x_max] = 1

            for obj_position in phrase_indices[obj_idx]:
                # Could potentially optimize to compute this for loop in batch.
                # Could crop the ref cross attention before saving to save memory.

                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

                # shape: (b, H * W)
                ca_map_obj = attn_map[:, :, obj_position]  # .reshape(b, H, W)
                k_fg = (mask.sum() * fg_top_p).long().clamp_(min=1)
                k_bg = ((1 - mask).sum() * bg_top_p).long().clamp_(min=1)

                mask_1d = mask.view(1, -1)

                # Max-based loss function

                # Take the topk over spatial dimension, and then take the sum over heads dim
                # The mean is over k_fg and k_bg dimension, so we don't need to sum and divide on our own.
                obj_loss += (1 - (ca_map_obj * mask_1d).topk(k=k_fg).values.mean(dim=1)).sum(dim=0) * fg_weight
                obj_loss += ((ca_map_obj * (1 - mask_1d)).topk(k=k_bg).values.mean(dim=1)).sum(dim=0) * bg_weight

            loss += obj_loss / len(phrase_indices[obj_idx])

        return loss

    def compute_ca_loss(
        self,
        saved_attn,
        bboxes,
        phrase_indices,
        guidance_attn_keys,
        verbose=False,
        **kwargs,
    ):
        """
        The `saved_attn` is supposed to be passed to `save_attn_to_dict` in `cross_attention_kwargs` prior to computing ths loss.
        `AttnProcessor` will put attention maps into the `save_attn_to_dict`.

        `index` is the timestep.
        `ref_ca_word_token_only`: This has precedence over `ref_ca_last_token_only` (i.e., if both are enabled, we take the token from word rather than the last token).
        `ref_ca_last_token_only`: `ref_ca_saved_attn` comes from the attention map of the last token of the phrase in single object generation, so we apply it only to the last token of the phrase in overall generation if this is set to True. If set to False, `ref_ca_saved_attn` will be applied to all the text tokens.
        """
        loss = torch.tensor(0).float().cuda()
        object_number = len(bboxes)
        if object_number == 0:
            return loss

        for attn_key in guidance_attn_keys:
            # We only have 1 cross attention for mid.

            attn_map_integrated = saved_attn[attn_key]
            if not attn_map_integrated.is_cuda:
                attn_map_integrated = attn_map_integrated.cuda()
            # Example dimension: [20, 64, 77]
            attn_map = attn_map_integrated.squeeze(dim=0)

            loss = self.add_ca_loss_per_attn_map_to_loss(
                loss, attn_map, object_number, bboxes, phrase_indices, **kwargs
            )

        num_attn = len(guidance_attn_keys)

        if num_attn > 0:
            loss = loss / (object_number * num_attn)

        return loss

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        gligen_scheduled_sampling_beta: float = 0.3,
        phrases: List[str] = None,
        boxes: List[List[float]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        lmd_guidance_kwargs: Optional[Dict[str, Any]] = {},
        phrase_indices: Optional[List[int]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            phrases (`List[str]`):
                The phrases to guide what to include in each of the regions defined by the corresponding
                `boxes`. There should only be one phrase per bounding box.
            boxes (`List[List[float]]`):
                The bounding boxes that identify rectangular regions of the image that are going to be filled with the
                content described by the corresponding `phrases`. Each rectangular box is defined as a
                `List[float]` of 4 elements `[xmin, ymin, xmax, ymax]` where each value is between [0,1].
            gligen_scheduled_sampling_beta (`float`, defaults to 0.3):
                Scheduled Sampling factor from [GLIGEN: Open-Set Grounded Text-to-Image
                Generation](https://arxiv.org/pdf/2301.07093.pdf). Scheduled Sampling factor is only varied for
                scheduled sampling during inference for improved quality and controllability.
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
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
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
            lmd_guidance_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to `latent_lmd_guidance` function. Useful keys include `loss_scale` (the guidance strength), `loss_threshold` (when loss is lower than this value, the guidance is not applied anymore), `max_iter` (the number of iterations of guidance for each step), and `guidance_timesteps` (the number of diffusion timesteps to apply guidance on). See `latent_lmd_guidance` for implementation details.
            phrase_indices (`list` of `list`, *optional*): The indices of the tokens of each phrase in the overall prompt. If omitted, the pipeline will match the first token subsequence. The pipeline will append the missing phrases to the end of the prompt by default.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            phrases,
            boxes,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            phrase_indices,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            if phrase_indices is None:
                phrase_indices, prompt = self.get_phrase_indices(prompt, phrases, add_suffix_if_not_found=True)
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            if phrase_indices is None:
                phrase_indices = []
                prompt_parsed = []
                for prompt_item in prompt:
                    (
                        phrase_indices_parsed_item,
                        prompt_parsed_item,
                    ) = self.get_phrase_indices(prompt_item, add_suffix_if_not_found=True)
                    phrase_indices.append(phrase_indices_parsed_item)
                    prompt_parsed.append(prompt_parsed_item)
                prompt = prompt_parsed
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )

        cond_prompt_embeds = prompt_embeds

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

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

        # 5.1 Prepare GLIGEN variables
        max_objs = 30
        if len(boxes) > max_objs:
            warnings.warn(
                f"More that {max_objs} objects found. Only first {max_objs} objects will be processed.",
                FutureWarning,
            )
            phrases = phrases[:max_objs]
            boxes = boxes[:max_objs]

        n_objs = len(boxes)
        if n_objs:
            # prepare batched input to the PositionNet (boxes, phrases, mask)
            # Get tokens for phrases from pre-trained CLIPTokenizer
            tokenizer_inputs = self.tokenizer(phrases, padding=True, return_tensors="pt").to(device)
            # For the token, we use the same pre-trained text encoder
            # to obtain its text feature
            _text_embeddings = self.text_encoder(**tokenizer_inputs).pooler_output

        # For each entity, described in phrases, is denoted with a bounding box,
        # we represent the location information as (xmin,ymin,xmax,ymax)
        cond_boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder.dtype)
        if n_objs:
            cond_boxes[:n_objs] = torch.tensor(boxes)
        text_embeddings = torch.zeros(
            max_objs,
            self.unet.config.cross_attention_dim,
            device=device,
            dtype=self.text_encoder.dtype,
        )
        if n_objs:
            text_embeddings[:n_objs] = _text_embeddings
        # Generate a mask for each object that is entity described by phrases
        masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
        masks[:n_objs] = 1

        repeat_batch = batch_size * num_images_per_prompt
        cond_boxes = cond_boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        text_embeddings = text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        masks = masks.unsqueeze(0).expand(repeat_batch, -1).clone()
        if do_classifier_free_guidance:
            repeat_batch = repeat_batch * 2
            cond_boxes = torch.cat([cond_boxes] * 2)
            text_embeddings = torch.cat([text_embeddings] * 2)
            masks = torch.cat([masks] * 2)
            masks[: repeat_batch // 2] = 0
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        cross_attention_kwargs["gligen"] = {
            "boxes": cond_boxes,
            "positive_embeddings": text_embeddings,
            "masks": masks,
        }

        num_grounding_steps = int(gligen_scheduled_sampling_beta * len(timesteps))
        self.enable_fuser(True)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        loss_attn = torch.tensor(10000.0)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Scheduled sampling
                if i == num_grounding_steps:
                    self.enable_fuser(False)

                if latents.shape[1] != 4:
                    latents = torch.randn_like(latents[:, :4])

                # 7.1 Perform LMD guidance
                if boxes:
                    latents, loss_attn = self.latent_lmd_guidance(
                        cond_prompt_embeds,
                        index=i,
                        boxes=boxes,
                        phrase_indices=phrase_indices,
                        t=t,
                        latents=latents,
                        loss=loss_attn,
                        **lmd_guidance_kwargs,
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.set_grad_enabled(True)
    def latent_lmd_guidance(
        self,
        cond_embeddings,
        index,
        boxes,
        phrase_indices,
        t,
        latents,
        loss,
        *,
        loss_scale=20,
        loss_threshold=5.0,
        max_iter=[3] * 5 + [2] * 5 + [1] * 5,
        guidance_timesteps=15,
        cross_attention_kwargs=None,
        guidance_attn_keys=DEFAULT_GUIDANCE_ATTN_KEYS,
        verbose=False,
        clear_cache=False,
        unet_additional_kwargs={},
        guidance_callback=None,
        **kwargs,
    ):
        scheduler, unet = self.scheduler, self.unet

        iteration = 0

        if index < guidance_timesteps:
            if isinstance(max_iter, list):
                max_iter = max_iter[index]

            if verbose:
                logger.info(
                    f"time index {index}, loss: {loss.item()/loss_scale:.3f} (de-scaled with scale {loss_scale:.1f}), loss threshold: {loss_threshold:.3f}"
                )

            try:
                self.enable_attn_hook(enabled=True)

                while (
                    loss.item() / loss_scale > loss_threshold and iteration < max_iter and index < guidance_timesteps
                ):
                    self._saved_attn = {}

                    latents.requires_grad_(True)
                    latent_model_input = latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=cond_embeddings,
                        cross_attention_kwargs=cross_attention_kwargs,
                        **unet_additional_kwargs,
                    )

                    # update latents with guidance
                    loss = (
                        self.compute_ca_loss(
                            saved_attn=self._saved_attn,
                            bboxes=boxes,
                            phrase_indices=phrase_indices,
                            guidance_attn_keys=guidance_attn_keys,
                            verbose=verbose,
                            **kwargs,
                        )
                        * loss_scale
                    )

                    if torch.isnan(loss):
                        raise RuntimeError("**Loss is NaN**")

                    # This callback allows visualizations.
                    if guidance_callback is not None:
                        guidance_callback(self, latents, loss, iteration, index)

                    self._saved_attn = None

                    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

                    latents.requires_grad_(False)

                    # Scaling with classifier guidance
                    alpha_prod_t = scheduler.alphas_cumprod[t]
                    # Classifier guidance: https://arxiv.org/pdf/2105.05233.pdf
                    # DDIM: https://arxiv.org/pdf/2010.02502.pdf
                    scale = (1 - alpha_prod_t) ** (0.5)
                    latents = latents - scale * grad_cond

                    iteration += 1

                    if clear_cache:
                        gc.collect()
                        torch.cuda.empty_cache()

                    if verbose:
                        logger.info(
                            f"time index {index}, loss: {loss.item()/loss_scale:.3f}, loss threshold: {loss_threshold:.3f}, iteration: {iteration}"
                        )

            finally:
                self.enable_attn_hook(enabled=False)

        return latents, loss

    # Below are methods copied from StableDiffusionPipeline
    # The design choice of not inheriting from StableDiffusionPipeline is discussed here: https://github.com/huggingface/diffusers/pull/5993#issuecomment-1834258517

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
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
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
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
            prompt_embeds (`torch.Tensor`, *optional*):
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
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
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
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
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

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
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
            `torch.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
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
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.guidance_scale
    def guidance_scale(self):
        return self._guidance_scale

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.guidance_rescale
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.clip_skip
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.do_classifier_free_guidance
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.cross_attention_kwargs
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.num_timesteps
    def num_timesteps(self):
        return self._num_timesteps
