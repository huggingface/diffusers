# Copyright 2025 The SandAI Team and The HuggingFace Team. All rights reserved.
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

import html
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import Magi1LoraLoaderMixin
from ...models import AutoencoderKLMagi1, Magi1Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline


SPECIAL_TOKEN_PATH = os.getenv("SPECIAL_TOKEN_PATH", "example/assets/special_tokens.npz")
SPECIAL_TOKEN = np.load(SPECIAL_TOKEN_PATH)
CAPTION_TOKEN = torch.tensor(SPECIAL_TOKEN["caption_token"].astype(np.float16))
LOGO_TOKEN = torch.tensor(SPECIAL_TOKEN["logo_token"].astype(np.float16))
TRANS_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][:1].astype(np.float16))
HQ_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][1:2].astype(np.float16))
STATIC_FIRST_FRAMES_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][2:3].astype(np.float16))  # static first frames
DYNAMIC_FIRST_FRAMES_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][3:4].astype(np.float16))  # dynamic first frames
BORDERNESS_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][4:5].astype(np.float16))
DURATION_TOKEN_LIST = [torch.tensor(SPECIAL_TOKEN["other_tokens"][i : i + 1].astype(np.float16)) for i in range(0 + 7, 8 + 7)]
THREE_D_MODEL_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][15:16].astype(np.float16))
TWO_D_ANIME_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][16:17].astype(np.float16))



SPECIAL_TOKEN_DICT = {
    "CAPTION_TOKEN": CAPTION_TOKEN,
    "LOGO_TOKEN": LOGO_TOKEN,
    "TRANS_TOKEN": TRANS_TOKEN,
    "HQ_TOKEN": HQ_TOKEN,
    "STATIC_FIRST_FRAMES_TOKEN": STATIC_FIRST_FRAMES_TOKEN,
    "DYNAMIC_FIRST_FRAMES_TOKEN": DYNAMIC_FIRST_FRAMES_TOKEN,
    "BORDERNESS_TOKEN": BORDERNESS_TOKEN,
    "THREE_D_MODEL_TOKEN": THREE_D_MODEL_TOKEN,
    "TWO_D_ANIME_TOKEN": TWO_D_ANIME_TOKEN,
}

def _pad_special_token(special_token: torch.Tensor, txt_feat: torch.Tensor, attn_mask: torch.Tensor = None):
    N, C, _, D = txt_feat.size()
    txt_feat = torch.cat(
        [special_token.unsqueeze(0).unsqueeze(0).to(txt_feat.device).to(txt_feat.dtype).expand(N, C, -1, D), txt_feat], dim=2
    )[:, :, :800, :]
    if attn_mask is not None:
        attn_mask = torch.cat([torch.ones(N, C, 1, dtype=attn_mask.dtype, device=attn_mask.device), attn_mask], dim=-1)[:, :, :800]
    return txt_feat, attn_mask



def pad_special_token(special_token_keys: List[str], caption_embs: torch.Tensor, emb_masks: torch.Tensor):
    device = caption_embs.device
    for special_token_key in special_token_keys:
        if special_token_key == "DURATION_TOKEN":
            new_caption_embs, new_emb_masks = [], []
            num_chunks = caption_embs.size(1)
            for i in range(num_chunks):
                chunk_caption_embs, chunk_emb_masks = _pad_special_token(
                    DURATION_TOKEN_LIST[min(num_chunks - i - 1, 7)].to(device),
                    caption_embs[:, i : i + 1],
                    emb_masks[:, i : i + 1],
                )
                new_caption_embs.append(chunk_caption_embs)
                new_emb_masks.append(chunk_emb_masks)
            caption_embs = torch.cat(new_caption_embs, dim=1)
            emb_masks = torch.cat(new_emb_masks, dim=1)
        else:
            special_token = SPECIAL_TOKEN_DICT.get(special_token_key)
            if special_token is not None:
                caption_embs, emb_masks = _pad_special_token(special_token.to(device), caption_embs, emb_masks)
    return caption_embs, emb_masks

def get_special_token_keys(
    use_static_first_frames_token: bool,
    use_dynamic_first_frames_token: bool,
    use_borderness_token: bool,
    use_hq_token: bool,
    use_3d_model_token: bool,
    use_2d_anime_token: bool,
    use_duration_token: bool,
):
    special_token_keys = []
    if use_static_first_frames_token:
        special_token_keys.append("STATIC_FIRST_FRAMES_TOKEN")
    if use_dynamic_first_frames_token:
        special_token_keys.append("DYNAMIC_FIRST_FRAMES_TOKEN")
    if use_borderness_token:
        special_token_keys.append("BORDERNESS_TOKEN")
    if use_hq_token:
        special_token_keys.append("HQ_TOKEN")
    if use_3d_model_token:
        special_token_keys.append("THREE_D_MODEL_TOKEN")
    if use_2d_anime_token:
        special_token_keys.append("TWO_D_ANIME_TOKEN")
    if use_duration_token:
        special_token_keys.append("DURATION_TOKEN")
    return special_token_keys

def get_negative_special_token_keys(
    use_negative_special_tokens: bool,
):
    if use_negative_special_tokens:
        return ["CAPTION_TOKEN", "LOGO_TOKEN", "TRANS_TOKEN", "BORDERNESS_TOKEN"]
    return []


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

# TODO: Write example_doc_string
EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        ```
"""

class Magi1Pipeline(DiffusionPipeline, Magi1LoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Magi1.

    Reference: https://github.com/SandAI-org/MAGI-1

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl) variant.
        transformer ([`Magi1Transformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        vae ([`AutoencoderKLMagi1`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A flow matching scheduler with Euler discretization, using SD3-style time resolution transform.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    #TODO: Add _callback_tensor_inputs and _optional_components?

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: T5EncoderModel,
        transformer: Magi1Transformer3DModel,
        vae: AutoencoderKLMagi1,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        # TODO: Double check if they are really read from config
        self.temporal_downscale_factor = getattr(self.vae.config, "scale_factor_temporal", 4)
        self.spatial_downscale_factor = getattr(self.vae.config, "scale_factor_spatial", 8)
        self.num_channels_latents = self.transformer.config.in_channels
        self.chunk_width = 6 # TODO: Double check this value
        self._callback_tensor_inputs = ["latents"]  # extend as needed
        # TODO: Add attributes


    def _build_text_pack(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_mask: Optional[torch.Tensor],
        num_infer_chunks: int,
        use_static_first_frames_token:  bool,
        use_dynamic_first_frames_token: bool,
        use_borderness_token: bool,
        use_hq_token: bool,
        use_3d_model_token: bool,
        use_2d_anime_token: bool,
        use_duration_token: bool,
        use_negative_special_tokens: bool,
    ):
        """
        Expand to chunk dim and prepend special tokens in MAGI order.
        """
        prompt_embeds = prompt_embeds.unsqueeze(1).repeat(1, num_infer_chunks, 1, 1)
        prompt_mask = prompt_mask.unsqueeze(1).repeat(1, num_infer_chunks, 1)
        special_token_keys = get_special_token_keys(
            use_static_first_frames_token=use_static_first_frames_token,
            use_dynamic_first_frames_token=use_dynamic_first_frames_token,
            use_borderness_token=use_borderness_token,
            use_hq_token=use_hq_token,
            use_3d_model_token=use_3d_model_token,
            use_2d_anime_token=use_2d_anime_token,
            use_duration_token=use_duration_token,
        )
        prompt_embeds, prompt_mask = pad_special_token(special_token_keys, prompt_embeds, prompt_mask)
        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                # TODO: Load negative prompt embeds, they are learned
                # null_caption_embedding = model.y_embedder.null_caption_embedding.unsqueeze(0)
                # Creating zeros for negative prompt embeds for now
                negative_prompt_embeds = torch.zeros(prompt_embeds.size(0), prompt_embeds.size(2), prompt_embeds.size(3)).to(prompt_embeds.device)
                negative_prompt_mask = torch.zeros_like(prompt_mask)
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1).repeat(1, num_infer_chunks, 1, 1)
                special_negative_token_keys = get_negative_special_token_keys(
                    use_negative_special_tokens=use_negative_special_tokens,
                )
                negative_prompt_embeds, _ = pad_special_token(special_negative_token_keys, negative_prompt_embeds, None)
                negative_token_length = 50
                negative_prompt_mask[:, :, :negative_token_length] = 1
                negative_prompt_mask[:, :, negative_token_length:] = 0
            if prompt_mask.sum() == 0:
                prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds])
                prompt_mask = torch.cat([negative_prompt_mask, negative_prompt_mask], dim=0)
            else:
                prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds])
                prompt_mask = torch.cat([prompt_mask, negative_prompt_mask], dim=0)
        return prompt_embeds, prompt_mask
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int,
        max_sequence_length: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Double check if MAGI-1 does some special handling during prompt encoding
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        # Just keep the clean function consistent with other pipelines
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # TODO: Debug if repeating mask is necessary because it's not used in any other pipeline
        # Repeat mask the same way as embeddings and keep size [B*num, L]
        mask = mask.repeat(1, num_videos_per_prompt)
        mask = mask.view(batch_size * num_videos_per_prompt, -1).to(device)
        # TODO: I think prompt_embeds are already float32, but double check
        prompt_embeds = prompt_embeds.float()
        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        negative_prompt: Optional[Union[str, List[str]]],
        num_videos_per_prompt: int,
        prompt_embeds: Optional[torch.Tensor],
        prompt_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_mask: Optional[torch.Tensor],
        max_sequence_length: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Encodes the prompt into text encoder hidden states.
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_videos_per_prompt (`int`):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_prompt_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        # TODO: Can we provide different prompts for different chunks?
        # If so, how are we gonna support that?
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt is not None:
            prompt_embeds, prompt_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        
        # Negative prompt embeddings are learned for MAGI-1
        # However, we still provide the option to pass them in
        if self.do_classifier_free_guidance:
            if negative_prompt is not None:
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                negative_prompt_embeds, negative_prompt_mask = self._get_t5_prompt_embeds(
                    prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
        return prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask

    def check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        negative_prompt: Optional[Union[str, List[str]]],
        height: int,
        width: int,
        prompt_embeds: Optional[torch.Tensor],
        prompt_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_mask: Optional[torch.Tensor],
        callback_on_step_end_tensor_inputs: List[str],
    ):
        r"""Checks the validity of the inputs."""

        # Check prompt and prompt_embeds
        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        
        # Check prompt_embeds and prompt_mask
        if prompt_embeds is not None and prompt_mask is None:
            raise ValueError("Must provide `prompt_mask` when specifying `prompt_embeds`.")
        if prompt_embeds is None and prompt_mask is not None:
            raise ValueError("Must provide `prompt_embeds` when specifying `prompt_mask`.")
        
        # Check negative_prompt and negative_prompt_embeds
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # Check negative_prompt_embeds and negative_prompt_mask
        if negative_prompt_embeds is not None and negative_prompt_mask is None:
            raise ValueError("Must provide `negative_prompt_mask` when specifying `negative_prompt_embeds`.")
        if negative_prompt_embeds is None and negative_prompt_mask is not None:
            raise ValueError("Must provide `negative_prompt_embeds` when specifying `negative_prompt_mask`.")


        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if prompt_mask is not None and negative_prompt_mask is not None:
            if prompt_mask.shape != negative_prompt_mask.shape:
                raise ValueError(
                    "`prompt_mask` and `negative_prompt_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_mask` {prompt_mask.shape} != `negative_prompt_mask`"
                    f" {negative_prompt_mask.shape}."
                )

        # Check height and width
        # TODO: Why 16?
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        # Check callback_on_step_end_tensor_inputs
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_chunks: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (
            batch_size,
            num_channels_latents,
            num_chunks * self.chunk_width,
            height // self.spatial_downscale_factor,
            width // self.spatial_downscale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # TODO: Fix default values of the parameters
    # TODO: Double-check if all parameters are needed/included
    # TODO: Double-check output type default (both in param and in docstring)
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 96,
        num_inference_steps: int = 32,
        guidance_scale: float = 7.5,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 800,
        use_static_first_frames_token: bool = False,
        use_dynamic_first_frames_token: bool = False,
        use_borderness_token: bool = False,
        use_hq_token: bool = True,
        use_3d_model_token: bool = False,
        use_2d_anime_token: bool = False,
        use_duration_token: bool = True,
        use_negative_special_tokens: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `720`):
                The height in pixels of the generated video.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `96`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `32`):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.5`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate videos that are closely linked to
                the text `prompt`, usually at the expense of lower video quality.
            num_videos_per_prompt (`int`, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            prompt_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, negative_prompt_embeds will be generated from the `negative_prompt` input argument.
            negative_prompt_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `"latent"`, `"pt"`, or `"np"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`Magi1PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `800`):
                The maximum sequence length for the text encoder. Sequences longer than this will be truncated. MAGI-1
                uses a max length of 800 tokens.

        Examples:

        Returns:
            [`~Magi1PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`Magi1PipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated videos.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            prompt_mask,
            negative_prompt_embeds,
            negative_prompt_mask,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        # TODO: Come back here later

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] # TODO: Check if linter complains here
        
        device = self._execution_device
        # 3. Encode input prompt
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.encode_prompt(
            prompt,
            negative_prompt,
            num_videos_per_prompt,
            prompt_embeds,
            prompt_mask,
            negative_prompt_embeds,
            negative_prompt_mask,
            max_sequence_length,
            device,
            self.text_encoder.dtype # TODO: double check what is passed here
        )

        num_infer_chunks = math.ceil((num_frames // self.temporal_downscale_factor) / self.chunk_width)
        prompt_embeds, prompt_mask = self._build_text_pack(
            prompt_embeds,
            prompt_mask,
            negative_prompt_embeds,
            negative_prompt_mask,
            num_infer_chunks,
            use_static_first_frames_token,
            use_dynamic_first_frames_token,
            use_borderness_token,
            use_hq_token,
            use_3d_model_token,
            use_2d_anime_token,
            use_duration_token,
            use_negative_special_tokens,
        )
        
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            self.num_channels_latents,
            height,
            width,
            num_infer_chunks,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )