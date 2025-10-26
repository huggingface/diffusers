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

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import Magi1LoraLoaderMixin
from ...models import AutoencoderKLMagi1, Magi1Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline


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
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
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
        text_encoder: UMT5EncoderModel,
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

        # TODO: Add attributes

    
    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        negative_prompt: Optional[Union[str, List[str]]],
        do_classifier_free_guidance: bool,
        num_videos_per_prompt: int,
        prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        max_sequence_length: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ):
        r"""Encodes the prompt into text encoder hidden states.
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
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

        if prompt_embeds is None:
            prompt_embeds, prompt_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_mask = None

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
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
            
            negative_prompt_embeds, negative_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_mask = None
        return prompt_embeds, negative_prompt_embeds, prompt_mask, negative_mask

    def check_inputs(
        self,
        prompt: Optional[Union[str, List[str]]],
        negative_prompt: Optional[Union[str, List[str]]],
        height: int,
        width: int,
        prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
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
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 800,
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
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, negative_prompt_embeds will be generated from the `negative_prompt` input argument.
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
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

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
        prompt_embeds, negative_prompt_embeds, prompt_mask, negative_mask = self.encode_prompt(
            prompt,
            negative_prompt,
            self.do_classifier_free_guidance,
            num_videos_per_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            max_sequence_length,
            device,
            self.text_encoder.dtype
        )