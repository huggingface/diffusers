# Copyright 2025 Alpha-VLLM and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers import (
    PreTrainedModel,
    Gemma3PreTrainedModel,
    GemmaTokenizer,
    GemmaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast
)

from ..pipeline_utils import ImagePipelineOutput
from ..lumina2 import Lumina2Pipeline

from ...models import AutoencoderKL
from ...models.transformers.transformer_lumina2 import Lumina2Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler

from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)

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
        >>> from diffusers import NewbiePipeline
        >>> from transformers import AutoModel

        >>> device = "cuda"
        >>> model_path = "Disty0/NewBie-image-Exp0.1-Diffusers"
        >>> text_encoder_2 = AutoModel.from_pretrained(model_path, subfolder="text_encoder_2", trust_remote_code=True, torch_dtype=torch.bfloat16)
        >>> pipe = NewbiePipeline.from_pretrained(model_path, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16)
        >>> del text_encoder_2

        >>> # Enable memory optimizations.
        >>> pipe.enable_model_cpu_offload(device=device)

        >>> prompt = \"""
        >>>   <character_1>
        >>>   <n>$character_1$</n>
        >>>   <gender>1girl</gender>
        >>>   <appearance>chibi, red_eyes, blue_hair, long_hair, hair_between_eyes, head_tilt, tareme, closed_mouth</appearance>
        >>>   <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, blue_skirt, miniskirt, pleated_skirt, blue_hat, mini_hat, thighhighs, grey_thighhighs, black_shoes, mary_janes</clothing>
        >>>   <expression>happy, smile</expression>
        >>>   <action>standing, holding, holding_briefcase</action>
        >>>   <position>center_left</position>
        >>>   </character_1>

        >>>   <character_2>
        >>>   <n>$character_2$</n>
        >>>   <gender>1girl</gender>
        >>>   <appearance>chibi, red_eyes, pink_hair, long_hair, very_long_hair, multi-tied_hair, open_mouth</appearance>
        >>>   <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, red_skirt, miniskirt, pleated_skirt, hair_bow, multiple_hair_bows, white_bow, ribbon_trim, ribbon-trimmed_bow, white_thighhighs, black_shoes, mary_janes, bow_legwear, bare_arms</clothing>
        >>>   <expression>happy, smile</expression>
        >>>   <action>standing, holding, holding_briefcase, waving</action>
        >>>   <position>center_right</position>
        >>>   </character_2>

        >>>   <general_tags>
        >>>   <count>2girls, multiple_girls</count>
        >>>   <style>anime_style, digital_art</style>
        >>>   <background>white_background, simple_background</background>
        >>>   <atmosphere>cheerful</atmosphere>
        >>>   <quality>high_resolution, detailed</quality>
        >>>   <objects>briefcase</objects>
        >>>   <other>alternate_costume</other>
        >>>   </general_tags>
        >>> \"""
        >>> negative_prompt = "blurry, worst quality, low quality, deformed hands, bad anatomy, extra limbs, poorly drawn face, mutated, extra eyes, bad proportions"

        >>> pipe.text_encoder_2 = pipe.text_encoder_2.to(device)
        >>> image = pipe(
        >>>     prompt,
        >>>     negative_prompt=negative_prompt,
        >>>     height=1024,
        >>>     width=1024,
        >>>     guidance_scale=2.5,
        >>>     num_inference_steps=30,
        >>>     generator=torch.manual_seed(42),
        >>> ).images[0]
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
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


class NewbiePipeline(Lumina2Pipeline):
    r"""
    Pipeline for text-to-image generation using Lumina-T2I.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Gemma3PreTrainedModel`]):
            Frozen Gemma3 text-encoder.
        text_encoder_2 ([`PreTrainedModel`]):
            Frozen JinaCLIPTextModel text-encoder. Requires `trust_remote_code=True`.
        tokenizer (`GemmaTokenizer` or `GemmaTokenizerFast`):
            Gemma tokenizer.
        tokenizer_2 (`XLMRobertaTokenizer` or `XLMRobertaTokenizerFast`):
            XLMRoberta tokenizer.
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    model_cpu_offload_seq = "text_encoder->transformer->vae" # JinaCLIPTextModel is not compatible with cpu offload

    def __init__(
        self,
        transformer: Lumina2Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: Gemma3PreTrainedModel,
        text_encoder_2: PreTrainedModel,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        tokenizer_2: Union[XLMRobertaTokenizer, XLMRobertaTokenizerFast],
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.register_modules(
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        num_images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because JinaCLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder_2.get_text_features(input_ids=text_input_ids.to(device))

        # Use pooled output of JinaCLIPTextModel
        if isinstance(prompt_embeds, (tuple, list)) and len(prompt_embeds) == 2:
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = prompt_embeds
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                NewbieAI, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For NewbieAI, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `512`):
                Maximum sequence length to use for the prompt.
        """
        if device is None:
            device = self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if system_prompt is None:
            system_prompt = self.system_prompt
        if prompt is not None:
            prompt = [system_prompt + " <Prompt Start> " + p for p in prompt]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt if negative_prompt is not None else ""

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            if negative_prompt_embeds is None:
                negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                    prompt=negative_prompt,
                    device=device,
                    max_sequence_length=max_sequence_length,
                )

                batch_size, seq_len, _ = negative_prompt_embeds.shape
                # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
                negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
                negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                    batch_size * num_images_per_prompt, -1
                )

            # encode with clip after gemma for better offloading
            if negative_pooled_prompt_embeds is None:
                negative_pooled_prompt_embeds = self._get_clip_prompt_embeds(
                    prompt=negative_prompt,
                    device=device,
                    max_sequence_length=max_sequence_length,
                    num_images_per_prompt=num_images_per_prompt,
                )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        super().check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        negative_prompt: Union[str, List[str]] = None,
        sigmas: List[float] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        system_prompt: Optional[str] = None,
        cfg_trunc_ratio: float = 1.0,
        cfg_normalization: bool = True,
        max_sequence_length: int = 512,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            attention_kwargs:
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            system_prompt (`str`, *optional*):
                The system prompt to use for the image generation.
            cfg_trunc_ratio (`float`, *optional*, defaults to `1.0`):
                The ratio of the timestep interval to apply normalization-based guidance scale.
            cfg_normalization (`bool`, *optional*, defaults to `True`):
                Whether to apply normalization-based guidance scale.
            max_sequence_length (`int`, defaults to `512`):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            system_prompt=system_prompt,
        )

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # compute whether apply classifier-free truncation on this timestep
                do_classifier_free_truncation = (i + 1) / num_inference_steps > cfg_trunc_ratio
                # reverse the timestep since Lumina uses t=0 as the noise and t=1 as the image
                current_timestep = 1 - t / self.scheduler.config.num_train_timesteps
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latents.shape[0])

                noise_pred_cond = self.transformer(
                    hidden_states=latents,
                    timestep=current_timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0]

                # perform normalization-based guidance scale on a truncated timestep interval
                if self.do_classifier_free_guidance and not do_classifier_free_truncation:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=current_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        return_dict=False,
                        attention_kwargs=self.attention_kwargs,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    # apply normalization after classifier-free guidance
                    if cfg_normalization:
                        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_pred = noise_pred * (cond_norm / noise_norm)
                else:
                    noise_pred = noise_pred_cond

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                noise_pred = -noise_pred
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
