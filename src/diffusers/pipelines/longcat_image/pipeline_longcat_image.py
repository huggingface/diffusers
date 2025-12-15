# Copyright 2025 MeiTuan LongCat-Image Team and The HuggingFace Team. All rights reserved.
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
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import LongCatImageTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from .pipeline_output import LongCatImagePipelineOutput
from .system_messages import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH


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
        >>> from diffusers import LongCatImagePipeline

        >>> pipe = LongCatImagePipeline.from_pretrained("meituan-longcat/LongCat-Image", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。镜头采用中距离视角，突出她的神态和服饰的细节。光线柔和地打在她的脸上，强调她的五官和饰品的质感，增加画面的层次感与亲和力。整个画面构图简洁，砖墙的纹理与阳光的光影效果相得益彰，突显出人物的优雅与从容。"
        >>> image = pipe(
        ...     prompt,
        ...     height=768,
        ...     width=1344,
        ...     num_inference_steps=50,
        ...     guidance_scale=4.5,
        ...     generator=torch.Generator("cpu").manual_seed(43),
        ...     enable_cfg_renorm=True,
        ... ).images[0]
        >>> image.save("longcat_image.png")
        ```
"""


def get_prompt_language(prompt):
    pattern = re.compile(r"[\u4e00-\u9fff]")
    if bool(pattern.search(prompt)):
        return "zh"
    return "en"


def split_quotation(prompt, quote_pairs=None):
    """
    Implement a regex-based string splitting algorithm that identifies delimiters defined by single or double quote
    pairs. Examples::
        >>> prompt_en = "Please write 'Hello' on the blackboard for me." >>> print(split_quotation(prompt_en)) >>> #
        output: [('Please write ', False), ("'Hello'", True), (' on the blackboard for me.', False)]
    """
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal_quote_pattern = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal_quote = []

    for i, word_src in enumerate(set(matches_word_internal_quote_pattern)):
        word_tgt = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal_quote.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
    pattern = "|".join([re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2) for q1, q2 in quote_pairs])
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal_quote:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


def prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=None, height=None, width=None):
    if type == "text":
        assert num_token
        if height or width:
            print('Warning: The parameters of height and width will be ignored in "text" type.')
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif type == "image":
        assert height and width
        if num_token:
            print('Warning: The parameter of num_token will be ignored in "image" type.')
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(f'Unknow type {type}, only support "text" or "image".')
    return pos_ids


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


class LongCatImagePipeline(DiffusionPipeline, FromSingleFileMixin):
    r"""
    The pipeline for text-to-image generation.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        text_processor: Qwen2VLProcessor,
        transformer: LongCatImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_processor=text_processor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.prompt_template_encode_prefix = "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.default_sample_size = 128
        self.tokenizer_max_length = 512

    def rewire_prompt(self, prompt, device):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        all_text = []
        for each_prompt in prompt:
            language = get_prompt_language(each_prompt)
            if language == "zh":
                question = SYSTEM_PROMPT_ZH + f"\n用户输入为：{each_prompt}\n改写后的prompt为："
            else:
                question = SYSTEM_PROMPT_EN + f"\nUser Input: {each_prompt}\nRewritten prompt:"
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                }
            ]
            # Preparation for inference
            text = self.text_processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            all_text.append(text)

        inputs = self.text_processor(text=all_text, padding=True, return_tensors="pt").to(device)

        self.text_encoder.to(device)
        generated_ids = self.text_encoder.generate(**inputs, max_new_tokens=self.tokenizer_max_length)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.text_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rewrite_prompt = output_text
        return rewrite_prompt

    def _encode_prompt(self, prompt: List[str]):
        batch_all_tokens = []

        for each_prompt in prompt:
            all_tokens = []
            for clean_prompt_sub, matched in split_quotation(each_prompt):
                if matched:
                    for sub_word in clean_prompt_sub:
                        tokens = self.tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                        all_tokens.extend(tokens)
                else:
                    tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)

            if len(all_tokens) > self.tokenizer_max_length:
                logger.warning(
                    "Your input was truncated because `max_sequence_length` is set to "
                    f" {self.tokenizer_max_length} input token nums : {len(all_tokens)}"
                )
                all_tokens = all_tokens[: self.tokenizer_max_length]
            batch_all_tokens.append(all_tokens)

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_all_tokens},
            max_length=self.tokenizer_max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        prefix_tokens = self.tokenizer(self.prompt_template_encode_prefix, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_tokens)
        suffix_len = len(suffix_tokens)

        prefix_tokens_mask = torch.tensor([1] * len(prefix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)
        suffix_tokens_mask = torch.tensor([1] * len(suffix_tokens), dtype=text_tokens_and_mask.attention_mask[0].dtype)

        prefix_tokens = torch.tensor(prefix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)
        suffix_tokens = torch.tensor(suffix_tokens, dtype=text_tokens_and_mask.input_ids.dtype)

        batch_size = text_tokens_and_mask.input_ids.size(0)

        prefix_tokens_batch = prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        suffix_tokens_batch = suffix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_mask_batch = prefix_tokens_mask.unsqueeze(0).expand(batch_size, -1)
        suffix_mask_batch = suffix_tokens_mask.unsqueeze(0).expand(batch_size, -1)

        input_ids = torch.cat((prefix_tokens_batch, text_tokens_and_mask.input_ids, suffix_tokens_batch), dim=-1)
        attention_mask = torch.cat((prefix_mask_batch, text_tokens_and_mask.attention_mask, suffix_mask_batch), dim=-1)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # [max_sequence_length, batch, hidden_size] -> [batch, max_sequence_length, hidden_size]
        # clone to have a contiguous tensor
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is None:
            prompt_embeds = self._encode_prompt(prompt)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_embeds.shape[1]).to(
            self.device
        )
        return prompt_embeds.to(self.device), text_ids

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

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
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(self.tokenizer_max_length, self.tokenizer_max_length),
            height=height // 2,
            width=width // 2,
        ).to(device)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device)
        latents = latents.to(dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def check_inputs(
        self, prompt, height, width, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
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

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        enable_cfg_renorm: Optional[bool] = True,
        cfg_renorm_min: Optional[float] = 0.0,
        enable_prompt_rewrite: Optional[bool] = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            enable_cfg_renorm: Whether to enable cfg_renorm. Enabling cfg_renorm will improve image quality,
                but it may lead to a decrease in the stability of some image outputs..
            cfg_renorm_min: The minimum value of the cfg_renorm_scale range (0-1).
                cfg_renorm_min = 1.0, renorm has no effect, while cfg_renorm_min=0.0, the renorm range is larger.
            enable_prompt_rewrite: whether to enable prompt rewrite.
        Examples:

        Returns:
            [`~pipelines.LongCatImagePipelineOutput`] or `tuple`: [`~pipelines.LongCatImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        if enable_prompt_rewrite:
            prompt = self.rewire_prompt(prompt, device)
            logger.info(f"Rewrite prompt {prompt}!")

        negative_prompt = "" if negative_prompt is None else negative_prompt
        (prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt, prompt_embeds=prompt_embeds, num_images_per_prompt=num_images_per_prompt
        )
        if self.do_classifier_free_guidance:
            (negative_prompt_embeds, negative_text_ids) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 4. Prepare latent variables
        num_channels_latents = 16
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
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

        # handle guidance
        guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred_text = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if enable_cfg_renorm:
                        cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
                        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                        noise_pred = noise_pred * scale
                else:
                    noise_pred = noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            if latents.dtype != self.vae.dtype:
                latents = latents.to(dtype=self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return LongCatImagePipelineOutput(images=image)
