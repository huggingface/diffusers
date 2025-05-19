# Copyright 2025 The SkyReels-V2 Team, The Wan Team and The HuggingFace Team. All rights reserved.
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
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import ftfy
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, SkyReelsV2Transformer3DModel
from ...schedulers import FlowMatchUniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import SkyReelsV2PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """\
    Examples:
        ```py
        >>> import torch
        >>> import PIL.Image
        >>> from diffusers import SkyReelsV2DiffusionForcingPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> # Load the pipeline
        >>> pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        ...     "HF_placeholder/SkyReels-V2-DF-1.3B-540P", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # TODO
        ```
"""


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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class SkyReelsV2DiffusionForcingPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """
    Pipeline for Text-to-Video (t2v) generation using SkyReels-V2 with diffusion forcing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a specific device, etc.).

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`UMT5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`SkyReelsV2Transformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchUniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: SkyReelsV2Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchUniPCMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
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

        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
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
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

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
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        causal_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // causal_block_size
        base_num_frames_block = base_num_frames // causal_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            if ar_step < min_ar_step:
                raise ValueError(f"ar_step should be at least {math.ceil(min_ar_step)} in your setting")
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, causal_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // causal_block_size] = num_iterations

        while not torch.all(pre_row >= (num_iterations - 1)):
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if causal_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, causal_block_size).flatten(1).contiguous()
            valid_interval = [(s * causal_block_size, e * causal_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
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
        max_sequence_length: int = 512,
        overlap_history: Optional[int] = 17,
        shift: float = 1.0,  # TODO: check this
        addnoise_condition: float = 20.0,
        base_num_frames: int = 97,
        ar_step: int = 5,
        causal_block_size: Optional[int] = 5,
        fps: int = 24,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `97`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SkyReelsV2PipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int`, *optional*, defaults to `512`):
                The maximum sequence length of the prompt.
            shift (`float`, *optional*, defaults to `1.0`):
            overlap_history (`int`, *optional*, defaults to `17`):
                Number of frames to overlap for smooth transitions in long videos
            addnoise_condition (`float`, *optional*, defaults to `20`):
                Improves consistency in long video generation
            base_num_frames (`int`, *optional*, defaults to `97`):
                97 or 121 | Base frame count (**97 for 540P**, **121 for 720P**)
            ar_step (`int`, *optional*, defaults to `5`):
                Controls asynchronous inference (0 for synchronous mode)
            causal_block_size (`int`, *optional*, defaults to `5`):
                Recommended when using asynchronous inference (--ar_step > 0)
            fps (`int`, *optional*, defaults to `24`):

        Examples: Returns:
            [`~SkyReelsV2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`SkyReelsV2PipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
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

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        prefix_video = None
        prefix_video_latent_length = 0
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else num_latent_frames

        if causal_block_size is None:
            causal_block_size = self.transformer.config.num_frame_per_block
        fps_embeds = [fps] * prompt_embeds.shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]

        if overlap_history is None or base_num_frames is None or num_frames <= base_num_frames:
            # Short video generation
            # 4. Prepare sample schedulers and timestep matrix
            sample_schedulers = [self.scheduler]
            for _ in range(num_latent_frames - 1):
                sample_scheduler = deepcopy(self.scheduler)
                sample_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
                sample_schedulers.append(sample_scheduler)
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
            sample_schedulers_counter = [0] * num_latent_frames
            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                num_latent_frames, timesteps, base_num_frames, ar_step, prefix_video_latent_length, causal_block_size
            )

            # 5. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )

            # 6. Denoising loop
            num_warmup_steps = len(step_matrix) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(step_matrix)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep_i in enumerate(step_matrix):
                    if self.interrupt:
                        continue

                    self._current_timestep = timestep_i

                    update_mask_i = step_update_mask[i]
                    valid_interval_i = valid_interval[i]
                    valid_interval_start, valid_interval_end = valid_interval_i
                    timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                    latent_model_input = (
                        latents[:, valid_interval_start:valid_interval_end, :, :].to(transformer_dtype).clone()
                    )
                    if addnoise_condition > 0 and valid_interval_start < prefix_video_latent_length:
                        noise_factor = 0.001 * addnoise_condition
                        timestep_for_noised_condition = addnoise_condition
                        latent_model_input[:, valid_interval_start:prefix_video_latent_length] = (
                            latent_model_input[:, valid_interval_start:prefix_video_latent_length]
                            * (1.0 - noise_factor)
                            + torch.randn_like(latent_model_input[:, valid_interval_start:prefix_video_latent_length])
                            * noise_factor
                        )
                        timestep[:, valid_interval_start:prefix_video_latent_length] = timestep_for_noised_condition
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        fps=fps_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    if self.do_classifier_free_guidance:
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            fps=fps_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    for idx in range(valid_interval_start, valid_interval_end):
                        if update_mask_i[idx].item():
                            latents[:, idx] = sample_schedulers[idx].step(
                                noise_pred[:, idx - valid_interval_start],
                                timestep_i[idx],
                                latents[:, idx],
                                return_dict=False,
                                generator=generator,
                            )[0]
                            sample_schedulers_counter[idx] += 1

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, timestep_i, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(step_matrix) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            latents = latents.unsqueeze(0)
        else:
            # Long video generation
            overlap_history_frames = (overlap_history - 1) // 4 + 1
            n_iter = 1 + (num_latent_frames - base_num_frames - 1) // (base_num_frames - overlap_history_frames) + 1
            video = None
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            for i in range(n_iter):
                if video is not None:
                    prefix_video = video[:, -overlap_history:].to(prompt_embeds.device)
                    prefix_video = [self.vae.encode(prefix_video.unsqueeze(0))[0]]  # [(c, f, h, w)]
                    if prefix_video[0].shape[1] % causal_block_size != 0:
                        truncate_len = prefix_video[0].shape[1] % causal_block_size
                        logger.warning("The length of prefix video is truncated for the causal block size alignment.")
                        prefix_video[0] = prefix_video[0][:, : prefix_video[0].shape[1] - truncate_len]
                    prefix_video_latent_length = prefix_video[0].shape[1]
                    finished_frame_num = i * (base_num_frames - overlap_history_frames) + overlap_history_frames
                    left_frame_num = num_latent_frames - finished_frame_num
                    base_num_frames_iter = min(left_frame_num + overlap_history_frames, base_num_frames)
                    if ar_step > 0:
                        num_steps = (
                            num_inference_steps
                            + ((base_num_frames_iter - overlap_history_frames) // causal_block_size - 1) * ar_step
                        )
                        self.transformer.config.num_steps = num_steps
                else:
                    base_num_frames_iter = base_num_frames

                # 4. Prepare sample schedulers and timestep matrix
                sample_schedulers = [deepcopy(self.scheduler)]
                for _ in range(base_num_frames_iter - 1):
                    sample_scheduler = deepcopy(self.scheduler)
                    sample_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
                    sample_schedulers.append(sample_scheduler)
                self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
                timesteps = self.scheduler.timesteps
                sample_schedulers_counter = [0] * base_num_frames_iter
                step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix(
                    base_num_frames_iter,
                    timesteps,
                    base_num_frames_iter,
                    ar_step,
                    prefix_video_latent_length,
                    causal_block_size,
                )

                # 5. Prepare latent variables
                num_channels_latents = self.transformer.config.in_channels
                latents = self.prepare_latents(
                    batch_size * num_videos_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    num_frames,
                    torch.float32,
                    device,
                    generator,
                    latents,
                )
                if prefix_video is not None:
                    latents[:, :prefix_video_latent_length] = prefix_video[0].to(transformer_dtype)

                # 6. Denoising loop
                num_warmup_steps = len(step_matrix) - num_inference_steps * self.scheduler.order
                self._num_timesteps = len(step_matrix)

                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, timestep_i in enumerate(step_matrix):
                        if self.interrupt:
                            continue

                        self._current_timestep = timestep_i
                        update_mask_i = step_update_mask[i]
                        valid_interval_i = valid_interval[i]
                        valid_interval_start, valid_interval_end = valid_interval_i
                        timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
                        latent_model_input = (
                            latents[:, valid_interval_start:valid_interval_end, :, :].to(transformer_dtype).clone()
                        )
                        if addnoise_condition > 0 and valid_interval_start < prefix_video_latent_length:
                            noise_factor = 0.001 * addnoise_condition
                            timestep_for_noised_condition = addnoise_condition
                            latent_model_input[:, valid_interval_start:prefix_video_latent_length] = (
                                latent_model_input[:, valid_interval_start:prefix_video_latent_length]
                                * (1.0 - noise_factor)
                                + torch.randn_like(
                                    latent_model_input[:, valid_interval_start:prefix_video_latent_length]
                                )
                                * noise_factor
                            )
                            timestep[:, valid_interval_start:prefix_video_latent_length] = (
                                timestep_for_noised_condition
                            )

                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            fps=fps_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        if self.do_classifier_free_guidance:
                            noise_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                fps=fps_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                        for idx in range(valid_interval_start, valid_interval_end):
                            if update_mask_i[idx].item():
                                latents[:, idx] = sample_schedulers[idx].step(
                                    noise_pred[:, idx - valid_interval_start],
                                    timestep_i[idx],
                                    latents[:, idx],
                                    return_dict=False,
                                    generator=generator,
                                )[0]
                                sample_schedulers_counter[idx] += 1

                        if callback_on_step_end is not None:
                            callback_kwargs = {}
                            for k in callback_on_step_end_tensor_inputs:
                                callback_kwargs[k] = locals()[k]
                            callback_outputs = callback_on_step_end(self, i, timestep_i, callback_kwargs)

                            latents = callback_outputs.pop("latents", latents)
                            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                            negative_prompt_embeds = callback_outputs.pop(
                                "negative_prompt_embeds", negative_prompt_embeds
                            )

                        # call the callback, if provided
                        if i == len(step_matrix) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                        ):
                            progress_bar.update()

                        if XLA_AVAILABLE:
                            xm.mark_step()

                latents = latents.unsqueeze(0)
                if not output_type == "latent":
                    latents = latents.to(self.vae.dtype)
                    latents = latents / latents_std + latents_mean
                    videos = self.vae.decode(latents, return_dict=False)[0]
                    if video is None:
                        video = videos  # c, f, h, w
                    else:
                        video = torch.cat([video, videos[:, overlap_history:]], 1)  # c, f, h, w
                else:
                    video = latents

        self._current_timestep = None

        if not output_type == "latent":
            if overlap_history is None:
                latents = latents.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return SkyReelsV2PipelineOutput(frames=video)
