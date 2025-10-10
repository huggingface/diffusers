# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
from transformers import Qwen2TokenizerFast, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor, CLIPTextModel, CLIPTokenizer
import torchvision
from torchvision.transforms import ToPILImage

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import KandinskyLoraLoaderMixin
from ...models import AutoencoderKLHunyuanVideo
from ...models.transformers import Kandinsky5Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import KandinskyPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
    
        ```python
        >>> import torch
        >>> from diffusers import Kandinsky5T2VPipeline, Kandinsky5Transformer3DModel
        >>> from diffusers.utils import export_to_video

        >>> pipe = Kandinsky5T2VPipeline.from_pretrained("ai-forever/Kandinsky-5.0-T2V")
        >>> pipe = pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=512,
        ...     width=768,
        ...     num_frames=25,
        ...     num_inference_steps=50,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=6)
        ```
"""

# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
from transformers import Qwen2TokenizerFast, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor, CLIPTextModel, CLIPTokenizer
import torchvision
from torchvision.transforms import ToPILImage

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import KandinskyLoraLoaderMixin
from ...models import AutoencoderKLHunyuanVideo
from ...models.transformers import Kandinsky5Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import KandinskyPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
    
        ```python
        >>> import torch
        >>> from diffusers import Kandinsky5T2VPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = Kandinsky5T2VPipeline.from_pretrained("ai-forever/Kandinsky-5.0-T2V")
        >>> pipe = pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=512,
        ...     width=768,
        ...     num_frames=25,
        ...     num_inference_steps=50,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=6)
        ```
"""


def basic_clean(text):
    if is_ftfy_available():
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


class Kandinsky5T2VPipeline(DiffusionPipeline, KandinskyLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Kandinsky 5.0.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`Kandinsky5Transformer3DModel`]):
            Conditional Transformer to denoise the encoded video latents.
        vae ([`AutoencoderKLHunyuanVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`Qwen2_5_VLForConditionalGeneration`]):
            Frozen text-encoder (Qwen2.5-VL).
        tokenizer ([`AutoProcessor`]):
            Tokenizer for Qwen2.5-VL.
        text_encoder_2 ([`CLIPTextModel`]):
            Frozen CLIP text encoder.
        tokenizer_2 ([`CLIPTokenizer`]):
            Tokenizer for CLIP.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        transformer: Kandinsky5Transformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2VLProcessor,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = vae.config.temporal_compression_ratio
        self.vae_scale_factor_spatial = vae.config.spatial_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _encode_prompt_qwen(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(p) for p in prompt]

        # Kandinsky specific prompt template
        prompt_template = "\n".join([
            "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
            "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
            "Describe the location of the video, main characters or objects and their action.",
            "Describe the dynamism of the video and presented actions.",
            "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
            "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
            "Pay attention to the order of key actions shown in the scene.<|im_end|>",
            "<|im_start|>user\n{}<|im_end|>",
        ])
        crop_start = 129
        
        full_texts = [prompt_template.format(p) for p in prompt]
        
        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_sequence_length + crop_start,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            embeds = self.text_encoder(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_hidden_states=True,
            )["hidden_states"][-1][:, crop_start:]
        
        batch_size = len(prompt)
        
        attention_mask = inputs["attention_mask"][:, crop_start:]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = torch.cat([torch.zeros_like(cu_seqlens)[:1], cu_seqlens]).to(dtype=torch.int32)

#         # duplicate for each generation per prompt
#         seq_len = embeds.shape[0] // batch_size
#         embeds = embeds.reshape(batch_size, seq_len, -1)
#         embeds = embeds.repeat(1, num_videos_per_prompt, 1)
#         embeds = embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

#         print(embeds.shape, cu_seqlens,  "ENCODE PROMPT")
        embeds = torch.cat([embeds[i].unsqueeze(dim=0).repeat(num_videos_per_prompt, 1, 1)  for i in range(batch_size)], dim=0)
    
        return embeds.to(dtype), cu_seqlens

    def _encode_prompt_clip(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(p) for p in prompt]

        inputs = self.tokenizer_2(
            prompt,
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            pooled_embed = self.text_encoder_2(**inputs)["pooler_output"]

        # duplicate for each generation per prompt
        batch_size = len(prompt)
        pooled_embed = pooled_embed.repeat(1, num_videos_per_prompt, 1)
        pooled_embed = pooled_embed.view(batch_size * num_videos_per_prompt, -1)

        return pooled_embed.to(dtype)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
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
                Number of videos that should be generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds[0].shape[0] if isinstance(prompt_embeds, (list, tuple)) else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds_qwen, prompt_cu_seqlens = self._encode_prompt_qwen(
                prompt=prompt,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )
            prompt_embeds_clip = self._encode_prompt_clip(
                prompt=prompt,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                dtype=dtype,
            )
        else:
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = prompt_embeds

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

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

            negative_prompt_embeds_qwen, negative_cu_seqlens = self._encode_prompt_qwen(
                prompt=negative_prompt,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )
            negative_prompt_embeds_clip = self._encode_prompt_clip(
                prompt=negative_prompt,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds_qwen = None
            negative_prompt_embeds_clip = None
            negative_cu_seqlens = None

        prompt_embeds_dict = {
            "text_embeds": prompt_embeds_qwen,
            "pooled_embed": prompt_embeds_clip,
        }
        negative_prompt_embeds_dict = {
            "text_embeds": negative_prompt_embeds_qwen,
            "pooled_embed": negative_prompt_embeds_clip,
        } if do_classifier_free_guidance else None

        return prompt_embeds_dict, negative_prompt_embeds_dict, prompt_cu_seqlens, negative_cu_seqlens

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
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
            num_channels_latents,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        
        if self.transformer.visual_cond:
            # For visual conditioning, concatenate with zeros and mask
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                [batch_size, num_latent_frames, int(height) // self.vae_scale_factor_spatial, int(width) // self.vae_scale_factor_spatial, 1], 
                dtype=latents.dtype, 
                device=latents.device
            )
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        return latents

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
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 25,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        scheduler_scale: float = 10.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `512`):
                The height in pixels of the generated video.
            width (`int`, defaults to `768`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `25`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in classifier-free guidance.
            scheduler_scale (`float`, defaults to `10.0`):
                Scale factor for the custom flow matching scheduler.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A torch generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`KandinskyPipelineOutput`].
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function that is called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length for text encoding.
        
        Examples:
        
        Returns:
            [`~KandinskyPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`KandinskyPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Reset embeddings dtype
        self.transformer.time_embeddings.reset_dtype()
        self.transformer.text_rope_embeddings.reset_dtype()
        self.transformer.visual_rope_embeddings.reset_dtype()

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
        self._interrupt = False

        device = self._execution_device
        dtype = self.transformer.dtype

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds[0].shape[0] if isinstance(prompt_embeds, (list, tuple)) else prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds_dict, negative_prompt_embeds_dict, prompt_cu_seqlens, negative_cu_seqlens = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = 16
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare rope positions
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        visual_rope_pos = [
            torch.arange(num_latent_frames, device=device),
            torch.arange(height // self.vae_scale_factor_spatial // 2, device=device),
            torch.arange(width // self.vae_scale_factor_spatial // 2, device=device),
        ]
        
        text_rope_pos = torch.arange(prompt_cu_seqlens.diff().max().item(), device=device)
        
        negative_text_rope_pos = (
            torch.arange(negative_cu_seqlens.diff().max().item(), device=device)
            if negative_cu_seqlens is not None
            else None
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.unsqueeze(0).repeat(batch_size * num_videos_per_prompt)

                # Predict noise residual                
                # print(
                #     latents.shape, 
                #     prompt_embeds_dict["text_embeds"].shape, 
                #     prompt_embeds_dict["pooled_embed"].shape, 
                #     timestep.shape, 
                #     [el.shape for el in visual_rope_pos], 
                #     text_rope_pos.shape,
                #     prompt_cu_seqlens,
                # )
                
                pred_velocity = self.transformer(
                    hidden_states=latents.to(dtype),
                    encoder_hidden_states=prompt_embeds_dict["text_embeds"].to(dtype),
                    pooled_projections=prompt_embeds_dict["pooled_embed"].to(dtype),
                    timestep=timestep.to(dtype),
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=(1, 2, 2), 
                    sparse_params=None,
                    return_dict=True
                ).sample

                if self.do_classifier_free_guidance and negative_prompt_embeds_dict is not None:
                    uncond_pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype),
                        encoder_hidden_states=negative_prompt_embeds_dict["text_embeds"].to(dtype),
                        pooled_projections=negative_prompt_embeds_dict["pooled_embed"].to(dtype),
                        timestep=timestep.to(dtype),
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=negative_text_rope_pos,
                        scale_factor=(1, 2, 2),
                        sparse_params=None,
                        return_dict=True
                    ).sample

                    pred_velocity = uncond_pred_velocity + guidance_scale * (
                        pred_velocity - uncond_pred_velocity
                    )
                
                # Compute previous sample
                latents[:, :, :, :, :16] = self.scheduler.step(
                    pred_velocity, t, latents[:, :, :, :, :16], return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_dict = callback_outputs.pop("prompt_embeds", prompt_embeds_dict)
                    negative_prompt_embeds_dict = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds_dict)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. Post-processing
        latents = latents[:, :, :, :, :16]

        # 9. Decode latents to video
        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            # Reshape and normalize latents
            video = latents.reshape(
                batch_size,
                num_videos_per_prompt,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                16,
            )
            video = video.permute(0, 1, 5, 2, 3, 4)  # [batch, num_videos, channels, frames, height, width]
            video = video.reshape(
                batch_size * num_videos_per_prompt, 
                16, 
                (num_frames - 1) // self.vae_scale_factor_temporal + 1, 
                height // self.vae_scale_factor_spatial, 
                width // self.vae_scale_factor_spatial
            )
            
            # Normalize and decode
            video = video / self.vae.config.scaling_factor
            video = self.vae.decode(video).sample
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return KandinskyPipelineOutput(frames=video)
