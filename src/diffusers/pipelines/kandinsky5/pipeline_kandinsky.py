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
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
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

    def _encode_prompt_qwen(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

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
        
        attention_mask = inputs["attention_mask"][:, crop_start:]
        embeds = embeds[attention_mask.bool()]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = torch.cat([torch.zeros_like(cu_seqlens)[:1], cu_seqlens]).to(dtype=torch.int32)

        # duplicate for each generation per prompt
        batch_size = len(prompt)
        seq_len = embeds.shape[0] // batch_size
        embeds = embeds.reshape(batch_size, seq_len, -1)
        embeds = embeds.repeat(1, num_videos_per_prompt, 1)
        embeds = embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return embeds, cu_seqlens

    def _encode_prompt_clip(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

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

        return pooled_embed

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt_embeds, prompt_cu_seqlens = self._encode_prompt_qwen(prompt, device, num_videos_per_prompt)
        pooled_embed = self._encode_prompt_clip(prompt, device, num_videos_per_prompt)

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            
            negative_prompt_embeds, negative_cu_seqlens = self._encode_prompt_qwen(negative_prompt, device, num_videos_per_prompt)
            negative_pooled_embed = self._encode_prompt_clip(negative_prompt, device, num_videos_per_prompt)
        else:
            negative_prompt_embeds = None
            negative_pooled_embed = None
            negative_cu_seqlens = None

        text_embeds = {
            "text_embeds": prompt_embeds,
            "pooled_embed": pooled_embed,
        }
        negative_text_embeds = {
            "text_embeds": negative_prompt_embeds,
            "pooled_embed": negative_pooled_embed,
        } if do_classifier_free_guidance else None

        return text_embeds, negative_text_embeds, prompt_cu_seqlens, negative_cu_seqlens

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        visual_cond: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            num_latent_frames = latents.shape[1]
            latents = latents.to(device=device, dtype=dtype)

        else:
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
                            
        if visual_cond:
            # For visual conditioning, concatenate with zeros and mask
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                [batch_size, num_latent_frames, int(height) // self.vae_scale_factor_spatial, int(width) // self.vae_scale_factor_spatial, 1], 
                dtype=latents.dtype, 
                device=latents.device
            )
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        return latents


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 25,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        scheduler_scale: float = 10.0,
        num_videos_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation.
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
            generator (`torch.Generator`, *optional*):
                A torch generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`KandinskyPipelineOutput`].
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step.
        
        Examples:
        
        Returns:
            [`~KandinskyPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`KandinskyPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        text_embeds, negative_text_embeds, prompt_cu_seqlens, negative_cu_seqlens = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )

        num_channels_latents = 16
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=16,
            height=height,
            width=width,
            num_frames=num_frames,
            visual_cond=self.transformer.visual_cond,
            dtype=self.transformer.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        
        visual_cond = latents[:, :, :, :, 16:]

        visual_rope_pos = [
            torch.arange(num_frames // 4 + 1, device=device),
            torch.arange(height // 8 // 2, device=device),
            torch.arange(width // 8 // 2, device=device),
        ]
        
        text_rope_pos = torch.arange(prompt_cu_seqlens[-1].item(), device=device)
        
        negative_text_rope_pos = (
            torch.arange(negative_cu_seqlens[-1].item(), device=device)
            if negative_cu_seqlens is not None
            else None
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = t.unsqueeze(0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # print(latents.shape)
                    pred_velocity = self.transformer(
                        latents,
                        text_embeds["text_embeds"],
                        text_embeds["pooled_embed"],
                        timestep,
                        visual_rope_pos,
                        text_rope_pos,
                        scale_factor=(1, 2, 2), 
                        sparse_params=None,
                        return_dict=False
                    )[0]
                    
                    if guidance_scale > 1.0 and negative_text_embeds is not None:
                        uncond_pred_velocity = self.transformer(
                            latents,
                            negative_text_embeds["text_embeds"],
                            negative_text_embeds["pooled_embed"],
                            timestep,
                            visual_rope_pos,
                            negative_text_rope_pos,
                            scale_factor=(1, 2, 2),
                            sparse_params=None,
                            return_dict=False
                        )[0]

                        pred_velocity = uncond_pred_velocity + guidance_scale * (
                            pred_velocity - uncond_pred_velocity
                        )
                
                latents = self.scheduler.step(pred_velocity, t, latents[:, :, :, :, :16], return_dict=False)[0]
                latents = torch.cat([latents, visual_cond], dim=-1)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    
        latents = latents[:, :, :, :, :16]

        # 9. Decode latents to video
        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            # Reshape and normalize latents
            video = latents.reshape(
                batch_size,
                num_videos_per_prompt,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // 8,
                width // 8,
                16,
            )
            video = video.permute(0, 1, 5, 2, 3, 4)  # [batch, num_videos, channels, frames, height, width]
            video = video.reshape(batch_size * num_videos_per_prompt, 16, (num_frames - 1) // self.vae_scale_factor_temporal + 1, height // 8, width // 8)
            
            # Normalize and decode
            video = video / self.vae.config.scaling_factor
            video = self.vae.decode(video).sample
            video = ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
            # Convert to output format
            if output_type == "pil":
                if num_frames == 1:
                    # Single image
                    video = [ToPILImage()(frame.squeeze(1)) for frame in video]
                else:
                    # Video frames
                    video = [video[i] for i in range(video.shape[0])]

        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return KandinskyPipelineOutput(frames=video)
