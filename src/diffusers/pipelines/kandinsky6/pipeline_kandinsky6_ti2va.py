# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

"""Kandinsky 6 TI2VA Diffusers pipeline."""

import copy
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ...utils import replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Kandinsky6TI2VAPipelineOutput


EXAMPLE_DOC_STRING = """
    Examples:

        ```python
        >>> import torch
        >>> from diffusers import Kandinsky6TI2VAPipeline
        >>> from diffusers.utils import encode_video

        >>> model_id = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
        >>> pipe = Kandinsky6TI2VAPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()

        >>> output = pipe(
        ...     prompt="A cat and a dog baking a cake together in a kitchen.",
        ...     height=480,
        ...     width=864,
        ...     num_frames=121,
        ...     num_inference_steps=16,
        ...     guidance_scale=1.0,
        ...     sample_audio=True,
        ... )

        >>> video = output.frames[0].permute(1, 2, 3, 0)
        >>> audio = torch.as_tensor(output.audio[0])[:, None].repeat(1, 2)
        >>> encode_video(
        ...     video,
        ...     fps=24,
        ...     output_path="output.mp4",
        ...     audio=audio,
        ...     audio_sample_rate=pipe.audio_sample_rate,
        ... )
        ```
"""

_PROMPT_TEMPLATE = "\n".join(
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        "Name the visual style of the video: whether it is a professional footage, user generated content, "
        "some kind of animation, video game or scren content.",
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
_QWEN_CROP_START = 129
_CLIP_MAX_LENGTH = 77
_T2VA_EXPANSION_INSTRUCTION = (
    "You are a prompt beautifier that transforms short user video+audio descriptions into rich, detailed English "
    "prompts specifically optimized for video+audio generation models. Preserve any direct speech between <S> "
    "and <E> exactly as written. If the prompt asks for speech without exact words, add suitable direct speech "
    "between those tags. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Describe the scene, "
    "actions, camera motion, visual style, and audio in detail. Make the prompt dynamic. Answer only with the "
    "expanded prompt.\n\n"
    "Rewrite Prompt: {}"
)
_I2VA_EXPANSION_INSTRUCTION = (
    "You are a prompt beautifier that transforms a short user video+audio description and a provided reference "
    "image into a rich, detailed English prompt optimized for video+audio generation. The reference image is the "
    "ground truth for the initial scene. Keep every visible fact that you mention consistent with it. Do not mainly "
    "describe the image: focus on the requested actions, motion, interactions, camera changes, speech, and audio, "
    "explaining the changes relative to the initial image. Add only details compatible with the reference image; "
    "never invent conflicting objects, identities, colors, locations, or actions. Preserve direct speech between "
    "<S> and <E> exactly. Put general audio descriptions between <AUDCAP> and <ENDAUDCAP>. Make the prompt dynamic "
    "and describe how the scene evolves from the provided image. Answer only with the expanded prompt.\n\n"
    "Rewrite Prompt: {}"
)


class Kandinsky6TI2VAPipeline(DiffusionPipeline):
    r"""Pipeline for text/image-to-video-and-audio generation with Kandinsky 6.

    This pipeline inherits the generic loading, device placement, and progress handling provided by
    [`DiffusionPipeline`]. The transformer must expose the K6 multimodal forward contract, while the VAE adapters
    must implement the portable video/audio post-processing contracts.

    Args:
        transformer ([`Kandinsky6Transformer3DModel`]):
            Multimodal K6 transformer used to denoise video and audio latents.
        vae (`nn.Module`):
            Video VAE used to encode and decode video latents.
        text_encoder:
            Qwen2.5-VL text encoder for token-level embeddings.
        audio_vae (`nn.Module`, *optional*):
            Audio VAE used to decode generated audio latents. May be `None` only when `sample_audio=False`.
        scheduler:
            Diffusion scheduler used by the denoising loop.
        tokenizer:
            Qwen2.5-VL processor.
        text_encoder_2:
            CLIP text encoder for pooled embeddings.
        tokenizer_2:
            CLIP tokenizer.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae->audio_vae"
    _optional_components = ["audio_vae"]
    _callback_tensor_inputs = ["latents"]
    _DEFAULT_NEGATIVE_PROMPT = (
        "Static, 2D cartoon, cartoon, 2d animation, paintings, images, "
        "worst quality, low quality, ugly, deformed, walking backwards"
    )

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: Any,
        audio_vae: nn.Module | None,
        scheduler: Any,
        tokenizer: Any,
        text_encoder_2: Any,
        tokenizer_2: Any,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            audio_vae=audio_vae,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )
        scale_factor = (
            self.transformer.config.get("scale_factor", (1.0, 2.0, 2.0))
            if getattr(self, "transformer", None) is not None
            else (1.0, 2.0, 2.0)
        )
        self.scale_factor = tuple(float(value) for value in scale_factor)

        audio_config = self.audio_vae.config if getattr(self, "audio_vae", None) is not None else {}
        self.audio_sample_rate = int(audio_config.get("sample_rate", 44_100))
        self.audio_downsample_factor = int(audio_config.get("downsample_factor", 1_024))

    @staticmethod
    def _value_batch_size(value: Any) -> int:
        if value is None:
            return 1
        if isinstance(value, list):
            return len(value)
        if isinstance(value, Tensor):
            return int(value.shape[0]) if value.ndim in (3, 5) else 1
        return 1

    @classmethod
    def _batch_size(
        cls,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        image: str | object | list[str | object] | None,
        latents: Tensor | None,
        audio_latents: Tensor | None,
        prompt_embeds: Tensor | None,
        negative_prompt_embeds: Tensor | None,
    ) -> int:
        sizes = [
            cls._value_batch_size(value)
            for value in (
                prompt,
                negative_prompt,
                image,
                latents,
                audio_latents,
                prompt_embeds,
                negative_prompt_embeds,
            )
        ]
        batch_size = max(sizes)
        if batch_size < 1 or any(size not in (1, batch_size) for size in sizes):
            raise ValueError(f"all batched inputs must have the same batch size, got {sizes}")
        return batch_size

    @staticmethod
    def _as_batch(value: Any, batch_size: int, name: str) -> list[Any] | None:
        if value is None:
            return None
        values = value if isinstance(value, list) else [value]
        if len(values) == 1:
            return values * batch_size
        if len(values) != batch_size:
            raise ValueError(f"{name} must contain one item or {batch_size} items, got {len(values)}")
        return values

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        max_sequence_length: int,
        output_type: str,
        prompt_embeds: Tensor | None = None,
        pooled_prompt_embeds: Tensor | None = None,
        negative_prompt_embeds: Tensor | None = None,
        negative_pooled_prompt_embeds: Tensor | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        sample_audio: bool = True,
        image: str | object | None = None,
        visual_cond_scheme: str = "pretrain",
    ) -> None:
        r"""Validate arguments shared by text-to-video and image-to-video calls.

        Args:
            prompt (`str` or `list[str]`, *optional*): Text prompt or a batch of prompts.
            negative_prompt (`str` or `list[str]`, *optional*): Negative text prompt or a batch of prompts.
            height (`int`): Requested output height in pixels.
            width (`int`): Requested output width in pixels.
            num_frames (`int`): Requested output frame count.
            num_inference_steps (`int`): Number of denoising steps.
            max_sequence_length (`int`): Maximum Qwen prompt length after the template.
            output_type (`str`): One of `"pt"`, `"torch"`, `"np"`, `"numpy"`, or `"latent"`.
            prompt_embeds (`torch.Tensor`, *optional*): Precomputed Qwen prompt embeddings.
            pooled_prompt_embeds (`torch.Tensor`, *optional*): Precomputed CLIP pooled prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*): Precomputed negative Qwen embeddings.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*): Precomputed negative CLIP pooled embeddings.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*): Tensor names exposed to the step callback.
            sample_audio (`bool`, *optional*, defaults to `True`): Whether to generate and decode audio.
            image (`str`, `PIL.Image.Image`, or a list thereof, *optional*): Reference image or batch of images.
            visual_cond_scheme (`str`, *optional*, defaults to `"pretrain"`): Image-conditioning scheme when `image`
                is set.

        Raises:
            ValueError: If an input combination is unsupported.
        """
        for name, value in (("prompt", prompt), ("negative_prompt", negative_prompt)):
            if value is not None and not isinstance(value, (str, list)):
                raise ValueError(f"`{name}` has to be a `str` or a `list`, but is {type(value)}")
            if isinstance(value, list) and (not value or not all(isinstance(item, str) for item in value)):
                raise ValueError(f"`{name}` must be a non-empty list of strings")

        if prompt is None and prompt_embeds is None:
            raise ValueError("provide either `prompt` or `prompt_embeds`")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("provide either `prompt` or `prompt_embeds`, not both")
        if (prompt_embeds is None) != (pooled_prompt_embeds is None):
            raise ValueError("`prompt_embeds` and `pooled_prompt_embeds` must be provided together")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("provide either `negative_prompt` or `negative_prompt_embeds`, not both")
        if (negative_prompt_embeds is None) != (negative_pooled_prompt_embeds is None):
            raise ValueError("`negative_prompt_embeds` and `negative_pooled_prompt_embeds` must be provided together")

        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if height < 1 or width < 1:
            raise ValueError("height and width must be positive")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("height and width must be divisible by 8")
        if output_type not in ("pt", "torch", "np", "numpy", "latent"):
            raise ValueError("output_type must be 'pt', 'torch', 'np', 'numpy', or 'latent'")
        if max_sequence_length < 1 or max_sequence_length > 1024:
            raise ValueError("max_sequence_length must be between 1 and 1024")
        if sample_audio and self.audio_vae is None:
            raise ValueError("sample_audio=True requires an audio_vae")

        patch_size = tuple(int(value) for value in getattr(self.transformer, "patch_size", (1, 2, 2)))
        if len(patch_size) < 3 or any(value <= 0 for value in patch_size):
            raise ValueError(f"transformer.patch_size must contain three positive values, got {patch_size}")
        if (height // 8) % patch_size[1] != 0 or (width // 8) % patch_size[2] != 0:
            raise ValueError(
                "height and width must produce whole DiT spatial patches; "
                f"got latent size {(height // 8, width // 8)} and patch size {patch_size}"
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            name in self._callback_tensor_inputs for name in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {callback_on_step_end_tensor_inputs}"
            )
        if visual_cond_scheme not in ("pretrain", "i2v", "tail_cond_first_frame"):
            raise ValueError(f"unknown visual_cond_scheme={visual_cond_scheme!r}")
        if image is not None:
            if not getattr(self.transformer, "visual_cond", False):
                raise ValueError("image conditioning requires transformer.visual_cond=True")
            if (
                visual_cond_scheme == "tail_cond_first_frame"
                and getattr(self.transformer, "visual_token_type_num_embeddings", 0) < 2
            ):
                raise ValueError("tail_cond_first_frame requires transformer.visual_token_type_num_embeddings >= 2")

    def _encode_single_prompt(
        self, text: str | list[str], max_sequence_length: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        r"""Encode one prompt (or batch of prompts) with Qwen2.5-VL and CLIP.

        Args:
            text (`str` or `list[str]`): Prompt or prompts to encode.
            max_sequence_length (`int`): Maximum number of Qwen text tokens.
            device (`torch.device`): Execution device to move tokenized inputs to. This must be the pipeline's
                `_execution_device`, not a component's own parameter device: under CPU/group offloading a
                component's resting parameter device can be `meta` or `cpu` even though it computes on the
                accelerator once its forward hook runs.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor or None]`: `(prompt_embeds, pooled_prompt_embeds,
            attention_mask)`. `attention_mask` is `None` when every prompt in the batch fills
            `max_sequence_length` (an all-`True` mask carries no extra information).
        """
        texts = [text] if isinstance(text, str) else text
        full_texts = [_PROMPT_TEMPLATE.format(item) for item in texts]
        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_sequence_length + _QWEN_CROP_START,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(device)
        qwen_output = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )
        prompt_embeds = qwen_output["hidden_states"][-1][:, _QWEN_CROP_START:]
        attention_mask = inputs["attention_mask"][:, _QWEN_CROP_START:].to(dtype=torch.bool)
        if attention_mask.all():
            # An all-True mask carries no information beyond "attend to everything", which is what the
            # transformer already does when no mask is passed (see transformer_kandinsky6.py's
            # _normalize_attn_mask). Drop it so the dense mask isn't threaded through the denoise loop.
            attention_mask = None

        clip_inputs = self.tokenizer_2(
            texts,
            max_length=_CLIP_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        pooled_prompt_embeds = self.text_encoder_2(**clip_inputs)["pooler_output"]
        return prompt_embeds, pooled_prompt_embeds, attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Tensor | None = None,
        pooled_prompt_embeds: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        negative_prompt_embeds: Tensor | None = None,
        negative_pooled_prompt_embeds: Tensor | None = None,
        negative_prompt_attention_mask: Tensor | None = None,
        max_sequence_length: int = 1024,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        r"""Encode the positive and (optionally) negative prompt.

        Args:
            prompt (`str` or `list[str]`): Prompt or prompts to encode.
            negative_prompt (`str` or `list[str]`, *optional*): Negative prompt or prompts, used when
                `do_classifier_free_guidance=True` and `negative_prompt_embeds` is not already provided.
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`): Whether to also encode
                `negative_prompt`.
            prompt_embeds (`torch.Tensor`, *optional*): Precomputed Qwen prompt embeddings, skipping encoding of
                `prompt`.
            pooled_prompt_embeds (`torch.Tensor`, *optional*): Precomputed CLIP pooled prompt embeddings.
            prompt_attention_mask (`torch.Tensor`, *optional*): Qwen attention mask paired with `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*): Precomputed negative Qwen embeddings.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*): Precomputed negative CLIP pooled embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*): Qwen attention mask paired with
                `negative_prompt_embeds`.
            max_sequence_length (`int`, *optional*, defaults to 1024): Maximum Qwen prompt length after the
                template.
            device (`torch.device`, *optional*): Execution device for newly-encoded prompts. Defaults to
                `self._execution_device`.

        Returns:
            `tuple[torch.Tensor, ...]`: `(prompt_embeds, pooled_prompt_embeds, prompt_attention_mask,
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask)`.
        """
        device = device or self._execution_device
        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self._encode_single_prompt(
                prompt, max_sequence_length, device
            )
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = (
                self._encode_single_prompt(negative_prompt, max_sequence_length, device)
            )
        return (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def expand_prompts(
        self,
        prompt: str | list[str],
        image: Any | list[Any] | None = None,
        max_sequence_length: int = 1024,
        mode: str | None = None,
        generator: torch.Generator | None = None,
    ) -> str | list[str]:
        """Expand T2VA prompts, or image-grounded I2VA prompts when an image is supplied.

        Args:
            prompt: Prompt or a batch of prompts to expand.
            image: Optional reference image or batch of reference images, for `mode="i2va"`.
            max_sequence_length: Maximum number of new tokens to generate per prompt.
            mode: `"t2va"` or `"i2va"`. Defaults to `"i2va"` when `image` is given, else `"t2va"`.
            generator: Optional generator whose seed makes the (sampled) expansion reproducible.

        Returns:
            The expanded prompt, or a list of expanded prompts when `prompt` is a list.
        """
        if isinstance(prompt, list):
            images = image if isinstance(image, (list, tuple)) else [image] * len(prompt)
            return [
                self.expand_prompts(
                    item, image=item_image, max_sequence_length=max_sequence_length, mode=mode, generator=generator
                )
                for item, item_image in zip(prompt, images, strict=True)
            ]
        mode = mode or ("i2va" if image is not None else "t2va")
        if mode not in ("t2va", "i2va"):
            raise ValueError(f"unknown prompt expansion mode={mode!r}")
        image = self._prepare_expansion_image(image) if image is not None else None
        instruction = (_I2VA_EXPANSION_INSTRUCTION if mode == "i2va" else _T2VA_EXPANSION_INSTRUCTION).format(prompt)
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = self._execution_device
        inputs = self.tokenizer(
            text=[text],
            images=[image] if image is not None else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)
        if generator is not None:
            # `generate()` samples from the global RNG rather than accepting a `torch.Generator`, so seed
            # the global RNG from the pipeline's own generator to make the expansion reproducible. This
            # doesn't affect later `randn_tensor(..., generator=generator)` calls, which use `generator`
            # directly rather than the global RNG state.
            torch.manual_seed(generator.initial_seed())
        generated = self.text_encoder.generate(**inputs, max_new_tokens=max_sequence_length)
        qwen_crop_start = inputs["input_ids"].shape[1]
        trimmed = [output[qwen_crop_start:] for output in generated]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    @staticmethod
    def _prepare_expansion_image(image: Any) -> Any:
        if isinstance(image, (str, Path)):
            from PIL import Image

            with Image.open(image) as pil_image:
                return pil_image.convert("RGB").copy()
        return image

    @staticmethod
    def _load_pil_rgb(image: str | object):
        from PIL import Image

        if isinstance(image, str):
            try:
                pil_image = Image.open(image).convert("RGB")
                pil_image.load()
            except Exception as exc:
                raise ValueError(f"Cannot decode i2va input image {image!r}: {exc}") from exc
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"i2va image must be a path or PIL image, got {type(image).__name__}")
        return pil_image

    def _encode_i2va_first_frame(self, image: str | object, device: torch.device, height: int, width: int) -> Tensor:
        """Resize, center-crop, and VAE-encode one reference image into the packed K6 first-frame latent."""
        try:
            import torchvision.transforms.functional as TF
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "I2VA image processing requires torchvision. Install it with `pip install torchvision`."
            ) from exc

        pil_image = self._load_pil_rgb(image)
        tensor = TF.pil_to_tensor(pil_image).unsqueeze(0)
        src_h, src_w = tensor.shape[-2:]
        scale = min(src_h / height, src_w / width)
        tensor = TF.resize(tensor, (int(src_h / scale), int(src_w / scale)))
        cur_h, cur_w = tensor.shape[-2:]
        tensor = TF.crop(tensor, (cur_h - height) // 2, (cur_w - width) // 2, height, width)

        tensor = tensor.to(device=device, dtype=next(self.vae.parameters()).dtype) / 127.5 - 1.0
        tensor = tensor.transpose(0, 1).unsqueeze(0)
        encoded = self.vae.encode(tensor)
        posterior = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
        latent = posterior.sample()
        latent = latent.squeeze(0).permute(1, 2, 3, 0)
        return latent * self.vae.config.scaling_factor

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_latent_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: Tensor | None = None,
    ) -> Tensor:
        shape = (batch_size, num_latent_frames, height // 8, width // 8, num_channels_latents)
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
            if latents.ndim == 4:
                latents = latents.unsqueeze(0)
            bchtw = (batch_size, num_channels_latents, num_latent_frames, height // 8, width // 8)
            if tuple(latents.shape) == bchtw:
                latents = latents.permute(0, 2, 3, 4, 1)
            if tuple(latents.shape) != shape:
                raise ValueError(
                    f"`latents` must have shape {shape} (or the BCHTW equivalent), got {tuple(latents.shape)}"
                )
            return latents
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def prepare_audio_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        audio_duration: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        audio_latents: Tensor | None = None,
    ) -> Tensor:
        shape = (batch_size, audio_duration, num_channels_latents)
        if audio_latents is not None:
            audio_latents = audio_latents.to(device=device, dtype=dtype)
            if audio_latents.ndim == 2:
                audio_latents = audio_latents.unsqueeze(0)
            if tuple(audio_latents.shape) != shape:
                raise ValueError(f"`audio_latents` must have shape {shape}, got {tuple(audio_latents.shape)}")
            return audio_latents
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def _audio_latent_duration(
        num_video_latent_frames: int,
        *,
        fps: float = 24.0,
        audio_fps: int = 44100,
        downsample_factor: int = 1024,
    ) -> int:
        """Audio latent length matching K5 T2VA: ceil(sample_frames/fps * audio_fps / downsample)."""
        sample_frames = (num_video_latent_frames - 1) * 4 + 1
        return int(math.ceil(sample_frames / fps * audio_fps / downsample_factor))

    @staticmethod
    def _append_i2va_tail_condition(
        latents: Tensor,
        first_frames: Tensor,
        batch_size: int,
        num_latent_frames: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append one clean reference frame to batched `(B, T, H, W, C)` latents.

        Returns the extended latents, the per-frame token-type ids (`0` = generated, `1` = reference), and a
        boolean mask selecting the generated frames.
        """
        if latents.shape[0] != batch_size or latents.shape[1] != num_latent_frames:
            raise ValueError(
                "generated visual latent shape mismatch: expected "
                f"({batch_size}, {num_latent_frames}, H, W, C), got {tuple(latents.shape)}"
            )
        _, _, height, width, channels = latents.shape
        first_frames = first_frames.to(device=latents.device, dtype=latents.dtype)
        expected = (batch_size, height, width, channels)
        if tuple(first_frames.shape) != expected:
            raise ValueError(
                f"first-frame latent shape mismatch: expected {expected}, got {tuple(first_frames.shape)}"
            )

        latents = torch.cat([latents, first_frames[:, None]], dim=1)
        token_types = torch.cat(
            [
                torch.zeros((batch_size, num_latent_frames), dtype=torch.long, device=latents.device),
                torch.ones((batch_size, 1), dtype=torch.long, device=latents.device),
            ],
            dim=1,
        )
        return latents, token_types, token_types == 0

    @staticmethod
    def _apply_visual_conditioning(latents: Tensor, first_frames: Tensor | None, visual_cond_scheme: str) -> Tensor:
        """Append visual-conditioning channels for batched `(B, T, H, W, C)` video latents."""
        cond = torch.zeros_like(latents)
        mask = torch.zeros((*latents.shape[:-1], 1), dtype=latents.dtype, device=latents.device)
        if first_frames is None:
            return torch.cat([latents, cond, mask], dim=-1)

        first_frames = first_frames.to(device=latents.device, dtype=latents.dtype)
        tail_cond = visual_cond_scheme == "tail_cond_first_frame"
        if visual_cond_scheme == "i2v":
            latents[:, 0] = first_frames
        elif tail_cond:
            latents[:, -1] = first_frames
        elif visual_cond_scheme == "pretrain":
            cond[:, 0] = first_frames
        else:
            raise ValueError(f"unknown visual_cond_scheme={visual_cond_scheme!r}")
        mask[:, -1 if tail_cond else 0] = 1
        return torch.cat([latents, cond, mask], dim=-1)

    @staticmethod
    def _compute_rope1d(rope: nn.Module, length: int, device: torch.device | None = None) -> Tensor:
        """Build 1-D RoPE and move non-persistent tables for Diffusers offload."""
        if length < 1:
            raise ValueError(f"rope length must be positive, got {length}")
        if device is None:
            device = next(rope.buffers()).device
        else:
            device = torch.device(device)
            # Accelerate's model CPU-offload hook only moves the transformer when its forward starts. RoPE is
            # materialized before that forward, so explicitly move its non-persistent lookup table first.
            rope.to(device)
        pos = torch.arange(length, device=device)
        return rope(pos)

    @staticmethod
    def _compute_visual_rope(
        rope: nn.Module,
        shape: tuple[int, int, int],
        scale_factor: tuple[float, float, float],
        device: torch.device | None = None,
    ) -> Tensor:
        """Build 3-D RoPE and move non-persistent tables for Diffusers offload."""
        t, h, w = (int(shape[0]), int(shape[1]), int(shape[2]))
        if t < 1 or h < 1 or w < 1:
            raise ValueError(f"visual rope shape must be positive, got {shape}")
        if device is None:
            device = next(rope.buffers()).device
        else:
            device = torch.device(device)
            rope.to(device)
        pos = [torch.arange(t, device=device), torch.arange(h, device=device), torch.arange(w, device=device)]
        return rope((t, h, w), pos, tuple(float(value) for value in scale_factor))

    def _prepare_text_ropes(
        self,
        text_length: int,
        negative_text_length: int,
        audio_length: int | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """Build the video/audio text-branch RoPE for the cond/uncond passes, plus the audio latent RoPE."""
        video_text_rope = self._compute_rope1d(self.transformer.video_text_rope_embeddings, text_length, device=device)
        audio_text_rope = self._compute_rope1d(self.transformer.audio_text_rope_embeddings, text_length, device=device)
        negative_video_text_rope = self._compute_rope1d(
            self.transformer.video_text_rope_embeddings, negative_text_length, device=device
        )
        negative_audio_text_rope = self._compute_rope1d(
            self.transformer.audio_text_rope_embeddings, negative_text_length, device=device
        )
        audio_rope = (
            self._compute_rope1d(self.transformer.audio_rope_embeddings, audio_length, device=device)
            if audio_length is not None
            else None
        )
        return video_text_rope, audio_text_rope, negative_video_text_rope, negative_audio_text_rope, audio_rope

    @staticmethod
    def _apply_guidance(cond: Tensor, uncond: Tensor, guidance_scale: float) -> Tensor:
        return uncond + guidance_scale * (cond - uncond)

    def _postprocess_video(self, latents: Tensor) -> Tensor:
        """Decode video latents `(B, T, H, W, C)` into `(B, 3, T, H, W)` `uint8` frames in `[0, 255]`."""
        frames = (latents / self.vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
        # Hunyuan VAE loads as fp16; DiT latents are bf16 — match weight dtype.
        vae_dtype = next(self.vae.parameters()).dtype
        frames = self.vae.decode(frames.to(dtype=vae_dtype)).sample
        return ((frames.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    def _postprocess_audio(self, audio_latents: Tensor | None) -> list[np.ndarray] | None:
        """Decode audio latents `(B, A, D)` into a list of int16 `(samples,)` waveforms, or `None` in T2V mode."""
        if audio_latents is None:
            return None
        audio = audio_latents / getattr(self.audio_vae, "scaling_factor", 1.0)
        audio = audio + getattr(self.audio_vae, "mean_value", 0.0)
        decoded = self.audio_vae.wrapped_decode(audio.transpose(1, 2))
        if decoded.ndim == 1:
            decoded = decoded.unsqueeze(0)
        elif decoded.ndim == 3 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        return [(np.clip(waveform.cpu().float().numpy(), -1.0, 1.0) * 32767).astype(np.int16) for waveform in decoded]

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: str | object | list[str | object] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        sample_fps: float = 24.0,
        num_inference_steps: int = 50,
        max_sequence_length: int = 1024,
        guidance_scale: float = 5.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        latents: Tensor | None = None,
        audio_latents: Tensor | None = None,
        prompt_embeds: Tensor | None = None,
        pooled_prompt_embeds: Tensor | None = None,
        negative_prompt_embeds: Tensor | None = None,
        negative_pooled_prompt_embeds: Tensor | None = None,
        sample_audio: bool = True,
        expand_prompts: bool = False,
        output_type: str = "pt",
        return_dict: bool = True,
        callback_on_step_end: Callable[..., Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        visual_cond_scheme: str | None = None,
    ) -> Kandinsky6TI2VAPipelineOutput | tuple[Tensor | np.ndarray, list[np.ndarray] | None]:
        r"""Generate synchronized video and, optionally, audio from text or an image.

        Args:
            prompt (`str` or `list[str]`, *optional*): Text prompt or a batch of prompts.
            image (`str`, `PIL.Image.Image`, or a list thereof, *optional*): Reference image or batch of images.
            negative_prompt (`str` or `list[str]`, *optional*): Negative prompt.
            height (`int`, *optional*, defaults to 512): Output video height in pixels; must be divisible by 8.
            width (`int`, *optional*, defaults to 768): Output video width in pixels; must be divisible by 8.
            num_frames (`int`, *optional*, defaults to 121): Number of decoded video frames.
            sample_fps (`float`, *optional*, defaults to 24.0): Output video frame rate. Kandinsky 6 is trained
                for 24.0 fps; other values may affect audio/video alignment.
            num_inference_steps (`int`, *optional*, defaults to 50): Number of denoising steps.
            max_sequence_length (`int`, *optional*, defaults to 1024): Maximum Qwen prompt length after the
                template.
            guidance_scale (`float`, *optional*, defaults to 5.0): Classifier-free guidance weight.
            num_images_per_prompt (`int`, *optional*, defaults to 1): Number of videos to generate per prompt.
            generator (`torch.Generator`, *optional*): Random generator used for latent initialization and, when
                `expand_prompts=True`, to seed prompt expansion.
            latents (`torch.Tensor`, *optional*): Precomputed video latents.
            audio_latents (`torch.Tensor`, *optional*): Precomputed audio latents.
            prompt_embeds (`torch.Tensor`, *optional*): Precomputed Qwen prompt embeddings.
            pooled_prompt_embeds (`torch.Tensor`, *optional*): Precomputed CLIP pooled prompt embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*): Precomputed negative Qwen embeddings.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*): Precomputed negative CLIP pooled embeddings.
            sample_audio (`bool`, *optional*, defaults to `True`): Whether to generate synchronized audio.
            expand_prompts (`bool`, *optional*, defaults to `False`): Whether to use the built-in Qwen video+audio
                prompt expander before encoding.
            output_type (`str`, *optional*, defaults to `"pt"`): `"pt"`/`"torch"` for tensors, `"np"`/`"numpy"` for
                NumPy arrays, or `"latent"` for undecoded video latents.
            return_dict (`bool`, *optional*, defaults to `True`): Whether to return `Kandinsky6TI2VAPipelineOutput`.
            callback_on_step_end (`Callable`, *optional*): Callback invoked after each step.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*): Names passed to the callback.
            visual_cond_scheme (`str`, *optional*): Image-conditioning scheme. Defaults to `"pretrain"` without
                `image` and `"tail_cond_first_frame"` with `image`.

        Examples:

        Returns:
            [`Kandinsky6TI2VAPipelineOutput`] or `tuple`: [`Kandinsky6TI2VAPipelineOutput`] containing video
            frames and int16 NumPy audio waveforms if `return_dict=True`, otherwise a `(frames, audio)` tuple.
        """
        if sample_fps != 24.0:
            warnings.warn(
                f"Kandinsky 6 was trained for 24.0 fps; received sample_fps={sample_fps}.",
                UserWarning,
                stacklevel=2,
            )

        # 1. Define call parameters
        batch_size = self._batch_size(
            prompt, negative_prompt, image, latents, audio_latents, prompt_embeds, negative_prompt_embeds
        )
        prompt_batch = self._as_batch(prompt, batch_size, "prompt")
        negative_prompt_batch = self._as_batch(negative_prompt, batch_size, "negative_prompt")
        image_batch = self._as_batch(image, batch_size, "image")
        if visual_cond_scheme is None:
            visual_cond_scheme = "tail_cond_first_frame" if image is not None else "pretrain"

        # 2. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt_batch,
            negative_prompt=negative_prompt_batch,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            sample_audio=sample_audio,
            image=image_batch,
            visual_cond_scheme=visual_cond_scheme,
        )
        callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs or self._callback_tensor_inputs

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        dtype = (
            self.transformer.dtype
            if isinstance(getattr(self.transformer, "dtype", None), torch.dtype)
            else torch.bfloat16
        )

        # 3. Encode the reference image, if any
        first_frames = None
        if image_batch is not None:
            first_frames = torch.cat(
                [self._encode_i2va_first_frame(item, device, height, width) for item in image_batch], dim=0
            )

        # 4. Encode input prompt
        if expand_prompts and prompt_batch is not None and prompt_embeds is None:
            prompt_batch = self.expand_prompts(
                prompt_batch,
                image=image_batch,
                max_sequence_length=max_sequence_length,
                mode="i2va" if image_batch is not None else "t2va",
                generator=generator,
            )
        negative_prompt_batch = negative_prompt_batch or [self._DEFAULT_NEGATIVE_PROMPT] * batch_size

        (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt_batch or [""] * batch_size,
            negative_prompt=negative_prompt_batch,
            do_classifier_free_guidance=True,
            device=device,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=device, dtype=dtype)
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device=device)
        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=device)

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
        if negative_prompt_attention_mask is not None:
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat_interleave(
                num_images_per_prompt, dim=0
            )
        if first_frames is not None:
            first_frames = first_frames.repeat_interleave(num_images_per_prompt, dim=0)
        batch_size = batch_size * num_images_per_prompt

        # 5. Prepare latent variables
        patch_size = tuple(int(value) for value in self.transformer.patch_size)
        num_latent_frames = (num_frames - 1) // 4 + 1
        num_channels_latents = int(self.transformer.in_visual_dim)
        latents = self.prepare_latents(
            batch_size, num_channels_latents, num_latent_frames, height, width, dtype, device, generator, latents
        )
        if sample_audio:
            audio_duration = self._audio_latent_duration(
                num_latent_frames,
                fps=sample_fps,
                audio_fps=self.audio_sample_rate,
                downsample_factor=self.audio_downsample_factor,
            )
            audio_latents = self.prepare_audio_latents(
                batch_size,
                int(self.transformer.in_audio_dim),
                audio_duration,
                dtype,
                device,
                generator,
                audio_latents,
            )
        elif audio_latents is not None:
            raise ValueError("`audio_latents` cannot be provided when `sample_audio=False`")
        else:
            audio_latents = None

        visual_token_type_ids = None
        generated_visual_mask = None
        if image_batch is not None and visual_cond_scheme == "tail_cond_first_frame":
            latents, visual_token_type_ids, generated_visual_mask = self._append_i2va_tail_condition(
                latents, first_frames, batch_size, num_latent_frames
            )

        # 6. Prepare rotary position embeddings
        visual_shape = (
            num_latent_frames // patch_size[0],
            (height // 8) // patch_size[1],
            (width // 8) // patch_size[2],
        )
        if visual_shape[0] < 1:
            raise ValueError(f"invalid visual latent shape {visual_shape}")
        visual_rope = self._compute_visual_rope(
            self.transformer.visual_rope_embeddings, visual_shape, self.scale_factor, device=device
        )
        if generated_visual_mask is not None:
            # The appended reference frame reuses the first frame's rotary position rather than a real T+1 one.
            visual_rope = torch.cat([visual_rope, visual_rope[:1]], dim=0)
        video_text_rope, audio_text_rope, negative_video_text_rope, negative_audio_text_rope, audio_rope = (
            self._prepare_text_ropes(
                text_length=prompt_embeds.shape[1],
                negative_text_length=negative_prompt_embeds.shape[1],
                audio_length=audio_latents.shape[1] if audio_latents is not None else None,
                device=device,
            )
        )

        # 7. Denoising loop
        is_piflow = bool(getattr(self.scheduler, "is_piflow", False))
        if is_piflow and guidance_scale != 1.0:
            raise ValueError("PiflowScheduler requires guidance_scale=1.0")
        do_classifier_free_guidance = abs(guidance_scale - 1.0) > 1e-6
        tail_cond = visual_cond_scheme == "tail_cond_first_frame"

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # A second scheduler instance keeps audio's step index independent, since the scheduler advances its own
        # internal step index on every `.step()` call.
        audio_scheduler = copy.deepcopy(self.scheduler) if audio_latents is not None else None
        if audio_scheduler is not None:
            audio_scheduler.set_timesteps(num_inference_steps, device=device)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.unsqueeze(0).expand(batch_size)

                video_model_input = (
                    self._apply_visual_conditioning(latents, first_frames, visual_cond_scheme)
                    if self.transformer.visual_cond
                    else latents
                )

                with self.transformer.cache_context("cond"):
                    model_pred = self.transformer(
                        x_video=video_model_input,
                        x_audio=audio_latents,
                        text_embed=prompt_embeds,
                        pooled_text_embed=pooled_prompt_embeds,
                        time=timestep,
                        visual_rope=visual_rope,
                        audio_rope=audio_rope,
                        video_text_rope=video_text_rope,
                        audio_text_rope=audio_text_rope,
                        attention_mask=prompt_attention_mask,
                        visual_token_type_ids=visual_token_type_ids,
                    )

                if do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        model_pred_uncond = self.transformer(
                            x_video=video_model_input,
                            x_audio=audio_latents,
                            text_embed=negative_prompt_embeds,
                            pooled_text_embed=negative_pooled_prompt_embeds,
                            time=timestep,
                            visual_rope=visual_rope,
                            audio_rope=audio_rope,
                            video_text_rope=negative_video_text_rope,
                            audio_text_rope=negative_audio_text_rope,
                            attention_mask=negative_prompt_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                        )
                    if audio_latents is not None:
                        video_pred = self._apply_guidance(model_pred[0], model_pred_uncond[0], self.guidance_scale)
                        audio_pred = self._apply_guidance(model_pred[1], model_pred_uncond[1], self.guidance_scale)
                    else:
                        video_pred = self._apply_guidance(model_pred, model_pred_uncond, self.guidance_scale)
                elif audio_latents is not None:
                    video_pred, audio_pred = model_pred
                else:
                    video_pred = model_pred

                latents = self.scheduler.step(video_pred, t, latents, return_dict=False)[0]
                if audio_latents is not None:
                    audio_latents = audio_scheduler.step(audio_pred, t, audio_latents, return_dict=False)[0]
                if tail_cond and first_frames is not None:
                    latents[:, -1] = first_frames.to(device=device, dtype=latents.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        self._current_timestep = None

        # 8. Restore the injected reference frame and drop it from the output
        if first_frames is not None:
            ff = first_frames.to(device=device, dtype=latents.dtype)
            if visual_cond_scheme == "i2v":
                latents[:, 0] = ff
            elif tail_cond:
                latents[:, -1] = ff
        if generated_visual_mask is not None:
            latents = latents[:, generated_visual_mask[0]]

        # 9. Decode outputs
        if output_type == "latent":
            frames: Tensor | np.ndarray = latents.permute(0, 4, 1, 2, 3)
        else:
            frames = self._postprocess_video(latents)
            if output_type not in ("pt", "torch"):
                frames = frames.permute(0, 2, 3, 4, 1).cpu().numpy()
        audio = self._postprocess_audio(audio_latents) if sample_audio else None

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames, audio
        return Kandinsky6TI2VAPipelineOutput(frames=frames, audio=audio)
