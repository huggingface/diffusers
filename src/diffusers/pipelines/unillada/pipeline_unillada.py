# Copyright 2025 Ant Group and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import PIL.Image
import torch

from ...schedulers import BlockRefinementScheduler
from ...utils import logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import UniLLaDaPipelineOutput


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from diffusers import UniLLaDaPipeline, BlockRefinementScheduler
        >>> from diffusers.pipelines.unillada.image_tokenizer import ImageTokenizer

        >>> model_id = "inclusionAI/LLaDA2.0-Uni"
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        ... )
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        >>> scheduler = BlockRefinementScheduler()
        >>> image_tokenizer = ImageTokenizer(model_path=model_id)

        >>> pipe = UniLLaDaPipeline(
        ...     transformer=model, tokenizer=tokenizer, scheduler=scheduler, image_tokenizer=image_tokenizer
        ... )

        >>> # Text-to-Image
        >>> result = pipe(prompt="A cat sitting on a windowsill at sunset")
        >>> result.images[0].save("output.png")
        ```
"""


class UniLLaDaPipeline(DiffusionPipeline):
    r"""
    Pipeline for UniLLaDA — a discrete diffusion LLM supporting text-to-image generation,
    image understanding, and image editing via block-wise iterative refinement.

    This pipeline supports three modes determined automatically by the inputs:
    - **Text-to-Image**: Provide `prompt` only.
    - **Image Understanding**: Provide `image` and `question`.
    - **Image Editing**: Provide `image` and `instruction`.

    The model (`transformer`) is expected to be a `transformers`-compatible causal LM with
    `generate_image`, `understand_image`, and `edit_image` methods (e.g., loaded with
    `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`).

    Args:
        transformer (`Any`):
            The UniLLaDA language model backbone with image generation capabilities.
            Expected to have `generate_image`, `understand_image`, and `edit_image` methods.
        tokenizer (`Any`):
            Tokenizer compatible with the transformer model.
        scheduler ([`BlockRefinementScheduler`]):
            A scheduler for block-wise refinement during discrete diffusion.
        image_tokenizer (`Any`, *optional*):
            An image tokenizer for encoding input images to VQ tokens (required for understanding and editing modes).
    """

    transformer: Any
    tokenizer: Any
    scheduler: BlockRefinementScheduler
    image_tokenizer: Any

    _optional_components = ["image_tokenizer"]
    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: Any,
        tokenizer: Any,
        scheduler: BlockRefinementScheduler,
        image_tokenizer: Any | None = None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_tokenizer=image_tokenizer,
        )

    # ================================================================
    # Image Encoding (for understanding and editing)
    # ================================================================

    def encode_image(
        self,
        image: PIL.Image.Image,
    ) -> tuple[list[int], int, int]:
        """
        Encode a PIL image to VQ token IDs with the `image_token_offset` applied.

        Args:
            image (`PIL.Image.Image`):
                Input PIL image.

        Returns:
            `tuple[list[int], int, int]`: Tuple of (token_ids_with_offset, h, w).
        """
        if self.image_tokenizer is None:
            raise ValueError(
                "`image_tokenizer` is required for image understanding and editing modes. "
                "Pass it to the pipeline constructor."
            )

        from .utils import generate_crop_size_list, var_center_crop

        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        image = var_center_crop(image, crop_size_list=crop_size_list)

        info = self.image_tokenizer.encode_with_info(image)
        # Add image_token_offset as the backbone expects
        image_token_offset = getattr(self.transformer.config, "image_token_offset", 0)
        image_tokens = [x + image_token_offset for x in info["token_ids"]]
        _, h, w = info["grid_thw"]
        return image_tokens, h, w

    # ================================================================
    # VQ Token Decoding
    # ================================================================

    @torch.inference_mode()
    def decode_tokens_to_image(
        self,
        token_ids: list[int],
        h: int,
        w: int,
        decode_fn: Callable | None = None,
        num_steps: int = 50,
        resolution_multiplier: int = 2,
        decode_mode: str = "normal",
        **decode_kwargs,
    ) -> PIL.Image.Image:
        """
        Decode VQ token IDs into a PIL Image.

        Args:
            token_ids (`list[int]`):
                VQ token IDs (without the image_token_offset).
            h (`int`):
                Semantic grid height.
            w (`int`):
                Semantic grid width.
            decode_fn (`Callable`, *optional*):
                Custom decode function. If not provided, the pipeline uses the transformer's
                built-in decode method if available.
            num_steps (`int`, defaults to 50):
                ODE/SDE sampling steps for the decoder.
            resolution_multiplier (`int`, defaults to 2):
                Upscale factor (2 = 1024px from 512px tokens).
            decode_mode (`str`, defaults to `"normal"`):
                Decoder mode: `"normal"` (50 steps) or `"decoder-turbo"` (8 steps).
            **decode_kwargs:
                Additional keyword arguments passed to the decode function.

        Returns:
            `PIL.Image.Image`: The decoded image.
        """
        if decode_fn is not None:
            return decode_fn(
                token_ids,
                h,
                w,
                resolution_multiplier=resolution_multiplier,
                num_steps=num_steps,
                decode_mode=decode_mode,
                **decode_kwargs,
            )

        # Fallback: try the transformer's own decode method
        if hasattr(self.transformer, "decode_image_tokens"):
            return self.transformer.decode_image_tokens(
                token_ids,
                h,
                w,
                num_steps=num_steps,
                resolution_multiplier=resolution_multiplier,
                decode_mode=decode_mode,
                **decode_kwargs,
            )

        raise ValueError(
            "No decode function available. Pass `decode_fn` to `__call__()` or ensure "
            "the transformer model has a `decode_image_tokens` method."
        )

    # ================================================================
    # Input Validation
    # ================================================================

    def check_inputs(
        self,
        prompt: str | None,
        image: PIL.Image.Image | None,
        question: str | None,
        instruction: str | None,
        output_type: str,
    ):
        """Validate input arguments."""
        has_prompt = prompt is not None
        has_image = image is not None
        has_question = question is not None
        has_instruction = instruction is not None

        if not has_prompt and not has_image:
            raise ValueError(
                "Invalid input combination. Provide one of:\n"
                "  - `prompt` only (text-to-image)\n"
                "  - `image` + `question` (image understanding)\n"
                "  - `image` + `instruction` (image editing)"
            )

        if has_prompt and (has_image or has_question or has_instruction):
            raise ValueError(
                "For text-to-image mode, provide `prompt` only without `image`, `question`, or `instruction`."
            )

        if has_image and not has_question and not has_instruction:
            raise ValueError(
                "When `image` is provided, also provide `question` (understanding) or `instruction` (editing)."
            )

        if has_question and has_instruction:
            raise ValueError("Provide either `question` or `instruction`, not both.")

        if output_type not in {"pil", "np", "tokens"}:
            raise ValueError(f"`output_type` must be one of 'pil', 'np', 'tokens', got {output_type!r}.")

    # ================================================================
    # Main __call__ method
    # ================================================================

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | None = None,
        image: PIL.Image.Image | None = None,
        question: str | None = None,
        instruction: str | None = None,
        height: int = 1024,
        width: int = 1024,
        steps: int = 8,
        guidance_scale: float = 2.0,
        block_length: int = 32,
        cfg_text_scale: float | None = None,
        cfg_image_scale: float = 0.0,
        decoder_steps: int | None = None,
        decode_mode: str = "decoder-turbo",
        decode_fn: Callable | None = None,
        resolution_multiplier: int = 2,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> UniLLaDaPipelineOutput | tuple:
        r"""
        Generate images or text based on the provided inputs.

        The mode is determined automatically by which arguments are provided:
        - **Text-to-Image**: Provide `prompt` only.
        - **Image Understanding**: Provide `image` + `question`.
        - **Image Editing**: Provide `image` + `instruction`.

        Args:
            prompt (`str`, *optional*):
                Text prompt for text-to-image generation.
            image (`PIL.Image.Image`, *optional*):
                Input image for understanding or editing.
            question (`str`, *optional*):
                Question about the image (understanding mode).
            instruction (`str`, *optional*):
                Editing instruction (editing mode).
            height (`int`, defaults to 1024):
                Output image height in pixels (text-to-image only).
            width (`int`, defaults to 1024):
                Output image width in pixels (text-to-image only).
            steps (`int`, defaults to 8):
                Number of block diffusion steps for the LLM backbone.
            guidance_scale (`float`, defaults to 2.0):
                CFG scale for the LLM backbone during token generation.
            block_length (`int`, defaults to 32):
                Block size for the LLM block diffusion.
            cfg_text_scale (`float`, *optional*):
                CFG scale for text in editing mode. Defaults to `guidance_scale`.
            cfg_image_scale (`float`, defaults to 0.0):
                CFG scale for image in editing mode.
            decoder_steps (`int`, *optional*):
                Number of decoder diffusion steps. Defaults to 8 for turbo, 50 for normal.
            decode_mode (`str`, defaults to `"decoder-turbo"`):
                Decoder mode: `"decoder-turbo"` (fast, ~8 steps) or `"normal"` (quality, ~50 steps).
            decode_fn (`Callable`, *optional*):
                Custom decode function for converting VQ tokens to images. If not provided,
                the transformer's built-in decode method is used.
            resolution_multiplier (`int`, defaults to 2):
                Upscale factor (2 = 1024px from 512px tokens).
            generator (`torch.Generator`, *optional*):
                Random generator for reproducibility.
            output_type (`str`, defaults to `"pil"`):
                Output format: `"pil"`, `"np"`, or `"tokens"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`UniLLaDaPipelineOutput`] or a tuple.

        Returns:
            [`UniLLaDaPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, a [`UniLLaDaPipelineOutput`] is returned, otherwise a tuple is returned.

        Examples:
        """
        # 1. Validate inputs
        self.check_inputs(
            prompt=prompt,
            image=image,
            question=question,
            instruction=instruction,
            output_type=output_type,
        )

        # 2. Determine default decoder steps
        if decoder_steps is None:
            decoder_steps = 8 if decode_mode == "decoder-turbo" else 50

        # 3. Route to the appropriate mode
        if image is not None and question is not None:
            return self._understand_image(
                image=image,
                question=question,
                steps=steps,
                output_type=output_type,
                return_dict=return_dict,
            )
        elif image is not None and instruction is not None:
            return self._edit_image(
                image=image,
                instruction=instruction,
                steps=steps,
                block_length=block_length,
                cfg_text_scale=cfg_text_scale if cfg_text_scale is not None else guidance_scale,
                cfg_image_scale=cfg_image_scale,
                decoder_steps=decoder_steps,
                decode_mode=decode_mode,
                decode_fn=decode_fn,
                resolution_multiplier=resolution_multiplier,
                output_type=output_type,
                return_dict=return_dict,
            )
        else:
            return self._generate_image(
                prompt=prompt,
                height=height,
                width=width,
                steps=steps,
                cfg_scale=guidance_scale,
                block_length=block_length,
                decoder_steps=decoder_steps,
                decode_mode=decode_mode,
                decode_fn=decode_fn,
                resolution_multiplier=resolution_multiplier,
                output_type=output_type,
                return_dict=return_dict,
            )

    # ================================================================
    # Mode implementations
    # ================================================================

    def _generate_image(
        self,
        prompt: str,
        height: int,
        width: int,
        steps: int,
        cfg_scale: float,
        block_length: int,
        decoder_steps: int,
        decode_mode: str,
        decode_fn: Callable | None,
        resolution_multiplier: int,
        output_type: str,
        return_dict: bool,
    ) -> UniLLaDaPipelineOutput | tuple:
        """Text-to-image generation."""
        result = self.transformer.generate_image(
            prompt,
            image_h=height,
            image_w=width,
            steps=steps,
            cfg_scale=cfg_scale,
            block_length=block_length,
        )

        if output_type == "tokens":
            if not return_dict:
                return (None, str(result))
            return UniLLaDaPipelineOutput(images=None, text=str(result))

        image = self.decode_tokens_to_image(
            result["token_ids"],
            result["h"],
            result["w"],
            decode_fn=decode_fn,
            num_steps=decoder_steps,
            resolution_multiplier=resolution_multiplier,
            decode_mode=decode_mode,
        )

        if output_type == "np":
            image = np.array(image)

        if not return_dict:
            return ([image],)
        return UniLLaDaPipelineOutput(images=[image])

    def _understand_image(
        self,
        image: PIL.Image.Image,
        question: str,
        steps: int,
        output_type: str,
        return_dict: bool,
    ) -> UniLLaDaPipelineOutput | tuple:
        """Image understanding."""
        image_tokens, h, w = self.encode_image(image)

        response = self.transformer.understand_image(image_tokens, h, w, question, steps=steps)

        if not return_dict:
            return (response,)
        return UniLLaDaPipelineOutput(text=response)

    def _edit_image(
        self,
        image: PIL.Image.Image,
        instruction: str,
        steps: int,
        block_length: int,
        cfg_text_scale: float,
        cfg_image_scale: float,
        decoder_steps: int,
        decode_mode: str,
        decode_fn: Callable | None,
        resolution_multiplier: int,
        output_type: str,
        return_dict: bool,
    ) -> UniLLaDaPipelineOutput | tuple:
        """Image editing."""
        image_tokens, h, w = self.encode_image(image)

        result = self.transformer.edit_image(
            image_tokens,
            h,
            w,
            instruction,
            steps=steps,
            block_length=block_length,
            cfg_text_scale=cfg_text_scale,
            cfg_image_scale=cfg_image_scale,
        )

        if output_type == "tokens":
            if not return_dict:
                return (None, str(result))
            return UniLLaDaPipelineOutput(images=None, text=str(result))

        edited_image = self.decode_tokens_to_image(
            result["token_ids"],
            result["h"],
            result["w"],
            decode_fn=decode_fn,
            num_steps=decoder_steps,
            resolution_multiplier=resolution_multiplier,
            decode_mode=decode_mode,
        )

        if output_type == "np":
            edited_image = np.array(edited_image)

        if not return_dict:
            return ([edited_image],)
        return UniLLaDaPipelineOutput(images=[edited_image])


__all__ = ["UniLLaDaPipeline"]
