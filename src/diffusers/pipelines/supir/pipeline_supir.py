# Copyright 2026 Fanghua-Yu and The HuggingFace Team. All rights reserved.
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
"""
SUPIR (Scaling Up to Excellence) image restoration / upscaler pipeline scaffold.

This is a SCAFFOLD ONLY. The full pipeline implementation is intentionally
deferred. The class signature, constructor wiring, and `__call__` argument
surface are exposed so downstream packaging, tests, and documentation can
land incrementally. All heavy-lifting paths raise NotImplementedError.

See the repo-root SUPIR_DESIGN.md for the porting plan from
https://github.com/Fanghua-Yu/SUPIR and the planned diffusers component
layout (degradation-robust encoder, trimmed ControlNet adaptor with
ZeroSFT connector, SDXL UNet generative prior, restoration-guided
sampler, optional LLaVA caption guidance).
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import BaseOutput, logging
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # NOTE: scaffold only. Calling SUPIRPipeline today raises NotImplementedError.
        >>> # Once the implementation lands the canonical usage will look like:
        >>> import torch
        >>> from diffusers import SUPIRPipeline
        >>> from diffusers.utils import load_image
        >>>
        >>> pipe = SUPIRPipeline.from_pretrained(
        ...     "Fanghua-Yu/SUPIR",
        ...     torch_dtype=torch.float16,
        ... ).to("cuda")
        >>>
        >>> low_quality = load_image("https://example.com/lq.png")
        >>> result = pipe(
        ...     prompt="a high quality photo, sharp details",
        ...     image=low_quality,
        ...     num_inference_steps=50,
        ...     upscale=2,
        ... ).images[0]
        ```
"""


@dataclass
class SUPIRPipelineOutput(BaseOutput):
    """
    Output class for SUPIR pipeline runs.

    Args:
        images (`list[PIL.Image.Image]` or `np.ndarray`):
            Restored / upscaled images, returned as a list of PIL images of
            length `batch_size` or as a numpy array of shape
            `(batch_size, height, width, num_channels)`.
    """

    images: list[PIL.Image.Image] | np.ndarray


class SUPIRPipeline(DiffusionPipeline, StableDiffusionMixin):
    """
    Pipeline for image restoration and super-resolution with SUPIR.

    SUPIR (Scaling Up to Excellence) restores and upscales degraded images by
    combining an SDXL generative prior, a degradation-robust encoder, and a
    trimmed ControlNet-style adaptor with ZeroSFT connectors. See the
    upstream paper at https://arxiv.org/abs/2401.13627 and
    https://github.com/Fanghua-Yu/SUPIR for the reference implementation.

    NOTE: this class is currently a SCAFFOLD. The constructor wires the
    expected components and `__call__` exposes the documented argument
    surface, but the actual restoration logic is not yet implemented and
    will raise `NotImplementedError`. The intent is to land the public API
    shape first so tests, docs, and downstream packaging can stabilise
    while the porting work in `SUPIR_DESIGN.md` proceeds.

    This model inherits from [`DiffusionPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines
    (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            VAE used to encode/decode images to/from latent space. SUPIR uses
            an SDXL VAE plus a fine-tuned degradation-robust encoder; for the
            scaffold only the standard VAE is wired here.
        text_encoder ([`CLIPTextModel`]):
            Frozen text encoder (SDXL primary). SUPIR can optionally consume
            captions produced by an external LLaVA model.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            Second frozen text encoder used by SDXL.
        tokenizer ([`CLIPTokenizer`]):
            Tokenizer for the primary text encoder.
        tokenizer_2 ([`CLIPTokenizer`]):
            Tokenizer for the secondary text encoder.
        unet ([`UNet2DConditionModel`]):
            SDXL UNet used as the generative prior.
        controlnet ([`ControlNetModel`]):
            Trimmed ControlNet-style adaptor that injects degraded-image
            features into the UNet via ZeroSFT connectors. The reference
            SUPIR repo ships dedicated `GLVControl` / `LightGLVUNet` modules;
            once ported they will replace the standard `ControlNetModel`.
        scheduler ([`KarrasDiffusionSchedulers`]):
            Sampler used during denoising. SUPIR layers a restoration-guided
            sampling step on top of a Karras-style scheduler.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = ["controlnet"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        # VAE downscale factor mirrors the SDXL conventions; matches the
        # reference repo where degraded inputs are encoded in pixel space
        # then driven through the same latent topology as SDXL.
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if vae is not None else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # ------------------------------------------------------------------
    # Helper hooks intentionally left as TODOs. Once implemented they will
    # mirror the structure used by `StableDiffusionXLControlNetPipeline`.
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
    ):
        """Encode text prompts using the SDXL dual-encoder stack.

        TODO: port from `StableDiffusionXLPipeline.encode_prompt`. Kept as a
        stub so the scaffold remains import-clean and so callers can locate
        the eventual extension point (e.g. for LLaVA-derived captions).
        """
        raise NotImplementedError(
            "SUPIRPipeline.encode_prompt is part of the scaffold and is not implemented yet. "
            "See SUPIR_DESIGN.md for the porting plan."
        )

    def prepare_low_quality_latents(
        self,
        image: PipelineImageInput,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        """Encode the degraded input image through the degradation-robust
        encoder (a fine-tuned SDXL VAE encoder in the reference repo) into
        the latent space used by the UNet.

        TODO: implement degradation-aware encoding and tiled inference for
        large inputs. Tracked in SUPIR_DESIGN.md > Stage 1.
        """
        raise NotImplementedError(
            "SUPIRPipeline.prepare_low_quality_latents is part of the scaffold "
            "and is not implemented yet."
        )

    def restoration_guided_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        lq_latents: torch.Tensor,
        guidance_scale: float,
        s_churn: float = 0.0,
        s_noise: float = 1.003,
    ) -> torch.Tensor:
        """One step of SUPIR's restoration-guided EDM-style sampler.

        TODO: implement the modified denoising step (LQ-anchored guidance,
        EDM noise injection) described in the paper, section 3.4.
        """
        raise NotImplementedError(
            "SUPIRPipeline.restoration_guided_step is part of the scaffold and "
            "is not implemented yet."
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        image: PipelineImageInput = None,
        height: int | None = None,
        width: int | None = None,
        upscale: int = 1,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        denoising_end: float | None = None,
        guidance_scale: float = 7.5,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: dict[str, Any] | None = None,
        controlnet_conditioning_scale: float = 1.0,
        s_churn: float = 0.0,
        s_noise: float = 1.003,
        callback_on_step_end: Callable[[Any, int, int, dict[str, Any]], dict[str, Any]] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> SUPIRPipelineOutput | tuple[list[PIL.Image.Image] | np.ndarray]:
        """Run SUPIR restoration / upscaling.

        Args:
            prompt (`str` or `list[str]`, optional):
                Text prompt describing the desired restoration. May be
                generated automatically from an LLaVA caption in a future
                revision; today it is the only text-conditioning surface.
            prompt_2 (`str` or `list[str]`, optional):
                Prompt for the second SDXL text encoder. Defaults to `prompt`.
            image (`PipelineImageInput`):
                The low-quality input image (or batch of images) to restore.
                Required.
            height (`int`, optional):
                Output height in pixels. Defaults to `image.height * upscale`.
            width (`int`, optional):
                Output width in pixels. Defaults to `image.width * upscale`.
            upscale (`int`, defaults to `1`):
                Convenience scale factor used when `height` and `width` are
                not provided. SUPIR commonly runs at 2x or 4x.
            num_inference_steps (`int`, defaults to `50`):
                Number of denoising steps.
            timesteps (`list[int]`, optional):
                Custom timesteps to use; bypasses scheduler defaults.
            denoising_end (`float`, optional):
                Fraction of the denoising schedule to complete; useful for
                pipelining a refiner.
            guidance_scale (`float`, defaults to `7.5`):
                Classifier-free guidance scale.
            negative_prompt / negative_prompt_2 (`str` or `list[str]`, optional):
                Negative text prompts for CFG.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of restored images per prompt.
            eta (`float`, defaults to `0.0`):
                DDIM eta parameter.
            generator (`torch.Generator` or list, optional):
                Generator(s) for deterministic sampling.
            latents (`torch.Tensor`, optional):
                Pre-computed initial latents.
            prompt_embeds / negative_prompt_embeds /
            pooled_prompt_embeds / negative_pooled_prompt_embeds
            (`torch.Tensor`, optional):
                Pre-computed embeddings; skip text encoding if provided.
            output_type (`str`, defaults to `"pil"`):
                Output format: `"pil"`, `"np"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`SUPIRPipelineOutput`] or a plain tuple.
            cross_attention_kwargs (`dict`, optional):
                Forwarded to attention processors (e.g. for LoRA scale).
            controlnet_conditioning_scale (`float`, defaults to `1.0`):
                Scale applied to the SUPIR adaptor outputs.
            s_churn (`float`, defaults to `0.0`):
                EDM stochasticity parameter for the SUPIR sampler.
            s_noise (`float`, defaults to `1.003`):
                EDM noise scaling parameter for the SUPIR sampler.
            callback_on_step_end / callback_on_step_end_tensor_inputs:
                Standard diffusers callback hooks.

        Returns:
            [`SUPIRPipelineOutput`] or `tuple`. When `return_dict=True`,
            returns `SUPIRPipelineOutput` containing the restored images.

        Raises:
            NotImplementedError: this scaffold does not yet implement the
                restoration loop. See SUPIR_DESIGN.md for status.
        """
        # Argument validation that doesn't require the model is intentionally
        # kept lightweight here so that `--dry-run` style smoke tests can
        # still exercise the call surface; the heavy lifting is gated.
        if image is None:
            raise ValueError("`image` is required: SUPIR is a restoration pipeline.")
        if upscale < 1:
            raise ValueError(f"`upscale` must be >= 1, got {upscale}.")
        if num_inference_steps < 1:
            raise ValueError(f"`num_inference_steps` must be >= 1, got {num_inference_steps}.")

        raise NotImplementedError(
            "SUPIRPipeline.__call__ is a scaffold and the restoration loop is not yet "
            "implemented. Track progress in SUPIR_DESIGN.md and the linked GitHub issue "
            "(huggingface/diffusers#7219)."
        )
