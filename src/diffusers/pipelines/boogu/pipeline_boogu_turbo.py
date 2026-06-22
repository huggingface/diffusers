"""
Boogu-Image-Turbo (DMD few-step) pipeline.

This module ports the DMD student few-step inference path from the standalone
turbo pipeline onto the in-repo `BooguImagePipeline` WITHOUT modifying
the original `pipeline_boogu.py`.

It is implemented as a thin subclass that:
  * adds the three DMD helper methods, and
  * overrides `processing(...)` to take a DMD branch when DMD inference is
    requested, otherwise delegating to the parent implementation unchanged.

The DMD path is pure text-to-image: it does not use the scheduler, reference
images, SDEdit, or classifier-free guidance. It builds its own sigma schedule,
runs `predict` -> renoise per step, then decodes the latents.

Note for reviewers: `.ai/pipelines.md` gotcha #4 asks each pipeline variant to
be its own standalone class (duplicated `__call__`, no subclassing of another
pipeline class). We deliberately keep `BooguImageTurboPipeline` as a subclass
here: `BooguImagePipeline` is a ~3.2k-line class and the Turbo variant only
changes the denoising step (the DMD branch in `processing`), so a standalone
copy would duplicate ~3.4k lines for a small behavioral delta — which conflicts
with the "keep code simple, don't duplicate" guidance in `.ai/AGENTS.md`. Left
as a subclass pending a maintainer decision on which convention should win for a
base pipeline of this size.

# Copyright (C) 2026 Boogu Team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

from .pipeline_boogu import BooguImagePipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class BooguImageTurboPipeline(BooguImagePipeline):
    """`BooguImagePipeline` plus a DMD student few-step T2I inference path.

    Enable it by passing `use_dmd_student_inference=True` to `__call__`. The DMD
    path requires pure T2I inputs and `text_guidance_scale == image_guidance_scale
    == 1.0` with `empty_instruction_guidance_scale == 0.0` (no CFG).
    """

    # ------------------------------------------------------------------ #
    # DMD helpers (ported verbatim from the standalone turbo pipeline)    #
    # ------------------------------------------------------------------ #
    def _build_dmd_student_sigmas(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        conditioning_sigma: float,
        timesteps: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if timesteps is not None:
            sigmas = torch.as_tensor(timesteps, device=device, dtype=dtype)
            if sigmas.ndim != 1 or sigmas.numel() == 0:
                raise ValueError("DMD inference timesteps must be a non-empty 1D sequence.")
            if sigmas.max().item() > 1.0:
                sigmas = sigmas / 1000.0
            return sigmas

        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1 for DMD student inference.")

        return torch.linspace(
            conditioning_sigma,
            1.0,
            num_inference_steps + 1,
            device=device,
            dtype=dtype,
        )[:-1]

    def _predict_dmd_student_step(
        self,
        latents: torch.FloatTensor,
        sigma: float,
        instruction_embeds: torch.FloatTensor,
        freqs_cis: torch.FloatTensor,
        instruction_attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        model_pred = self.predict(
            t=torch.tensor(sigma, device=latents.device, dtype=latents.dtype),
            latents=latents,
            instruction_embeds=instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=instruction_attention_mask,
            ref_image_hidden_states=None,
        )

        sigma_expanded = torch.full(
            (latents.shape[0], 1, 1, 1),
            sigma,
            device=latents.device,
            dtype=latents.dtype,
        )
        return latents + (1 - sigma_expanded) * model_pred

    def _renoise_dmd_latents(
        self,
        latents: torch.FloatTensor,
        sigma: float,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.FloatTensor:
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        sigma_expanded = torch.full(
            (latents.shape[0], 1, 1, 1),
            sigma,
            device=latents.device,
            dtype=latents.dtype,
        )
        return (1 - sigma_expanded) * noise + sigma_expanded * latents

    # ------------------------------------------------------------------ #
    # Entry point: stash DMD options, then reuse the parent __call__       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(
        self,
        *args,
        use_dmd_student_inference: bool = True,
        dmd_conditioning_sigma: float = 0.001,
        **kwargs,
    ):
        # Stash DMD options on the instance so the overridden `processing`
        # can pick them up without changing the parent __call__ signature.
        self._use_dmd_student_inference = bool(use_dmd_student_inference)
        self._dmd_conditioning_sigma = float(dmd_conditioning_sigma)
        self._dmd_generator = kwargs.get("generator", None)

        kwargs.setdefault("text_guidance_scale", 1.0)
        kwargs.setdefault("image_guidance_scale", 1.0)
        kwargs.setdefault("empty_instruction_guidance_scale", 0.0)

        return super().__call__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Denoising: take the DMD branch when requested, else delegate         #
    # ------------------------------------------------------------------ #
    def processing(self, *args, **kwargs):
        if not getattr(self, "_use_dmd_student_inference", True):
            return super().processing(*args, **kwargs)

        # Bind the parent `processing` positional/keyword args we need.
        # The parent call site passes everything by keyword, so read kwargs.
        latents = kwargs["latents"]
        ref_latents = kwargs["ref_latents"]
        instruction_embeds = kwargs["instruction_embeds"]
        freqs_cis = kwargs["freqs_cis"]
        instruction_attention_mask = kwargs["instruction_attention_mask"]
        num_inference_steps = kwargs["num_inference_steps"]
        timesteps = kwargs.get("timesteps", None)
        device = kwargs["device"]
        dtype = kwargs["dtype"]
        step_func = kwargs.get("step_func", None)

        # --- DMD constraints (mirror the standalone turbo pipeline) ---
        task_type = self._get_task_type_by_ref_latents(ref_latents)
        if task_type != "t2i":
            raise ValueError(f"DMD student inference only supports pure T2I inputs (got task_type={task_type!r}).")
        if (
            self.text_guidance_scale != 1.0
            or self.image_guidance_scale != 1.0
            or self.empty_instruction_guidance_scale != 0.0
        ):
            raise ValueError(
                "DMD student inference currently requires text_guidance_scale=1.0, "
                "image_guidance_scale=1.0, and empty_instruction_guidance_scale=0.0."
            )

        logger.info("[Turbo Pipeline Processing]: DMD student few-step T2I inference.")

        generator = getattr(self, "_dmd_generator", None)
        dmd_sigmas = self._build_dmd_student_sigmas(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=latents.dtype,
            conditioning_sigma=self._dmd_conditioning_sigma,
            timesteps=timesteps,
        )
        num_inference_steps = int(dmd_sigmas.numel())
        self._num_timesteps = num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, sigma in enumerate(dmd_sigmas.tolist()):
                latents = self._predict_dmd_student_step(
                    latents=latents,
                    sigma=sigma,
                    instruction_embeds=instruction_embeds,
                    freqs_cis=freqs_cis,
                    instruction_attention_mask=instruction_attention_mask,
                ).to(dtype=dtype)

                if i < num_inference_steps - 1:
                    latents = self._renoise_dmd_latents(
                        latents,
                        sigma=dmd_sigmas[i + 1].item(),
                        generator=generator,
                    ).to(dtype=dtype)

                progress_bar.update()
                if step_func is not None:
                    step_func(i, self._num_timesteps)

        # Decode latents (same logic as the parent `processing` tail).
        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return image
