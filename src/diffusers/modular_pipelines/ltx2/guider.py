# Copyright 2026 The HuggingFace Team. All rights reserved.
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

# SPIKE (guider-API variant, uncommitted): two single-modality guiders (`LTX2Guidance`, one as the video `guider`
# and one as the `audio_guider`), driven through the standard guider API. Wired into `LTX2LoopDenoiser`.
#
# The denoiser builds a `guider_inputs` dict whose values are 4-tuples indexed by pass [cond, uncond, stg,
# modality]; `guider.prepare_inputs(guider_inputs)` yields one identifier-tagged batch per active pass carrying
# that pass's encoder inputs AND its per-pass model flags (`spatio_temporal_guidance_blocks`, `isolate_modalities`).
# The denoiser runs each pass as its own single-batch forward, then each guider's `forward`/`__call__` combines its
# modality (CFG + STG + modality-isolation, delta formulation in x0 space). Carrying the flags as inputs is what
# lets STG/modality reuse the same mechanism as the encoder tensors -- the reason this uses the API end-to-end.
#
# Parity note. Because every pass (cond/uncond included) is a separate single-batch forward -- not the batched
# `torch.cat([latents] * 2)` the standard `LTX2Pipeline` uses -- this does NOT match the reference bitwise. GPU
# matmul is not batch-invariant, so `cond` computed alone differs from `cond` inside a batch-of-2: ~1e-6/op in fp32
# (still within the harness's fp32 tolerance) but ~1e-2/op in bf16, which the CFG delta and sampler amplify to
# ~10% mean-relative latent divergence. The batched-CFG variant (in git history) is the only one that is fp32
# bitwise; this design trades that for using the guider API end-to-end. Since fp32-within-tolerance (not bitwise)
# is the modular-ecosystem norm, gate parity on fp32 and treat bf16 as a close-but-not-bitwise check.

import math

import torch

from ...configuration_utils import register_to_config
from ...guiders.guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg
from ..modular_pipeline import BlockState


class LTX2Guidance(BaseGuidance):
    """
    Single-modality guider for LTX-2.X. Combines up to three terms via the delta formulation:
      - classifier-free guidance (CFG),
      - spatio-temporal guidance (STG), an extra pass with a set of transformer blocks perturbed,
      - modality-isolation guidance, an extra pass with A2V/V2A cross-attention disabled.

    Instantiated once per modality (`guider` for video, `audio_guider` for audio) with that modality's scales;
    that keeps the per-modality scales in independent, fully-defined component configs. The combine is done in
    whatever space the caller feeds it — the LTX-2 denoiser converts velocity->x0 before calling this and back
    afterwards, so `forward` operates on x0 predictions.
    """

    # `pred_cond` is the base; the others are the guidance passes.
    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_stg", "pred_cond_modality"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 1.0,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        # STG perturbs whole transformer blocks, and one forward feeds *both* modalities, so the block list is a
        # shared/transformer-level knob rather than a per-modality one; only the video guider carries it (the audio
        # guider reuses the same STG forward via its own `stg_scale`). The denoiser reads it off the video guider.
        spatio_temporal_guidance_blocks: list[int] | None = None,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)
        self.guidance_scale = guidance_scale
        self.stg_scale = stg_scale
        self.modality_scale = modality_scale
        self.guidance_rescale = guidance_rescale
        self.spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks

    # --- per-term enabled checks (public: the denoiser ORs these across the two guiders) --------------------
    def _within_window(self) -> bool:
        if self._num_inference_steps is None:
            return True
        start_step = int(self._start * self._num_inference_steps)
        stop_step = int(self._stop * self._num_inference_steps)
        return start_step <= self._step < stop_step

    def is_cfg_enabled(self) -> bool:
        return self._enabled and self._within_window() and not math.isclose(self.guidance_scale, 1.0)

    def is_stg_enabled(self) -> bool:
        return self._enabled and self._within_window() and not math.isclose(self.stg_scale, 0.0)

    def is_modality_enabled(self) -> bool:
        return self._enabled and self._within_window() and not math.isclose(self.modality_scale, 1.0)

    def active_predictions(self) -> list[str]:
        preds = ["pred_cond"]
        if self.is_cfg_enabled():
            preds.append("pred_uncond")
        if self.is_stg_enabled():
            preds.append("pred_cond_stg")
        if self.is_modality_enabled():
            preds.append("pred_cond_modality")
        return preds

    @property
    def num_conditions(self) -> int:
        return len(self.active_predictions())

    @property
    def is_conditional(self) -> bool:
        # Not load-bearing here: the denoiser owns pass execution and carries the STG/modality flags on the
        # batches, so we never drive `prepare_models`/hooks off `_count_prepared` the way SkipLayerGuidance does.
        return True

    # Each pass reads its own slot of every `guider_inputs` tuple, indexed by identifier. Distinct slots (rather
    # than cond/uncond only) let per-pass model flags ride alongside the encoder inputs -- see `prepare_inputs`.
    _PREDICTION_INDEX = {"pred_cond": 0, "pred_uncond": 1, "pred_cond_stg": 2, "pred_cond_modality": 3}

    def prepare_inputs(self, guider_inputs: dict) -> list[BlockState]:
        # One identifier-tagged batch per active pass. Every value in `guider_inputs` is a 4-tuple indexed by
        # `_PREDICTION_INDEX` ([cond, uncond, stg, modality]); each pass reads its own slot, so the per-pass model
        # flags (`spatio_temporal_guidance_blocks`, `isolate_modalities`) are carried exactly like the encoder
        # inputs. The denoiser fills each returned batch's `noise_pred` before calling the guider to combine.
        return [
            self._prepare_batch(guider_inputs, self._PREDICTION_INDEX[pred], pred)
            for pred in self.active_predictions()
        ]

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor | None = None,
        pred_cond_stg: torch.Tensor | None = None,
        pred_cond_modality: torch.Tensor | None = None,
    ) -> GuiderOutput:
        # Delta formulation (matches the reference LTX-2 combine, applied in the x0 space the denoiser feeds):
        #   x0_cond + (gs-1)*(cond-uncond) + stg*(cond-stg) + (mod-1)*(cond-modality), then optional rescale.
        pred = pred_cond
        if self.is_cfg_enabled():
            pred = pred + (self.guidance_scale - 1.0) * (pred_cond - pred_uncond)
        if self.is_stg_enabled():
            pred = pred + self.stg_scale * (pred_cond - pred_cond_stg)
        if self.is_modality_enabled():
            pred = pred + (self.modality_scale - 1.0) * (pred_cond - pred_cond_modality)
        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)
        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)
