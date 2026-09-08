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

import math
from typing import TYPE_CHECKING

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


class LTX2Guidance(BaseGuidance):
    """
    Single-modality guider for LTX-2.X. Combines up to three terms via the delta formulation:
      - classifier-free guidance (CFG),
      - spatio-temporal guidance (STG), an extra pass with a set of transformer blocks perturbed,
      - modality-isolation guidance, an extra pass with A2V/V2A cross-attention disabled.

    One instance drives one modality, so a multimodal checkpoint carries several: LTX-2.X pairs a video `guider`
    with an `audio_guider`, each holding its own scales. The final guidance estimate is computed in whatever space
    the caller feeds it — the LTX-2 denoiser converts velocity->x0 before calling this and back afterwards, so
    `forward` operates on x0 predictions.

    Unlike `SkipLayerGuidance`, which applies its perturbations with hooks, `LTX2Guidance` expects its non-CFG
    perturbations to be configurable through model forward arguments (`spatio_temporal_guidance_blocks` and
    `isolate_modalities` in `LTX2VideoTransformer3DModel`). The calling denoiser block is responsible for setting
    those per pass identifier; see `LTX2LoopDenoiser` for a usage example.

    Args:
        guidance_scale (`float`, defaults to `1.0`):
            CFG scale for this modality. The CFG pass is skipped entirely at `1.0`.
        stg_scale (`float`, defaults to `0.0`):
            Spatio-temporal guidance scale for this modality. The STG pass is skipped entirely at `0.0`.
        modality_scale (`float`, defaults to `1.0`):
            Modality-isolation guidance scale for this modality. That pass is skipped entirely at `1.0`.
        guidance_rescale (`float`, defaults to `0.0`):
            Rescaling factor to prevent overexposure from high guidance scales. Based on [Common Diffusion Noise
            Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891). Range: 0.0 (no rescaling)
            to 1.0 (full rescaling).
        spatio_temporal_guidance_blocks (`list[int]`, *optional*):
            Transformer blocks to perturb on the STG pass. How the value is consumed is up to the denoiser block:
            LTX-2's perturbs whole blocks in a single forward that feeds every modality, so it reads this off the video
            guider alone — setting it on the audio guider has no effect, and that guider joins the same STG forward
            through its own `stg_scale`.
        start (`float`, defaults to `0.0`):
            Fraction of denoising steps (0.0-1.0) after which guidance starts.
        stop (`float`, defaults to `1.0`):
            Fraction of denoising steps (0.0-1.0) after which guidance stops.
        enabled (`bool`, defaults to `True`):
            Whether this guider applies guidance at all. When `False`, only the conditional pass runs.
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

    # Each pass reads its own slot ([cond, uncond, stg, modality]) of every `guider_input_fields` tuple, selected
    # by this per-identifier index. Distinct slots (rather than cond/uncond only) let STG/modality pull the same
    # conditioning as `pred_cond` while the denoiser overrides their per-pass model flags after preparation.
    _PREDICTION_INDEX = {"pred_cond": 0, "pred_uncond": 1, "pred_cond_stg": 2, "pred_cond_modality": 3}

    def prepare_inputs(self, guider_inputs: dict) -> list["BlockState"]:
        # One identifier-tagged batch per active pass. Every value in `guider_inputs` is a 4-tuple indexed by
        # `_PREDICTION_INDEX` ([cond, uncond, stg, modality]); each pass reads its own slot, so the per-pass model
        # flags (`spatio_temporal_guidance_blocks`, `isolate_modalities`) are carried exactly like the encoder
        # inputs. The denoiser fills each returned batch's `noise_pred` before calling the guider to combine.
        return [
            self._prepare_batch(guider_inputs, self._PREDICTION_INDEX[pred], pred)
            for pred in self.active_predictions()
        ]

    def prepare_inputs_from_block_state(self, data: "BlockState", input_fields: dict) -> list["BlockState"]:
        # One identifier-tagged batch per active pass. Each value in `input_fields` maps a transformer argument to a
        # 4-tuple of block-state attribute names indexed by `_PREDICTION_INDEX` ([cond, uncond, stg, modality]); the
        # base helper reads the pass's slot off `data`. The denoiser then sets the per-pass model flags and fills
        # each batch's `noise_pred` before calling the guider to combine.
        return [
            self._prepare_batch_from_block_state(input_fields, data, self._PREDICTION_INDEX[pred], pred)
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
