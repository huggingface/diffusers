# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any

import torch

from ..utils import BaseOutput
from .guider_utils import BaseGuidance


@dataclass
class LTX2GuiderOutput(BaseOutput):
    r"""
    Output of the LTX2 multi-modal guider.

    Args:
        pred (`torch.Tensor`): The guided video prediction.
        pred_audio (`torch.Tensor`): The guided audio prediction.
        pred_cond (`torch.Tensor`, *optional*): Conditional video prediction before guidance.
        pred_uncond (`torch.Tensor`, *optional*): Unconditional video prediction before guidance.
    """

    pred: "torch.Tensor"
    pred_audio: "torch.Tensor"
    pred_cond: "torch.Tensor" = None
    pred_uncond: "torch.Tensor" = None


class LTX2MultiModalGuidance(BaseGuidance):
    r"""
    Multi-modal guidance for LTX-2.3 audiovisual generation.

    Handles 4 guidance types using native transformer kwargs (no hooks):
    1. **CFG** — classifier-free guidance (cond vs uncond)
    2. **STG** — spatio-temporal guidance (skip self-attention at specified blocks)
    3. **Modality isolation** — skip cross-modality attention (A2V + V2A) at all blocks
    4. **Rescale** — prevent over-saturation by matching conditioned std

    The guider passes per-batch transformer kwargs via `_model_kwargs` on each BlockState:
    - STG batch: `{"spatio_temporal_guidance_blocks": [28]}`
    - Modality batch: `{"isolate_modalities": True}`

    The denoise loop passes these through to the transformer, which handles them natively.
    Video and audio have independent guidance scales.

    Args:
        guidance_scale (`float`, defaults to `3.0`):
            Video CFG scale.
        audio_guidance_scale (`float`, defaults to `7.0`):
            Audio CFG scale.
        skip_layer_guidance_scale (`float`, defaults to `1.0`):
            STG scale for video.
        audio_skip_layer_guidance_scale (`float`, *optional*):
            STG scale for audio. Falls back to video STG scale.
        skip_layer_guidance_start (`float`, defaults to `0.0`):
            Fraction of steps after which STG starts.
        skip_layer_guidance_stop (`float`, defaults to `1.0`):
            Fraction of steps after which STG stops.
        spatio_temporal_guidance_blocks (`list[int]`):
            Transformer block indices at which to apply STG (skip self-attention).
        modality_guidance_scale (`float`, defaults to `3.0`):
            Video modality isolation scale.
        audio_modality_guidance_scale (`float`, *optional*):
            Audio modality isolation scale. Falls back to video.
        guidance_rescale (`float`, defaults to `0.7`):
            Video rescale factor.
        audio_guidance_rescale (`float`, *optional*):
            Audio rescale factor. Falls back to video.
    """

    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_skip", "pred_cond_mod"]

    def __init__(
        self,
        guidance_scale: float = 3.0,
        audio_guidance_scale: float = 7.0,
        skip_layer_guidance_scale: float = 1.0,
        audio_skip_layer_guidance_scale: float | None = None,
        skip_layer_guidance_start: float = 0.0,
        skip_layer_guidance_stop: float = 1.0,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        modality_guidance_scale: float = 3.0,
        audio_modality_guidance_scale: float | None = None,
        guidance_rescale: float = 0.7,
        audio_guidance_rescale: float | None = None,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.audio_guidance_scale = audio_guidance_scale
        self.skip_layer_guidance_scale = skip_layer_guidance_scale
        self.audio_skip_layer_guidance_scale = audio_skip_layer_guidance_scale
        self.skip_layer_guidance_start = skip_layer_guidance_start
        self.skip_layer_guidance_stop = skip_layer_guidance_stop
        self.spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or [28]
        self.modality_guidance_scale = modality_guidance_scale
        self.audio_modality_guidance_scale = audio_modality_guidance_scale
        self.guidance_rescale = guidance_rescale
        self.audio_guidance_rescale = audio_guidance_rescale

    # --- Batch preparation ---

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: dict[str, str | tuple[str, str]]
    ) -> list["BlockState"]:
        batches = []
        passes = [(0, "pred_cond", {})]
        if self._is_cfg_enabled():
            passes.append((1, "pred_uncond", {}))
        if self._is_stg_enabled():
            passes.append((0, "pred_cond_skip", {
                "spatio_temporal_guidance_blocks": self.spatio_temporal_guidance_blocks,
            }))
        if self._is_mod_enabled():
            passes.append((0, "pred_cond_mod", {
                "isolate_modalities": True,
            }))

        for tuple_idx, identifier, model_kwargs in passes:
            batch = self._prepare_batch_from_block_state(input_fields, data, tuple_idx, identifier)
            batch._model_kwargs = model_kwargs
            batches.append(batch)
        return batches

    # --- Guidance combination ---

    def __call__(self, data: list) -> LTX2GuiderOutput:
        if len(data) != self.num_conditions:
            raise ValueError(f"Expected {self.num_conditions} data items, but got {len(data)}.")

        video_preds = {getattr(d, self._identifier_key): d.noise_pred for d in data}
        audio_preds = {getattr(d, self._identifier_key): d.noise_pred_audio for d in data}

        return self.forward(video_preds=video_preds, audio_preds=audio_preds)

    def forward(
        self,
        video_preds: dict[str, torch.Tensor],
        audio_preds: dict[str, torch.Tensor],
        **kwargs,
    ) -> LTX2GuiderOutput:
        v_cond = video_preds["pred_cond"]
        a_cond = audio_preds["pred_cond"]

        has_uncond = "pred_uncond" in video_preds
        has_stg = "pred_cond_skip" in video_preds
        has_mod = "pred_cond_mod" in video_preds

        # Video weights
        v_cfg = (self.guidance_scale - 1) if has_uncond else 0.0
        v_stg = self.skip_layer_guidance_scale if has_stg else 0.0
        v_mod = (self.modality_guidance_scale - 1) if has_mod else 0.0

        # Audio weights
        a_cfg = (self.audio_guidance_scale - 1) if has_uncond else 0.0
        a_stg_scale = self.audio_skip_layer_guidance_scale if self.audio_skip_layer_guidance_scale is not None else self.skip_layer_guidance_scale
        a_stg = a_stg_scale if has_stg else 0.0
        a_mod_scale = self.audio_modality_guidance_scale if self.audio_modality_guidance_scale is not None else self.modality_guidance_scale
        a_mod = (a_mod_scale - 1) if has_mod else 0.0

        v_uncond = video_preds.get("pred_uncond", 0.0)
        a_uncond = audio_preds.get("pred_uncond", 0.0)
        v_skip = video_preds.get("pred_cond_skip", 0.0)
        a_skip = audio_preds.get("pred_cond_skip", 0.0)
        v_mod_pred = video_preds.get("pred_cond_mod", 0.0)
        a_mod_pred = audio_preds.get("pred_cond_mod", 0.0)

        any_guidance = v_cfg != 0 or v_stg != 0 or v_mod != 0 or a_cfg != 0 or a_stg != 0 or a_mod != 0
        if any_guidance:
            # Single expression matching reference's MultiModalGuider.calculate()
            guided_video = (
                v_cond
                + v_cfg * (v_cond - v_uncond)
                + v_stg * (v_cond - v_skip)
                + v_mod * (v_cond - v_mod_pred)
            )
            guided_audio = (
                a_cond
                + a_cfg * (a_cond - a_uncond)
                + a_stg * (a_cond - a_skip)
                + a_mod * (a_cond - a_mod_pred)
            )

            # Rescale matching reference: global std() (no dim arg)
            v_rescale = self.guidance_rescale
            a_rescale = self.audio_guidance_rescale if self.audio_guidance_rescale is not None else v_rescale
            if v_rescale > 0:
                factor = v_cond.std() / guided_video.std()
                factor = v_rescale * factor + (1 - v_rescale)
                guided_video = guided_video * factor
            if a_rescale > 0:
                factor = a_cond.std() / guided_audio.std()
                factor = a_rescale * factor + (1 - a_rescale)
                guided_audio = guided_audio * factor
        else:
            guided_video = v_cond
            guided_audio = a_cond

        return LTX2GuiderOutput(
            pred=guided_video,
            pred_audio=guided_audio,
            pred_cond=v_cond,
            pred_uncond=v_uncond if has_uncond else None,
        )

    # --- State queries ---

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared != 2

    @property
    def num_conditions(self) -> int:
        n = 1
        if self._is_cfg_enabled():
            n += 1
        if self._is_stg_enabled():
            n += 1
        if self._is_mod_enabled():
            n += 1
        return n

    def _is_cfg_enabled(self) -> bool:
        if not self._enabled:
            return False
        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step
        return is_within_range and not math.isclose(self.guidance_scale, 1.0)

    def _is_stg_enabled(self) -> bool:
        if not self._enabled:
            return False
        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self.skip_layer_guidance_start * self._num_inference_steps)
            skip_stop_step = int(self.skip_layer_guidance_stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step
        return is_within_range and not math.isclose(self.skip_layer_guidance_scale, 0.0)

    def _is_mod_enabled(self) -> bool:
        if not self._enabled:
            return False
        return not math.isclose(self.modality_guidance_scale, 1.0)
