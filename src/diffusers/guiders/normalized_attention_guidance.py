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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from .classifier_free_guidance import ClassifierFreeGuidance


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


def _patch_qwenimage_attention_processors_for_nag(
    denoiser: torch.nn.Module,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
) -> Optional[Dict[str, object]]:
    original_attn_processors = getattr(denoiser, "attn_processors", None)
    if original_attn_processors is None or not hasattr(denoiser, "set_attn_processor"):
        return None

    from ..models.transformers.transformer_qwenimage import (
        QwenDoubleStreamAttnProcessor2_0,
        QwenDoubleStreamNAGAttnProcessor2_0,
    )

    nag_processor = QwenDoubleStreamNAGAttnProcessor2_0(
        nag_scale=nag_scale,
        nag_tau=nag_tau,
        nag_alpha=nag_alpha,
    )

    base_proc = next(
        (proc for proc in original_attn_processors.values() if isinstance(proc, QwenDoubleStreamAttnProcessor2_0)),
        None,
    )
    if base_proc is None:
        return None

    nag_processor._attention_backend = getattr(base_proc, "_attention_backend", None)
    nag_processor._parallel_config = getattr(base_proc, "_parallel_config", None)

    patched = {
        name: (nag_processor if isinstance(proc, QwenDoubleStreamAttnProcessor2_0) else proc)
        for name, proc in original_attn_processors.items()
    }
    denoiser.set_attn_processor(patched)
    return original_attn_processors


class NormalizedAttentionCFGGuidance(ClassifierFreeGuidance):
    """
    CFG + Normalized Attention Guidance (NAG).

    This guider keeps the standard CFG denoiser combination, while additionally applying NAG inside attention
    processors during the *conditional* forward pass (ComfyUI-style). This strengthens negative prompting especially
    in few-step regimes, without introducing an extra denoiser forward pass beyond CFG.

    - CFG reference: https://huggingface.co/papers/2207.12598
    - NAG reference: https://huggingface.co/papers/2505.21179
    """

    def __init__(
        self,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        nag_scale: float = 5.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        nag_sigma_end: Optional[float] = None,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            use_original_formulation=use_original_formulation,
            start=start,
            stop=stop,
            enabled=enabled,
        )

        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.nag_sigma_end = nag_sigma_end
        self.register_to_config(
            nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, nag_sigma_end=nag_sigma_end
        )

        self._negative_encoder_hidden_states: Optional[torch.Tensor] = None
        self._negative_encoder_hidden_states_mask: Optional[torch.Tensor] = None
        self._original_attn_processors: Optional[Dict[str, object]] = None

        if nag_tau < 1.0:
            raise ValueError(f"Expected `nag_tau` to be >= 1.0, but got {nag_tau}.")
        if not (0.0 <= nag_alpha <= 1.0):
            raise ValueError(f"Expected `nag_alpha` to be in [0, 1], but got {nag_alpha}.")
        if nag_sigma_end is not None and nag_sigma_end < 0.0:
            raise ValueError(f"Expected `nag_sigma_end` to be >= 0.0, but got {nag_sigma_end}.")

    @property
    def requires_unconditional_embeds(self) -> bool:
        # We might need unconditional embeds for NAG even when CFG is disabled.
        return self._is_cfg_enabled() or self._is_nag_enabled()

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        self._negative_encoder_hidden_states = data.get("nag_negative_prompt_embeds")
        self._negative_encoder_hidden_states_mask = data.get("nag_negative_prompt_embeds_mask")

        if self._negative_encoder_hidden_states is None:
            encoder_hidden_states = data.get("encoder_hidden_states")
            if isinstance(encoder_hidden_states, tuple):
                self._negative_encoder_hidden_states = encoder_hidden_states[1]
        if self._negative_encoder_hidden_states_mask is None:
            encoder_hidden_states_mask = data.get("encoder_hidden_states_mask")
            if isinstance(encoder_hidden_states_mask, tuple):
                self._negative_encoder_hidden_states_mask = encoder_hidden_states_mask[1]

        return super().prepare_inputs(data)

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: Dict[str, Union[str, Tuple[str, str]]]
    ) -> List["BlockState"]:
        self._negative_encoder_hidden_states = None
        self._negative_encoder_hidden_states_mask = None

        nag_field = input_fields.get("nag_negative_prompt_embeds", None)
        if isinstance(nag_field, str):
            self._negative_encoder_hidden_states = getattr(data, nag_field, None)

        nag_mask_field = input_fields.get("nag_negative_prompt_embeds_mask", None)
        if isinstance(nag_mask_field, str):
            self._negative_encoder_hidden_states_mask = getattr(data, nag_mask_field, None)

        if self._negative_encoder_hidden_states is None:
            encoder_hidden_states_field = input_fields.get("encoder_hidden_states", None)
            if isinstance(encoder_hidden_states_field, tuple) and encoder_hidden_states_field[1] is not None:
                self._negative_encoder_hidden_states = getattr(data, encoder_hidden_states_field[1], None)

        if self._negative_encoder_hidden_states_mask is None:
            encoder_hidden_states_mask_field = input_fields.get("encoder_hidden_states_mask", None)
            if isinstance(encoder_hidden_states_mask_field, tuple) and encoder_hidden_states_mask_field[1] is not None:
                self._negative_encoder_hidden_states_mask = getattr(data, encoder_hidden_states_mask_field[1], None)

        return super().prepare_inputs_from_block_state(data, input_fields)

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        super().prepare_models(denoiser)

        # Apply NAG only for the conditional forward (first pass).
        if not self._is_nag_enabled() or self._negative_encoder_hidden_states is None or not self.is_conditional:
            return

        original_attn_processors = _patch_qwenimage_attention_processors_for_nag(
            denoiser=denoiser,
            nag_scale=self.nag_scale,
            nag_tau=self.nag_tau,
            nag_alpha=self.nag_alpha,
        )
        if original_attn_processors is None:
            return

        self._original_attn_processors = original_attn_processors

        denoiser._nag_negative_encoder_hidden_states = self._negative_encoder_hidden_states
        denoiser._nag_negative_encoder_hidden_states_mask = self._negative_encoder_hidden_states_mask

    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        if self._original_attn_processors is not None:
            denoiser.set_attn_processor(self._original_attn_processors.copy())
        self._original_attn_processors = None

        for attr in ("_nag_negative_encoder_hidden_states", "_nag_negative_encoder_hidden_states_mask"):
            if hasattr(denoiser, attr):
                delattr(denoiser, attr)

    def _is_nag_enabled(self) -> bool:
        if not self._enabled or math.isclose(self.nag_scale, 1.0):
            return False

        if self._num_inference_steps is not None:
            start_step = int(self._start * self._num_inference_steps)
            stop_step = int(self._stop * self._num_inference_steps)
            if not (start_step <= self._step < stop_step):
                return False

        if self.nag_sigma_end is not None and self._timestep is not None:
            timestep = self._timestep
            max_t = float(timestep.detach().float().max().item())
            threshold = self.nag_sigma_end * (1000.0 if max_t > 1.5 else 1.0)
            if not bool((timestep >= threshold).all().item()):
                return False

        return True
