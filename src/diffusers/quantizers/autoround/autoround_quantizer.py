# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    is_auto_round_available,
    is_torch_available,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class AutoRoundQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for AutoRound (https://github.com/intel/auto-round).

    AutoRound is a weight-only quantization method that uses sign gradient descent to jointly optimize
    rounding values and min-max ranges for weights. It supports W4A16 (4-bit weight, 16-bit activation)
    quantization for efficient inference.

    This quantizer only supports loading pre-quantized AutoRound models. On-the-fly quantization
    (calibration) is not supported through this interface.
    """

    # AutoRound requires data calibration — we only support loading pre-quantized checkpoints.
    requires_calibration = True
    required_packages = ["auto_round"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        """
        Validates that the auto-round library (>= 0.5) is installed and captures the device_map
        for later use during model conversion.
        """
        self.device_map = kwargs.get("device_map", None)
        if not is_auto_round_available():
            raise ImportError(
                "Loading an AutoRound quantized model requires the auto-round library "
                "(`pip install 'auto-round>=0.5'`)"
            )

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map,
        keep_in_fp32_modules: list[str] = [],
        **kwargs,
    ):
        """
        Replaces target nn.Linear layers with AutoRound's quantized QuantLinear layers before
        weights are loaded from the checkpoint.

        Uses `auto_round.inference.convert_model.convert_hf_model` which:
        - Inspects the model architecture and the quantization config (bits, group_size, sym, backend).
        - Replaces eligible nn.Linear modules with the appropriate QuantLinear variant
          (the packed-weight layer that stores qweight, scales, qzeros).
        - Returns the converted model and a set of used backend names.

        `infer_target_device` resolves the device_map into a single target device string
        that AutoRound uses to select the correct kernel backend (e.g. "cuda", "cpu").
        """
        from auto_round.inference.convert_model import convert_hf_model, infer_target_device

        if self.pre_quantized:
            target_device = infer_target_device(self.device_map)
            model, used_backends = convert_hf_model(model, target_device)
            self.used_backends = used_backends

    def _process_model_after_weight_loading(self, model, **kwargs):
        """
        Finalizes the model after all quantized weights (qweight, scales, qzeros, etc.) have
        been loaded into the QuantLinear layers.

        Uses `auto_round.inference.convert_model.post_init` which:
        - Performs backend-specific finalization (e.g. repacking weights into the kernel's
          expected memory layout, moving buffers to the correct device).
        - Freezes quantized parameters (requires_grad=False).
        - Prepares the model for inference.

        Raises ValueError if the model is not pre-quantized, since AutoRound does not support
        on-the-fly quantization through this loading path.
        """
        if self.pre_quantized:
            from auto_round.inference.convert_model import post_init

            post_init(model, self.used_backends)
        else:
            raise ValueError(
                "AutoRound quantizer in diffusers only supports pre-quantized models. "
                "Please provide a model that has already been quantized with AutoRound."
            )
        return model

    @property
    def is_trainable(self) -> bool:
        """AutoRound W4A16 pre-quantized models do not support training."""
        return False

    @property
    def is_serializable(self):
        """AutoRound quantized models can be serialized (the quantization config may be
        updated by the backend, e.g. for GPTQ/AWQ-compatible formats)."""
        return True

