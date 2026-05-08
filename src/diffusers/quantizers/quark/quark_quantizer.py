# Copyright 2025 - 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusers quantizer for AMD Quark.

Two load paths are supported:

* **Pre-quantized.**  The model on the Hub already carries a
  ``quantization_config`` block in ``config.json`` and the state dict is
  in Quark's QParams format.  ``_map_to_quark`` swaps ``nn.Linear`` /
  ``nn.Conv2d`` for ``QParamsLinear`` so the state-dict load populates
  the right modules.  ``QParamsLinear.forward`` dispatches to the FP8
  ``scaled_mm`` kernel when applicable, giving native-inference
  performance without any extra plumbing.  This mirrors the Quark
  integration in 🤗 Transformers.

* **On-the-fly weight-only.**  The user passes
  ``quantization_config=QuarkConfig(...)`` against a vanilla fp16/bf16
  model.  Weights load normally and Quark applies weight-only
  quantization after loading via ``ModelQuantizer``.  No calibration
  data is required.  The current implementation produces fake-quant
  modules (slower than the pre-quantized path).  See the
  NATIVE-INFERENCE SLOT-IN comments below for the upgrade path.

On-the-fly **activation** quantization (SmoothQuant, SVDQuant w4a4,
FP8 with calibrated activations, etc.) is not supported through
``from_pretrained``: calibration data for diffusion models depends on
running the full pipeline, which is not naturally available at load
time.  Use ``quark.torch.utils.diffusers.get_calib_dataloader`` +
``quark.torch.ModelQuantizer`` to quantize first, then save and reload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...utils import (
    get_module_from_name,
    is_quark_available,
    is_torch_available,
    logging,
)
from ..base import DiffusersQuantizer


if TYPE_CHECKING:
    from ...models.modeling_utils import ModelMixin
    from ..quantization_config import QuarkConfig

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def _qconfig_needs_activation_calibration(qconfig: Any) -> bool:
    """True if any layer in *qconfig* has an input or output quantizer.

    Used to gate the on-the-fly load path: weight-only configs are
    supported without calibration data; activation-quantized configs
    require pre-quantization with a calibration dataloader.
    """
    layer_configs = [qconfig.global_quant_config]
    layer_configs.extend((qconfig.layer_type_quant_config or {}).values())
    layer_configs.extend((qconfig.layer_quant_config or {}).values())
    for layer in layer_configs:
        if getattr(layer, "input_tensors", None) is not None:
            return True
        if getattr(layer, "output_tensors", None) is not None:
            return True
    return False


class QuarkDiffusersQuantizer(DiffusersQuantizer):
    """Diffusers quantizer for [Quark](https://quark.docs.amd.com/latest/)."""

    requires_calibration = False
    required_packages = ["amd-quark"]
    quantization_config: "QuarkConfig"

    def __init__(self, quantization_config: "QuarkConfig", **kwargs: Any) -> None:
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args: Any, **kwargs: Any) -> None:
        if not is_quark_available():
            raise ImportError(
                "Loading a Quark-quantized diffusion model requires the `amd-quark` library "
                "but it was not found in the environment.  Install it with `pip install amd-quark` "
                "or refer to https://quark.docs.amd.com/latest/install.html."
            )

    def _process_model_before_weight_loading(
        self,
        model: "ModelMixin",
        device_map: dict[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> None:
        if self.pre_quantized:
            # Mirror the Transformers Quark integration: swap nn.Linear
            # / nn.Conv2d for QParamsLinear here, so the saved QParams-
            # format state dict loads directly into the right modules
            # and ``QParamsLinear.forward`` can dispatch to native
            # kernels (FP8 scaled_mm today; MXFP4 / NVFP4 once the
            # native-inference roadmap lands).
            from quark.torch.export.api import _map_to_quark

            _map_to_quark(
                model,
                self.quantization_config.quant_config,
                pack_method=self.quantization_config.json_export_config.pack_method,
                custom_mode=self.quantization_config.custom_mode,
            )
        else:
            # On-the-fly: weights load into the original nn.Linear /
            # nn.Conv2d and we quantize after loading.  Reject configs
            # that would need calibration data we cannot produce at
            # load time.
            if _qconfig_needs_activation_calibration(self.quantization_config.quant_config):
                raise NotImplementedError(
                    "Quark on-the-fly quantization at load time is currently "
                    "limited to weight-only configurations.  The provided QConfig "
                    "has activation (input or output) quantizers, which require a "
                    "calibration dataloader.  Please quantize the model with "
                    "`quark.torch.ModelQuantizer` first using "
                    "`quark.torch.utils.diffusers.get_calib_dataloader` to collect "
                    "calibration data, save with `quark.torch.export_safetensors`, "
                    "then reload."
                )

        model.config.quantization_config = self.quantization_config  # type: ignore[attr-defined]

    def check_if_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        if not self.pre_quantized:
            # Fp16/bf16 weights load through the normal Diffusers path;
            # quantization happens in _process_model_after_weight_loading.
            return False

        # `QparamsOperator` is the marker base class shared by `QParamsLinear`
        # (and `QParamsLinearWithRotation`) inside amd-quark>=0.10.  In older
        # quark releases this lived under a different name; if upstream renames
        # it again, broaden the fallback below rather than tightening the import.
        from quark.torch.export.nn.modules.qparamslinear import QparamsOperator

        module, _ = get_module_from_name(model, param_name)
        return isinstance(module, QparamsOperator)

    def create_quantized_param(
        self,
        model: "ModelMixin",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
        unexpected_keys: list[str],
        **kwargs: Any,
    ) -> None:
        module, tensor_name = get_module_from_name(model, param_name)
        new_value = param_value.to(device=target_device)
        if tensor_name in dict(module.named_parameters(recurse=False)):
            module._parameters[tensor_name] = torch.nn.Parameter(new_value, requires_grad=False)
        else:
            module._buffers[tensor_name] = new_value

    def _process_model_after_weight_loading(self, model: "ModelMixin", **kwargs: Any) -> "ModelMixin":
        if not self.pre_quantized:
            # On-the-fly weight-only: quantize now that fp16/bf16
            # weights are populated.  ``dataloader=None`` is fine because
            # we already rejected activation-quantized configs.
            #
            # NATIVE-INFERENCE SLOT-IN: this branch currently lands in
            # the FrozenFakeQuantize slow path.  Once the on-the-fly
            # path runs ``ModelPostProcessor`` (or equivalent) to convert
            # QuantLinear -> QParamsLinear here, the same FP8 / MXFP4 /
            # NVFP4 kernels available on the pre-quantized branch will
            # apply automatically.  See Quark PR #4841 for the
            # NativeInferenceLinear extension.
            from quark.torch import ModelQuantizer

            qconfig = self.quantization_config.quant_config
            ModelQuantizer(qconfig).quantize_model(model, dataloader=None)
            ModelQuantizer.freeze(model, quantize=True)
        else:
            # Pre-quantized: modules are already QParamsLinear and the
            # state dict has populated their weights.  Nothing else to
            # do -- ``QParamsLinear.forward`` dispatches to native
            # kernels when applicable.

            # Non-persistent buffers absent from the saved state dict
            # remain on the meta device under low_cpu_mem_usage.
            # Surface this loudly rather than silently zero-filling.
            for module in model.modules():
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf.device.type == "meta":
                        raise RuntimeError(
                            f"Buffer '{name}' in {type(module).__name__} is still on the meta device "
                            f"after weight loading.  Ensure non-persistent buffers are properly "
                            f"initialized (e.g. via no_init_weights / init_empty_weights decorators)."
                        )

        return model

    @property
    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False
