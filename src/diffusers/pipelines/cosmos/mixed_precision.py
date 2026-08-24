# Copyright 2026 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

"""Mixed W8A8/W8A16 denoising for serialized Cosmos3 ModelOpt FP8 checkpoints.

The checkpoint's restored ModelOpt forward is the native W8A8 path. W8A16
bypasses that forward, dequantizes the FP8 weight to the activation dtype, and
uses :func:`torch.nn.functional.linear`. This matches the uncached strategy in
vLLM-Omni (vllm-project/vllm-omni#6560).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


MixedPrecisionFormat = Literal["none", "fp8"]
ReasonerPolicy = Literal["high_precision", "base_precision"]
MIXED_PRECISION_FORMATS = frozenset({"none", "fp8"})
REASONER_POLICIES = frozenset({"high_precision", "base_precision"})
_RUNTIME_ATTRIBUTE = "_cosmos3_mixed_precision_runtime"


@dataclass(frozen=True)
class Cosmos3MixedPrecisionConfig:
    """Validated first/last-step W8A16 policy for Cosmos3 FP8 denoising."""

    format: MixedPrecisionFormat = "none"
    first_steps: int = 3
    last_steps: int = 3
    reasoner_policy: ReasonerPolicy = "high_precision"

    @classmethod
    def from_kwargs(
        cls,
        mixed_precision_format: str = "none",
        mixed_precision_first_steps: int = 3,
        mixed_precision_last_steps: int = 3,
        mixed_precision_reasoner_policy: str = "high_precision",
    ) -> Cosmos3MixedPrecisionConfig:
        precision_format = str(mixed_precision_format).strip().lower()
        if precision_format not in MIXED_PRECISION_FORMATS:
            raise ValueError(
                "mixed_precision_format must be one of "
                f"{sorted(MIXED_PRECISION_FORMATS)}, got {mixed_precision_format!r}"
            )
        reasoner_policy = str(mixed_precision_reasoner_policy).strip().lower()
        if reasoner_policy not in REASONER_POLICIES:
            raise ValueError(
                "mixed_precision_reasoner_policy must be one of "
                f"{sorted(REASONER_POLICIES)}, got {mixed_precision_reasoner_policy!r}"
            )
        return cls(
            format=precision_format,  # type: ignore[arg-type]
            first_steps=_non_negative_int(mixed_precision_first_steps, "mixed_precision_first_steps"),
            last_steps=_non_negative_int(mixed_precision_last_steps, "mixed_precision_last_steps"),
            reasoner_policy=reasoner_policy,  # type: ignore[arg-type]
        )

    @property
    def enabled(self) -> bool:
        return self.format != "none"

    def use_high_precision(self, step_index: int, num_steps: int) -> bool:
        """Return True when this scheduler step should run W8A16 (no activation quant)."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
        # Match vLLM-Omni: a 1-step request keeps the checkpoint's base (W8A8) path.
        if num_steps == 1:
            return False
        return step_index < self.first_steps or step_index >= num_steps - self.last_steps

    def precision_name(self, step_index: int, num_steps: int) -> str:
        if not self.enabled:
            return "base"
        return "W8A16" if self.use_high_precision(step_index, num_steps) else "W8A8"


class Cosmos3MixedPrecisionRuntime:
    """Own the per-step selection and wrapped ModelOpt linear inventory."""

    def __init__(self, transformer: nn.Module, config: Cosmos3MixedPrecisionConfig) -> None:
        self.transformer = transformer
        self.config = config
        self.active = False
        self.generation_high_precision = False
        self.installed_counts = {"reasoner": 0, "generation": 0}
        self._install()

    def _install(self) -> None:
        inventory = []
        for name, layer in self.transformer.named_modules():
            path = _classify_cosmos3_linear(name)
            if path is None or not _is_modelopt_fp8_linear(layer):
                continue
            _validate_modelopt_fp8_linear(name, layer)
            inventory.append((name, layer, path))
            self.installed_counts[path] += 1

        missing = [path for path, count in self.installed_counts.items() if count == 0]
        if missing:
            raise ValueError(
                "Cosmos3 mixed precision found no compatible serialized ModelOpt FP8 linears under "
                f"{missing}; discovered counts={self.installed_counts}"
            )

        for name, layer, path in inventory:
            original_forward = layer.forward

            def mixed_forward(layer_self, inputs, *args, __name=name, __path=path, __base=original_forward, **kwargs):
                if not self.active or not self.use_high_precision(__path):
                    return __base(inputs, *args, **kwargs)
                if args or kwargs:
                    raise TypeError(f"{__name} W8A16 only supports the standard Linear forward(input) signature")
                return _w8a16_linear(layer_self, inputs, __name)

            layer.forward = MethodType(mixed_forward, layer)

    def use_high_precision(self, path: str) -> bool:
        if path == "reasoner":
            return self.config.reasoner_policy == "high_precision"
        return self.generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> str:
        self.active = True
        self.generation_high_precision = self.config.use_high_precision(step_index, num_steps)
        return "W8A16" if self.generation_high_precision else "W8A8"

    def reset(self) -> None:
        self.active = False
        self.generation_high_precision = False


def _classify_cosmos3_linear(name: str) -> str | None:
    """Classify the two MoT paths in Diffusers' fused Cosmos3 decoder layer."""
    if ".mlp_moe_gen." in name or any(
        name.endswith(suffix)
        for suffix in (
            ".self_attn.to_q",
            ".self_attn.to_k",
            ".self_attn.to_v",
            ".self_attn.to_out",
        )
    ):
        return "generation"
    if ".mlp." in name or any(
        name.endswith(suffix)
        for suffix in (
            ".self_attn.add_q_proj",
            ".self_attn.add_k_proj",
            ".self_attn.add_v_proj",
            ".self_attn.to_add_out",
        )
    ):
        return "reasoner"
    return None


def _is_modelopt_fp8_linear(layer: nn.Module) -> bool:
    weight = getattr(layer, "weight", None)
    weight_quantizer = getattr(layer, "weight_quantizer", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.dtype == torch.float8_e4m3fn
        and weight_quantizer is not None
        and hasattr(layer, "_should_run_real_quant_gemm")
    )


def _validate_modelopt_fp8_linear(name: str, layer: nn.Module) -> None:
    weight = layer.weight
    input_quantizer = getattr(layer, "input_quantizer", None)
    if input_quantizer is not None and not input_quantizer.is_enabled:
        raise ValueError(f"{name} has a disabled input quantizer and therefore is not a native W8A8 linear")
    if not layer.weight_quantizer.is_enabled:
        raise ValueError(f"{name} has a disabled weight quantizer and therefore is not a native W8A8 linear")
    scale = getattr(layer.weight_quantizer, "_scale", None)
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
        raise ValueError(f"{name} requires one tensorwise ModelOpt FP8 weight scale")
    if weight.ndim != 2:
        raise ValueError(f"{name} expected a 2D FP8 weight, got shape {tuple(weight.shape)}")
    if hasattr(layer, "pre_quant_scale"):
        raise ValueError(f"{name} uses SmoothQuant pre_quant_scale, which mixed W8A8/W8A16 does not support")


def _w8a16_linear(layer: nn.Module, inputs: torch.Tensor, name: str) -> torch.Tensor:
    if inputs.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"{name} W8A16 requires BF16/FP16 activations, got {inputs.dtype}")
    scale = layer.weight_quantizer._scale.to(device=layer.weight.device, dtype=inputs.dtype)
    dense_weight = layer.weight.to(dtype=inputs.dtype) * scale
    return F.linear(inputs, dense_weight, layer.bias)


def _get_or_create_runtime(module: nn.Module, config: Cosmos3MixedPrecisionConfig) -> Cosmos3MixedPrecisionRuntime:
    runtime = getattr(module, _RUNTIME_ATTRIBUTE, None)
    if runtime is None:
        runtime = Cosmos3MixedPrecisionRuntime(module, config)
        setattr(module, _RUNTIME_ATTRIBUTE, runtime)
    elif runtime.config != config:
        runtime.config = config
    return runtime


def apply_cosmos3_mixed_precision_step(
    module: nn.Module,
    config: Cosmos3MixedPrecisionConfig,
    step_index: int,
    num_steps: int,
    trace: list[str] | None = None,
) -> str:
    """Select W8A8 vs W8A16 for one scheduler step. No-op when mixed precision is disabled."""
    name = config.precision_name(step_index, num_steps)
    if trace is not None:
        trace.append(name)
    if not config.enabled:
        return name
    runtime = _get_or_create_runtime(module, config)
    selected = runtime.set_step(step_index, num_steps)
    if selected != name:
        raise RuntimeError(f"Mixed-precision schedule mismatch: config={name}, runtime={selected}")
    return selected


def reset_cosmos3_mixed_precision(module: nn.Module, config: Cosmos3MixedPrecisionConfig) -> None:
    """Return installed wrappers to the checkpoint's native W8A8 path."""
    if not config.enabled:
        return
    runtime = getattr(module, _RUNTIME_ATTRIBUTE, None)
    if runtime is not None:
        runtime.reset()


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    return value
