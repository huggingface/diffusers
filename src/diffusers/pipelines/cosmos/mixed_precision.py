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

Schedule defaults come from ``quantization_config.runtime.diffusion_step_policy``
on the transformer (Cosmos3-Experimental discussion #19). Distilled checkpoints
omit that policy and stay on native W8A8.
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
OverlapMode = Literal["a16", "native"]
MIXED_PRECISION_FORMATS = frozenset({"none", "fp8"})
REASONER_POLICIES = frozenset({"high_precision", "base_precision"})
OVERLAP_MODES = frozenset({"a16", "native"})
_RUNTIME_ATTRIBUTE = "_cosmos3_mixed_precision_runtime"
_SUPPORTED_POLICY_TYPE = "first_last_n"
_SUPPORTED_INDEX_SPACE = "denoising_loop_iteration"
_SUPPORTED_SCHEMA_VERSION = 1

# Official Nano/Super/Super-I2V FP8 policy (nvidia/Cosmos3-Experimental#19).
FIRST_LAST_N_FP8_POLICY = {
    "schema_version": 1,
    "type": "first_last_n",
    "index_space": "denoising_loop_iteration",
    "scope": ["transformer"],
    "default_mode": "native",
    "first_steps": {"count": 3, "mode": "a16"},
    "last_steps": {"count": 3, "mode": "a16"},
    "overlap": "a16",
    "reasoner": "a16",
}


@dataclass(frozen=True)
class Cosmos3MixedPrecisionConfig:
    """Validated first/last-step W8A16 policy for Cosmos3 FP8 denoising."""

    format: MixedPrecisionFormat = "none"
    first_steps: int = 3
    last_steps: int = 3
    reasoner_policy: ReasonerPolicy = "high_precision"
    overlap: OverlapMode = "a16"

    @classmethod
    def from_kwargs(
        cls,
        mixed_precision_format: str = "none",
        mixed_precision_first_steps: int = 3,
        mixed_precision_last_steps: int = 3,
        mixed_precision_reasoner_policy: str = "high_precision",
        mixed_precision_overlap: str = "a16",
    ) -> Cosmos3MixedPrecisionConfig:
        return cls.resolve(
            mixed_precision_format=mixed_precision_format,
            mixed_precision_first_steps=mixed_precision_first_steps,
            mixed_precision_last_steps=mixed_precision_last_steps,
            mixed_precision_reasoner_policy=mixed_precision_reasoner_policy,
            mixed_precision_overlap=mixed_precision_overlap,
        )

    @classmethod
    def resolve(
        cls,
        transformer: nn.Module | None = None,
        *,
        mixed_precision_format: str | None = None,
        mixed_precision_first_steps: int | None = None,
        mixed_precision_last_steps: int | None = None,
        mixed_precision_reasoner_policy: str | None = None,
        mixed_precision_overlap: str | None = None,
        quantization_config: dict[str, Any] | None = None,
    ) -> Cosmos3MixedPrecisionConfig:
        """Build a schedule from the checkpoint policy, with optional call-site overrides.

        ``mixed_precision_format=None`` (pipeline default) means auto: enable mixed
        precision only when the transformer declares ``diffusion_step_policy``.
        Distilled FP8 checkpoints omit that field and stay native W8A8. Pass
        ``"fp8"`` to force the schedule, or ``"none"`` to disable it.
        """
        parsed = _parsed_checkpoint_policy(transformer, quantization_config)
        format_override = _optional_lower(mixed_precision_format)
        if format_override is not None and format_override not in MIXED_PRECISION_FORMATS:
            raise ValueError(
                "mixed_precision_format must be one of "
                f"{sorted(MIXED_PRECISION_FORMATS)} or None, got {mixed_precision_format!r}"
            )
        if mixed_precision_first_steps is not None:
            mixed_precision_first_steps = _non_negative_int(
                mixed_precision_first_steps, "mixed_precision_first_steps"
            )
        if mixed_precision_last_steps is not None:
            mixed_precision_last_steps = _non_negative_int(
                mixed_precision_last_steps, "mixed_precision_last_steps"
            )
        if mixed_precision_reasoner_policy is not None:
            mixed_precision_reasoner_policy = _validated_reasoner(mixed_precision_reasoner_policy)
        if mixed_precision_overlap is not None:
            mixed_precision_overlap = _validated_overlap(mixed_precision_overlap)

        if format_override == "none":
            return cls(format="none")

        if parsed is None and format_override != "fp8":
            return cls(format="none")

        first_steps = parsed.first_steps if parsed is not None else 3
        last_steps = parsed.last_steps if parsed is not None else 3
        reasoner_policy = parsed.reasoner_policy if parsed is not None else "high_precision"
        overlap = parsed.overlap if parsed is not None else "a16"

        if mixed_precision_first_steps is not None:
            first_steps = mixed_precision_first_steps
        if mixed_precision_last_steps is not None:
            last_steps = mixed_precision_last_steps
        if mixed_precision_reasoner_policy is not None:
            reasoner_policy = mixed_precision_reasoner_policy
        if mixed_precision_overlap is not None:
            overlap = mixed_precision_overlap

        return cls(
            format="fp8",
            first_steps=first_steps,
            last_steps=last_steps,
            reasoner_policy=reasoner_policy,  # type: ignore[arg-type]
            overlap=overlap,  # type: ignore[arg-type]
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
        in_first = step_index < self.first_steps
        in_last = step_index >= num_steps - self.last_steps
        if in_first and in_last:
            return self.overlap == "a16"
        return in_first or in_last

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


@dataclass(frozen=True)
class _ParsedCheckpointPolicy:
    first_steps: int
    last_steps: int
    reasoner_policy: ReasonerPolicy
    overlap: OverlapMode


def _optional_lower(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _validated_reasoner(value: str) -> ReasonerPolicy:
    reasoner_policy = str(value).strip().lower()
    if reasoner_policy not in REASONER_POLICIES:
        raise ValueError(
            "mixed_precision_reasoner_policy must be one of "
            f"{sorted(REASONER_POLICIES)}, got {value!r}"
        )
    return reasoner_policy  # type: ignore[return-value]


def _validated_overlap(value: str) -> OverlapMode:
    overlap = str(value).strip().lower()
    if overlap not in OVERLAP_MODES:
        raise ValueError(f"overlap must be one of {sorted(OVERLAP_MODES)}, got {value!r}")
    return overlap  # type: ignore[return-value]


def _maybe_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return None


def quantization_config_from_module(module: nn.Module | None) -> dict[str, Any] | None:
    """Read ModelOpt ``quantization_config`` from a loaded transformer, if present."""
    if module is None:
        return None
    candidates: list[Any] = [getattr(module, "quantization_config", None)]
    config = getattr(module, "config", None)
    if config is not None:
        candidates.append(getattr(config, "quantization_config", None))
        if hasattr(config, "get"):
            candidates.append(config.get("quantization_config"))
    for candidate in candidates:
        mapped = _maybe_mapping(candidate)
        if mapped is not None:
            return mapped

    name_or_path = None
    if config is not None:
        name_or_path = getattr(config, "_name_or_path", None)
        if name_or_path is None and hasattr(config, "get"):
            name_or_path = config.get("_name_or_path")
    loader = getattr(type(module), "load_config", None)
    if name_or_path and callable(loader):
        try:
            raw = loader(name_or_path, local_files_only=True)
        except TypeError:
            try:
                raw = loader(name_or_path)
            except Exception:
                raw = None
        except Exception:
            raw = None
        if isinstance(raw, dict):
            mapped = _maybe_mapping(raw.get("quantization_config"))
            if mapped is not None:
                return mapped
    return None


def _parsed_checkpoint_policy(
    transformer: nn.Module | None,
    quantization_config: dict[str, Any] | None,
) -> _ParsedCheckpointPolicy | None:
    quant_config = _maybe_mapping(quantization_config) or quantization_config_from_module(transformer)
    if not quant_config:
        return None
    runtime = quant_config.get("runtime")
    if runtime is None:
        return None
    runtime_map = _maybe_mapping(runtime)
    if runtime_map is None:
        raise ValueError("quantization_config.runtime must be a mapping or null")
    policy = runtime_map.get("diffusion_step_policy")
    if policy is None:
        return None
    return parse_diffusion_step_policy(policy, quant_config)


def parse_diffusion_step_policy(
    policy: Any,
    quantization_config: dict[str, Any] | None = None,
) -> _ParsedCheckpointPolicy:
    """Validate the versioned first/last-N policy from transformer/config.json."""
    policy_map = _maybe_mapping(policy)
    if policy_map is None:
        raise ValueError("diffusion_step_policy must be a mapping")

    schema_version = policy_map.get("schema_version", _SUPPORTED_SCHEMA_VERSION)
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported diffusion_step_policy.schema_version={schema_version}; "
            f"Diffusers supports version {_SUPPORTED_SCHEMA_VERSION}"
        )
    policy_type = policy_map.get("type")
    if policy_type != _SUPPORTED_POLICY_TYPE:
        raise ValueError(
            f"Unsupported diffusion_step_policy.type={policy_type!r}; "
            f"expected {_SUPPORTED_POLICY_TYPE!r}"
        )
    index_space = policy_map.get("index_space", _SUPPORTED_INDEX_SPACE)
    if index_space != _SUPPORTED_INDEX_SPACE:
        raise ValueError(
            f"Unsupported diffusion_step_policy.index_space={index_space!r}; "
            f"expected {_SUPPORTED_INDEX_SPACE!r}"
        )

    if quantization_config:
        algo = str(
            quantization_config.get("quant_algo")
            or quantization_config.get("quant_type")
            or ""
        ).upper()
        if algo and "FP8" not in algo:
            raise ValueError(
                "Cosmos3 mixed precision in Diffusers currently supports ModelOpt FP8 "
                f"checkpoints, got quant_algo/quant_type={algo!r}"
            )

    first_steps = _window_count(policy_map.get("first_steps"), "first_steps")
    last_steps = _window_count(policy_map.get("last_steps"), "last_steps")
    overlap = _validated_overlap(str(policy_map.get("overlap", "a16")))
    reasoner = policy_map.get("reasoner", "a16")
    if reasoner == "a16":
        reasoner_policy: ReasonerPolicy = "high_precision"
    elif reasoner == "native":
        reasoner_policy = "base_precision"
    else:
        raise ValueError(f"diffusion_step_policy.reasoner must be 'a16' or 'native', got {reasoner!r}")
    return _ParsedCheckpointPolicy(
        first_steps=first_steps,
        last_steps=last_steps,
        reasoner_policy=reasoner_policy,
        overlap=overlap,
    )


def _window_count(spec: Any, name: str) -> int:
    spec_map = _maybe_mapping(spec)
    if spec_map is None:
        raise ValueError(f"diffusion_step_policy.{name} must be a mapping with count and mode")
    mode = spec_map.get("mode")
    if mode not in {"a16", "native"}:
        raise ValueError(f"diffusion_step_policy.{name}.mode must be 'a16' or 'native', got {mode!r}")
    count = _non_negative_int(spec_map.get("count"), f"diffusion_step_policy.{name}.count")
    if mode == "native":
        return 0
    return count

