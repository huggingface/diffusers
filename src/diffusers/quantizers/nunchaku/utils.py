from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


_SCHEMA = "nunchaku_lite.runtime_manifest"
_SUPPORTED_VERSION = 1
_SUPPORTED_FORMAT_VERSION = 1
_SUPPORTED_OPS = {"svdq_w4a4", "awq_w4a16"}
_SUPPORTED_PRECISIONS = {"int4", "fp4"}
_HF_KERNEL_REPO = "rootonchair/nunchaku-lite-kernels"
_HF_KERNEL_VERSION = 1


_ops = None


def _get_ops():
    global _ops
    if _ops is None:
        from kernels import get_kernel

        _ops = get_kernel(_HF_KERNEL_REPO, version=_HF_KERNEL_VERSION, trust_remote_code=True).ops
    return _ops


@dataclass(frozen=True)
class RuntimeManifestTarget:
    checkpoint_prefix: str
    source_modules: tuple[str, ...]
    kind: str
    nunchaku_op: str
    precision: str
    group_size: int
    rank: int
    has_bias: bool
    op_options: dict[str, Any] = field(default_factory=dict)
    activation: dict[str, Any] = field(default_factory=dict)

    @property
    def runtime_precision(self) -> str:
        return "nvfp4" if self.precision == "fp4" else self.precision


@dataclass(frozen=True)
class RuntimeManifest:
    schema: str
    version: int
    component: str
    nunchaku_format_version: int
    requirements: dict[str, Any]
    structural_patches: tuple[dict[str, Any], ...]
    targets: tuple[RuntimeManifestTarget, ...]


def parse_runtime_manifest(quantization_config: dict[str, Any]) -> RuntimeManifest:
    raw = quantization_config.get("runtime_manifest")
    if raw is None:
        raise ValueError("Nunchaku checkpoints must include `quantization_config.runtime_manifest` metadata.")
    if not isinstance(raw, dict):
        raise ValueError("`quantization_config.runtime_manifest` must be a JSON object.")

    schema = _required(raw, "schema", str)
    if schema != _SCHEMA:
        raise ValueError(f"Unsupported Nunchaku runtime manifest schema {schema!r}; expected {_SCHEMA!r}.")

    version = _required(raw, "version", int)
    if version != _SUPPORTED_VERSION:
        raise ValueError(
            f"Unsupported Nunchaku runtime manifest version {version}; expected {_SUPPORTED_VERSION}."
        )

    nunchaku_format_version = _required(raw, "nunchaku_format_version", int)
    if nunchaku_format_version != _SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            "Unsupported Nunchaku runtime manifest format version "
            f"{nunchaku_format_version}; expected {_SUPPORTED_FORMAT_VERSION}."
        )

    structural_patches = _required(raw, "structural_patches", list)
    if structural_patches:
        raise ValueError("Nunchaku runtime manifest structural patches are not supported by Diffusers yet.")

    targets = _required(raw, "targets", list)
    if not targets:
        raise ValueError("Nunchaku runtime manifest must contain at least one target.")

    return RuntimeManifest(
        schema=schema,
        version=version,
        component=_required(raw, "component", str),
        nunchaku_format_version=nunchaku_format_version,
        requirements=_required(raw, "requirements", dict),
        structural_patches=tuple(structural_patches),
        targets=tuple(_parse_target(index, target) for index, target in enumerate(targets)),
    )


def parse_compact_quantization_config(model: nn.Module, quantization_config: dict[str, Any]) -> RuntimeManifest:
    targets = []
    svdq_config = quantization_config.get("svdq_w4a4")
    awq_config = quantization_config.get("awq_w4a16")

    if svdq_config is not None:
        targets.extend(_parse_compact_targets(model, "svdq_w4a4", svdq_config))
    if awq_config is not None:
        targets.extend(_parse_compact_targets(model, "awq_w4a16", awq_config))
    if not targets:
        raise ValueError(
            "Nunchaku compact quantization config must include `svdq_w4a4.targets` or `awq_w4a16.targets`."
        )

    return RuntimeManifest(
        schema=_SCHEMA,
        version=_SUPPORTED_VERSION,
        component="transformer",
        nunchaku_format_version=_SUPPORTED_FORMAT_VERSION,
        requirements={},
        structural_patches=(),
        targets=tuple(targets),
    )


def replace_with_nunchaku_linear(model: nn.Module, manifest: RuntimeManifest, compute_dtype: torch.dtype) -> None:
    for target in manifest.targets:
        _replace_target(model, target, compute_dtype)


class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        act_unsigned: bool = False,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        if precision == "nvfp4":
            group_size = 16
        elif precision == "int4":
            group_size = 64
        else:
            raise ValueError(f"Invalid Nunchaku Lite SVDQ precision: {precision!r}.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.precision = precision
        self.group_size = group_size
        self.torch_dtype = torch_dtype
        self.act_unsigned = act_unsigned

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=False)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(
                in_features // group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device), requires_grad=False)
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device), requires_grad=False)

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False)
            self.wtscale = nn.Parameter(torch.ones(1, dtype=torch_dtype, device=device), requires_grad=False)
        else:
            self.wcscales = None
            self.wtscale = None

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        original_shape = x.shape
        channels = x.shape[-1]
        x = x.reshape(-1, channels)
        rows = x.shape[0]
        if output is None:
            output = torch.empty(rows, self.out_features, dtype=self.torch_dtype, device=x.device)

        pad_size = 256
        batch_size_pad = math.ceil(x.shape[0] / pad_size) * pad_size
        quantized_x = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=x.device)
        if self.precision == "nvfp4":
            ascales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=x.device)
        else:
            ascales = torch.empty(channels // 64, batch_size_pad, dtype=x.dtype, device=x.device)
        lora_act = torch.empty(batch_size_pad, self.rank, dtype=torch.float32, device=x.device)

        _get_ops().quantize_w4a4_act_fuse_lora(
            x,
            quantized_x,
            ascales,
            self.proj_down,
            lora_act,
            self.smooth_factor,
            False,
            self.precision == "nvfp4",
        )
        lora_scales = [1.0] * math.ceil(self.rank / 16)
        _get_ops().gemm_w4a4(
            quantized_x,
            self.qweight,
            output,
            None,
            ascales,
            self.wscales,
            None,
            None,
            lora_act,
            self.proj_up,
            None,
            None,
            None,
            None,
            None,
            self.bias,
            None,
            None,
            None,
            self.act_unsigned,
            lora_scales,
            False,
            self.precision == "nvfp4",
            self.wtscale,
            self.wcscales,
            None,
            None,
            None,
            0,
        )
        return output.reshape(*original_shape[:-1], self.out_features)


class AWQW4A16Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if group_size != 64:
            raise ValueError(f"Nunchaku AWQ W4A16 currently supports group_size=64 only, got {group_size}.")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(out_features // 4, in_features // 2, dtype=torch.int32, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=False)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // group_size, out_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.wzeros = nn.Parameter(
            torch.empty(in_features // group_size, out_features, dtype=torch_dtype, device=device), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"AWQW4A16Linear expected input last dimension {self.in_features}, got shape {tuple(x.shape)}."
            )

        output_shape = (*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, self.in_features).contiguous()
        if x_flat.shape[0] == 0:
            output = x.new_empty(output_shape)
        elif self._use_gemm(x_flat.shape[0]):
            output = _get_ops().awq_gemm_w4a16_g64_int32(x_flat, self.qweight, self.wscales, self.wzeros).reshape(
                output_shape
            )
        else:
            output = self._forward_gemv_chunks(x_flat, _get_ops().gemv_awq).reshape(output_shape)

        if self.bias is not None:
            output = output + self.bias.view([1] * (output.ndim - 1) + [-1])
        return output

    def _use_gemm(self, rows: int) -> bool:
        return rows >= 16 and self.in_features % 64 == 0 and self.out_features % 128 == 0

    def _forward_gemv_chunks(self, x_flat: torch.Tensor, gemv) -> torch.Tensor:
        outputs = []
        for start in range(0, x_flat.shape[0], 8):
            chunk = x_flat[start : start + 8]
            outputs.append(
                gemv(
                    chunk,
                    self.qweight,
                    self.wscales,
                    self.wzeros,
                    chunk.shape[0],
                    self.out_features,
                    self.in_features,
                    64,
                )
            )
        return torch.cat(outputs, dim=0)


def _parse_target(index: int, raw: Any) -> RuntimeManifestTarget:
    if not isinstance(raw, dict):
        raise ValueError(f"Nunchaku runtime manifest target {index} must be a JSON object.")

    checkpoint_prefix = _required(raw, "checkpoint_prefix", str)
    kind = _required(raw, "kind", str)
    nunchaku_op = _required(raw, "nunchaku_op", str)
    precision = _required(raw, "precision", str)
    group_size = _required(raw, "group_size", int)
    rank = _required(raw, "rank", int)
    has_bias = _required(raw, "has_bias", bool)

    if kind != "linear":
        raise ValueError(f"Unsupported Nunchaku target kind {kind!r} at {checkpoint_prefix!r}.")
    if nunchaku_op not in _SUPPORTED_OPS:
        raise ValueError(f"Unsupported Nunchaku op {nunchaku_op!r} at {checkpoint_prefix!r}.")
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported Nunchaku precision {precision!r} at {checkpoint_prefix!r}.")
    if group_size <= 0:
        raise ValueError(f"Nunchaku target {checkpoint_prefix!r} must have positive group_size.")
    if rank < 0:
        raise ValueError(f"Nunchaku target {checkpoint_prefix!r} must have non-negative rank.")

    source_modules = _required(raw, "source_modules", list)
    if not all(isinstance(item, str) for item in source_modules):
        raise ValueError(f"Nunchaku target {checkpoint_prefix!r} source_modules must be strings.")

    return RuntimeManifestTarget(
        checkpoint_prefix=checkpoint_prefix,
        source_modules=tuple(source_modules),
        kind=kind,
        nunchaku_op=nunchaku_op,
        precision=precision,
        group_size=group_size,
        rank=rank,
        has_bias=has_bias,
        op_options=dict(_required(raw, "op_options", dict)),
        activation=dict(_required(raw, "activation", dict)),
    )


def _parse_compact_targets(model: nn.Module, op: str, raw: Any) -> list[RuntimeManifestTarget]:
    if not isinstance(raw, dict):
        raise ValueError(f"Nunchaku compact config section {op!r} must be a JSON object.")
    if op not in _SUPPORTED_OPS:
        raise ValueError(f"Unsupported Nunchaku op {op!r}.")

    precision = _required(raw, "precision", str, context=op)
    group_size = _required(raw, "group_size", int, context=op)
    targets = _required(raw, "targets", list, context=op)

    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported Nunchaku precision {precision!r} for {op!r}.")
    if group_size <= 0:
        raise ValueError(f"Nunchaku compact config section {op!r} must have positive group_size.")
    if not targets:
        raise ValueError(f"Nunchaku compact config section {op!r} must contain at least one target.")
    if not all(isinstance(target, str) for target in targets):
        raise ValueError(f"Nunchaku compact config section {op!r} targets must be strings.")

    if op == "svdq_w4a4":
        rank = _required(raw, "rank", int, context=op)
        if rank < 0:
            raise ValueError(f"Nunchaku compact config section {op!r} must have non-negative rank.")
    else:
        rank = 0

    parsed_targets = []
    for target in targets:
        try:
            module = model.get_submodule(target)
        except AttributeError as exc:
            raise ValueError(f"Nunchaku target {target!r} does not exist in the model.") from exc

        bias = getattr(module, "bias", None)
        parsed_targets.append(
            RuntimeManifestTarget(
                checkpoint_prefix=target,
                source_modules=(target,),
                kind="linear",
                nunchaku_op=op,
                precision=precision,
                group_size=group_size,
                rank=rank,
                has_bias=bias is not None,
                op_options={},
                activation={},
            )
        )

    return parsed_targets


def _replace_target(model: nn.Module, target: RuntimeManifestTarget, compute_dtype: torch.dtype) -> None:
    for source_module in target.source_modules:
        try:
            model.get_submodule(source_module)
        except AttributeError as exc:
            raise ValueError(
                f"Nunchaku target {target.checkpoint_prefix!r} source module {source_module!r} does not exist."
            ) from exc

    try:
        module = model.get_submodule(target.checkpoint_prefix)
    except AttributeError as exc:
        raise ValueError(f"Nunchaku target {target.checkpoint_prefix!r} does not exist in the model.") from exc

    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        raise TypeError(
            f"Nunchaku target {target.checkpoint_prefix!r} must expose integer in_features/out_features."
        )

    if target.nunchaku_op == "svdq_w4a4":
        expected_group_size = 16 if target.precision == "fp4" else 64
        if target.group_size != expected_group_size:
            raise ValueError(
                f"Nunchaku SVDQ target {target.checkpoint_prefix!r} with precision={target.precision!r} "
                f"requires group_size={expected_group_size}, got {target.group_size}."
            )
        replacement = SVDQW4A4Linear(
            in_features,
            out_features,
            rank=target.rank,
            bias=target.has_bias,
            precision=target.runtime_precision,
            torch_dtype=compute_dtype,
            device=_module_device(module),
            act_unsigned=target.op_options.get("act_unsigned", False),
        )
    elif target.nunchaku_op == "awq_w4a16":
        if target.precision != "int4":
            raise ValueError(f"Nunchaku AWQ target {target.checkpoint_prefix!r} requires precision='int4'.")
        replacement = AWQW4A16Linear(
            in_features,
            out_features,
            bias=target.has_bias,
            group_size=target.group_size,
            torch_dtype=compute_dtype,
            device=_module_device(module),
        )
    else:
        raise ValueError(f"Unsupported Nunchaku op {target.nunchaku_op!r}.")

    _set_submodule(model, target.checkpoint_prefix, replacement)


def _set_submodule(model: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(recurse=False), None)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


def check_strict_state_dict_match(model: nn.Module, state_dict: dict[str, Any]) -> None:
    import itertools

    expected_keys = {n for n, _ in itertools.chain(model.named_parameters(), model.named_buffers())}
    loaded_keys = set(state_dict.keys())
    missing_keys = sorted(expected_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - expected_keys)
    if missing_keys or unexpected_keys:
        message = "Nunchaku checkpoint keys must exactly match the patched model state dict."
        if missing_keys:
            message += f" Missing keys: {missing_keys[:10]}"
            if len(missing_keys) > 10:
                message += f" and {len(missing_keys) - 10} more"
            message += "."
        if unexpected_keys:
            message += f" Unexpected keys: {unexpected_keys[:10]}"
            if len(unexpected_keys) > 10:
                message += f" and {len(unexpected_keys) - 10} more"
            message += "."
        raise ValueError(message)


def _required(raw: dict[str, Any], key: str, expected_type: type, context: str = "") -> Any:
    if context:
        message_prefix = f"Nunchaku compact config section {context!r}"
    else:
        message_prefix = "Nunchaku runtime manifest"
    if key not in raw:
        raise ValueError(f"{message_prefix} is missing required field {key!r}.")
    value = raw[key]
    if not isinstance(value, expected_type):
        raise ValueError(f"{message_prefix} field {key!r} must be {expected_type.__name__}.")
    return value
