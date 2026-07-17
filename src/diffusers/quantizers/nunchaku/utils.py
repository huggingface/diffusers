from __future__ import annotations

import itertools
import math
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from ...utils import is_accelerate_available, is_kernels_available


if is_accelerate_available():
    from accelerate import init_empty_weights


_HF_KERNEL_REPO = "rootonchair/nunchaku-lite-kernels"
_HF_KERNEL_VERSION = 2


if is_kernels_available():
    from kernels import get_kernel

    ops = get_kernel(_HF_KERNEL_REPO, version=_HF_KERNEL_VERSION, trust_remote_code=True).ops
else:
    raise ImportError(
        "Loading Nunchaku checkpoints requires the Hugging Face `kernels` package. "
        "Install it with `pip install kernels`."
    )


def _gemm_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    bias: torch.Tensor | None,
    act_unsigned: bool,
    lora_scales: list[float],
    nvfp4: bool,
    alpha: torch.Tensor | None,
    wcscales: torch.Tensor | None,
) -> None:
    ops.gemm_w4a4(
        act,
        wgt,
        out,
        None,
        ascales,
        wscales,
        None,
        None,
        lora_act_in,
        lora_up,
        None,
        None,
        None,
        None,
        None,
        bias,
        None,
        None,
        None,
        act_unsigned,
        lora_scales,
        False,
        nvfp4,
        alpha,
        wcscales,
        None,
        None,
        None,
        0,
    )


def replace_with_nunchaku_linear(
    model: nn.Module, quantization_config: dict[str, Any], compute_dtype: torch.dtype
) -> int:
    num_replaced = 0
    svdq_config = quantization_config.get("svdq_w4a4")
    awq_config = quantization_config.get("awq_w4a16")

    if svdq_config is not None:
        num_replaced += _replace_quantize_targets(model, "svdq_w4a4", svdq_config, compute_dtype)
    if awq_config is not None:
        num_replaced += _replace_quantize_targets(model, "awq_w4a16", awq_config, compute_dtype)
    if num_replaced == 0:
        raise ValueError(
            "Nunchaku compact quantization config must include `svdq_w4a4.targets` or `awq_w4a16.targets`."
        )

    return num_replaced


class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        act_unsigned: bool = False,
    ):
        super().__init__()

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
        self.proj_down = nn.Parameter(
            torch.empty(in_features, rank, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.proj_up = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch_dtype, device=device), requires_grad=False
        )

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
            )
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

        ops.quantize_w4a4_act_fuse_lora(
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
        _gemm_w4a4(
            quantized_x,
            self.qweight,
            output,
            ascales,
            self.wscales,
            lora_act,
            self.proj_up,
            self.bias,
            self.act_unsigned,
            lora_scales,
            self.precision == "nvfp4",
            self.wtscale,
            self.wcscales,
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
            output = ops.awq_gemm_w4a16_g64_int32(x_flat, self.qweight, self.wscales, self.wzeros).reshape(
                output_shape
            )
        else:
            output = self._forward_gemv_chunks(x_flat, ops.gemv_awq).reshape(output_shape)

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


def _replace_quantize_targets(model: nn.Module, op: str, raw: Any, compute_dtype: torch.dtype) -> int:
    precision = raw["precision"]
    group_size = raw["group_size"]
    targets = raw["targets"]
    rank = raw["rank"] if op == "svdq_w4a4" else 0

    for target in targets:
        try:
            module = model.get_submodule(target)
        except AttributeError as exc:
            raise ValueError(f"Nunchaku target {target!r} does not exist in the model.") from exc

        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        bias = getattr(module, "bias", None)
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            raise TypeError(f"Nunchaku target {target!r} must expose integer in_features/out_features.")

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            if op == "svdq_w4a4":
                replacement = SVDQW4A4Linear(
                    in_features,
                    out_features,
                    rank=rank,
                    bias=bias is not None,
                    precision=precision,
                    group_size=group_size,
                    torch_dtype=compute_dtype,
                )
            elif op == "awq_w4a16":
                replacement = AWQW4A16Linear(
                    in_features,
                    out_features,
                    bias=bias is not None,
                    group_size=group_size,
                    torch_dtype=compute_dtype,
                )

        _set_submodule(model, target, replacement)

    return len(targets)


def _set_submodule(model: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def check_strict_state_dict_match(model: nn.Module, state_dict: dict[str, Any]) -> None:
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
