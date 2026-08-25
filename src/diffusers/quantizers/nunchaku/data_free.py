"""Data-free SVDQuant quantization for the Nunchaku Lite backend.

Quantizes a bf16 linear weight at load time — no calibration data required —
into the exact packed parameter layout consumed by ``SVDQW4A4Linear``:
weight-span smoothing, a rank-``r`` SVD low-rank branch, and int4/nvfp4 group
quantization of the residual. The packing mirrors DeepCompressor's Nunchaku
W4A4 converter, so the produced tensors are indistinguishable from a
pre-quantized checkpoint's.

This module is pure PyTorch and must stay importable without the ``kernels``
package (unlike ``.utils``, which fetches the CUDA kernels at import time).
"""

from __future__ import annotations

import torch


_SMOOTH_EPS = 1e-6
_FP8_MAX = 448.0


def _ceil_divide(x: int, divisor: int) -> int:
    return (x + divisor - 1) // divisor


def _pad(
    tensor: torch.Tensor, divisor: tuple[int, ...], dim: tuple[int, ...], fill_value: float = 0.0
) -> torch.Tensor:
    shape = list(tensor.shape)
    for axis, axis_divisor in zip(dim, divisor):
        shape[axis] = _ceil_divide(shape[axis], axis_divisor) * axis_divisor
    if shape == list(tensor.shape):
        return tensor
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return result


def _fp4_e2m1_codebook(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=dtype,
        device=device,
    )


def _fp_quantize(x: torch.Tensor) -> torch.Tensor:
    """Quantize values to the nearest FP4 E2M1 codebook index."""

    codebook = _fp4_e2m1_codebook(x.device, x.dtype)
    positive = codebook[:8]
    thresholds = (positive[:-1] + positive[1:]) / 2
    codes = torch.bucketize(x.abs(), thresholds, right=False)
    negative = x.lt(0) & codes.ne(0)
    codes.add_(negative, alpha=8)
    codes.masked_fill_(~x.isfinite(), 0)
    return codes


class _NunchakuWeightPacker:
    """Pack-only subset of DeepCompressor's Nunchaku MMA weight packer (4-bit)."""

    def __init__(self, warp_n: int = 128):
        self.bits = 4
        self.comp_n = 16
        self.comp_k = 256 // self.bits
        self.insn_k = self.comp_k
        self.num_lanes = 32
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        self.warp_n = warp_n
        self.reg_k = 32 // self.bits
        self.reg_n = 1
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        self.mem_k = self.comp_k
        self.mem_n = warp_n
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        weight = _pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))
        n, k = weight.shape
        weight = weight.reshape(
            n // self.mem_n,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k // self.mem_k,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        weight = weight.bitwise_and_(0xF)
        shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
        weight = weight.bitwise_left_shift_(shift).sum(dim=-1, dtype=torch.int32)
        return weight.view(dtype=torch.int8).view(n, -1)

    def pack_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Pack a per-channel vector (smooth factor, bias) into scale layout."""

        vector = _pad(vector, divisor=(self.warp_n,), dim=(0,), fill_value=1.0)
        n = vector.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        vector = vector.reshape(n // self.warp_n, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        vector = vector.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return vector.view(-1)

    def pack_group_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """Pack per-group scales in ``[out, groups]`` layout (int4, group size 64)."""

        scale = _pad(
            scale.view(scale.shape[0], 1, -1, 1), divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=1.0
        )
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        scale = scale.reshape(n // self.warp_n, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1, n)

    def pack_micro_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """Pack FP8 per-group scales in ``[out, groups]`` layout (nvfp4, group size 16)."""

        group_fragment = self.insn_k // 16
        scale = _pad(
            scale.view(scale.shape[0], 1, -1, 1), divisor=(self.warp_n, group_fragment), dim=(0, 2), fill_value=1.0
        )
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 32
        num_s_packs = _ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        scale = scale.view(n // self.warp_n, num_s_packs, s_pack_size, 4, 8, -1, group_fragment)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        reg_n, reg_k = 1, 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        weight = _pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        if down:
            r, c = weight.shape
            r_packs, c_packs = r // pack_n, c // pack_k
            weight = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
        else:
            c, r = weight.shape
            c_packs, r_packs = c // pack_n, r // pack_k
            weight = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
        weight = weight.reshape(
            c_packs, r_packs, self.n_pack_size, self.num_n_lanes, reg_n, self.k_pack_size, self.num_k_lanes, reg_k
        )
        weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return weight.view(c, r)


def infer_data_free_targets(
    model: "torch.nn.Module",
    *,
    group_size: int,
    modules_to_not_convert: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Infer quantization targets for data-free mode from a model's structure.

    Every ``nn.Linear`` whose dimensions fit the Nunchaku packing constraints
    (``in_features``/``out_features`` multiples of 128 and ``in_features``
    divisible by ``group_size``) is selected, unless its module path contains
    one of the ``modules_to_not_convert`` substrings or the model lists it in
    ``_keep_in_fp32_modules``.
    """

    exclude = list(modules_to_not_convert) + list(getattr(model, "_keep_in_fp32_modules", None) or [])
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(pattern in name for pattern in exclude):
            continue
        if module.out_features % 128 or module.in_features % 128 or module.in_features % group_size:
            continue
        targets.append(name)
    if not targets:
        raise ValueError(
            "Could not infer any data-free quantization targets: no nn.Linear module satisfies the "
            "Nunchaku packing constraints (in/out features multiples of 128) outside the excluded modules."
        )
    return targets


def _check_packable(out_features: int, in_features: int, rank: int, group_size: int) -> None:
    if out_features % 128 != 0 or in_features % 128 != 0:
        raise ValueError(
            "Data-free Nunchaku quantization requires in_features and out_features to be multiples of 128, "
            f"got ({out_features}, {in_features})."
        )
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size}).")
    if rank % 16 != 0:
        raise ValueError(f"Low-rank branch rank must be a multiple of 16 (or 0), got {rank}.")


def _weight_span_smooth_scale(weight: torch.Tensor) -> torch.Tensor:
    """Data-free weight-span smoothing: ``s_j = 1 / absmax(W[:, j]) ** 0.5``.

    The weight is stored multiplied by ``s`` (equalizing per-channel magnitudes)
    and the kernel divides the activations by ``s`` at runtime.
    """

    span = weight.abs().amax(dim=0).clamp_min(_SMOOTH_EPS)
    scale = 1.0 / span.pow(0.5)
    scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    return scale.clamp_min(_SMOOTH_EPS)


def _group_scales(residual: torch.Tensor, group_size: int, float_point: bool) -> torch.Tensor:
    out_features, in_features = residual.shape
    groups = in_features // group_size
    max_q = 6.0 if float_point else 7.0
    return residual.view(out_features, groups, group_size).abs().amax(dim=2).clamp_min(1e-6) / max_q


def quantize_linear_data_free(
    weight: torch.Tensor,
    *,
    precision: str,
    group_size: int,
    rank: int,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Quantize one linear weight into ``SVDQW4A4Linear``'s packed parameters.

    Args:
        weight: Unquantized weight in ``[out_features, in_features]`` layout.
        precision: ``"int4"`` or ``"nvfp4"``.
        group_size: Weight quantization group size (64 for int4, 16 for nvfp4).
        rank: Low-rank branch rank (multiple of 16, or 0 to disable).
        torch_dtype: Floating-point dtype of the produced auxiliary tensors.

    Returns:
        Mapping with keys ``qweight``, ``wscales``, ``smooth_factor``,
        ``proj_down``, ``proj_up`` and, for nvfp4, ``wcscales`` and ``wtscale``.
    """

    out_features, in_features = weight.shape
    _check_packable(out_features, in_features, rank, group_size)
    packer = _NunchakuWeightPacker()
    weight = weight.to(dtype=torch.float32)

    smooth = _weight_span_smooth_scale(weight)
    smoothed = weight * smooth.view(1, -1)

    if rank > 0:
        u, s, vh = torch.linalg.svd(smoothed, full_matrices=False)
        proj_up = (u[:, :rank] * s[:rank].view(1, -1)).contiguous()
        proj_down = vh[:rank, :].contiguous()
        residual = smoothed - proj_up @ proj_down
    else:
        proj_up = smoothed.new_zeros((out_features, 0))
        proj_down = smoothed.new_zeros((0, in_features))
        residual = smoothed

    groups = in_features // group_size
    state: dict[str, torch.Tensor] = {}
    if precision == "nvfp4":
        effective = _group_scales(residual, group_size, float_point=True)
        wtscale = (effective.amax() / _FP8_MAX).clamp_min(1e-12)
        subscale = (effective / wtscale).clamp(min=0.0, max=_FP8_MAX)
        subscale = subscale.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        divisor = (subscale * wtscale).view(out_features, groups, 1)
        scaled = residual.view(out_features, groups, group_size) / divisor
        codes = _fp_quantize(scaled.reshape(out_features, in_features)).to(torch.int32)
        state["wscales"] = packer.pack_micro_scale(subscale)
        state["wcscales"] = torch.ones(out_features, dtype=torch_dtype, device=weight.device)
        state["wtscale"] = wtscale.view(1).to(dtype=torch_dtype)
    elif precision == "int4":
        scale = _group_scales(residual, group_size, float_point=False)
        scaled = residual.view(out_features, groups, group_size) / scale.view(out_features, groups, 1)
        codes = scaled.reshape(out_features, in_features).round_().clamp_(-8, 7).to(torch.int32)
        state["wscales"] = packer.pack_group_scale(scale.to(dtype=torch_dtype))
    else:
        raise ValueError(f"Unsupported precision for data-free quantization: {precision!r}")

    state["qweight"] = packer.pack_weight(codes)
    state["smooth_factor"] = packer.pack_vector(smooth.to(dtype=torch_dtype))
    # The kernel's low-rank branch consumes the unsmoothed input, so fold 1/smooth
    # into the down projection; the residual weight stays in smoothed coordinates.
    proj_down = proj_down / smooth.view(1, -1)
    state["proj_down"] = packer.pack_lowrank_weight(proj_down.to(dtype=torch_dtype), down=True)
    state["proj_up"] = packer.pack_lowrank_weight(proj_up.to(dtype=torch_dtype), down=False)
    return state


def pack_data_free_bias(bias: torch.Tensor, torch_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Pack a bias vector into the layout ``SVDQW4A4Linear.bias`` expects."""

    packer = _NunchakuWeightPacker()
    packed = packer.pack_vector(bias.to(dtype=torch.float32))
    return packed[: bias.shape[0]].to(dtype=torch_dtype)
