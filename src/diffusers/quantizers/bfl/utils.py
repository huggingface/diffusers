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

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import is_accelerate_available, is_flashinfer_available


if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module

if is_flashinfer_available():
    import flashinfer


FP8_MAX = 448.0
E2M1_MAX = 6.0
NVFP4_BLOCK = 16
# E2M1 values indexed by their 4-bit code: bit 3 is the sign, bits 2-0 the magnitude.
E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def quantize_fp8r(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`[N, K]` float weight -> E4M3 weight `[N, K]` and fp32 per-row scales `[N]`."""
    weight = weight.float()
    weight_scale = (weight.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    weight_q = (weight / weight_scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return weight_q, weight_scale


def dequantize_fp8r(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * weight_scale[:, None]


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`[N, K]` float weight -> packed E2M1 weight `[N, K/2]` (even element in the low nibble), E4M3 block scales
    `[N, K/16]` and the fp32 global scale `weight_scale_2 = amax / (448 * 6)`, following the ModelOpt NVFP4 layout."""
    weight = weight.float()
    out_features, in_features = weight.shape
    weight_scale_2 = weight.abs().amax() / (FP8_MAX * E2M1_MAX)
    blocks = weight.reshape(out_features, in_features // NVFP4_BLOCK, NVFP4_BLOCK)
    weight_scale = (blocks.abs().amax(dim=-1) / E2M1_MAX / weight_scale_2).to(torch.float8_e4m3fn)
    scale = (weight_scale.float() * weight_scale_2)[..., None]
    scaled = torch.where(scale > 0, blocks / scale, torch.zeros_like(blocks))
    magnitude = scaled.abs().clamp(max=E2M1_MAX)
    # Round to nearest even on the E2M1 grid, whose spacing is 0.5 below 2, 1 below 4 and 2 above.
    magnitude = torch.where(
        magnitude < 2,
        torch.round(magnitude * 2) / 2,
        torch.where(magnitude < 4, torch.round(magnitude), torch.round(magnitude / 2) * 2),
    )
    codes = torch.where(magnitude < 2, magnitude * 2, torch.where(magnitude < 4, magnitude + 2, magnitude / 2 + 4))
    codes = codes.to(torch.uint8) | ((scaled < 0).to(torch.uint8) << 3)
    codes = codes.reshape(out_features, in_features)
    weight_q = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return weight_q, weight_scale, weight_scale_2


def dequantize_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor) -> torch.Tensor:
    codes = torch.stack([weight & 0x0F, weight >> 4], dim=-1).reshape(weight.shape[0], -1)
    values = E2M1_LUT.to(weight.device)[codes.long()]
    scale = (weight_scale.float() * weight_scale_2).repeat_interleave(NVFP4_BLOCK, dim=1)
    return values * scale


class BFLQuantizedParameter(nn.Parameter):
    """A quantized linear weight bundled with its scales.

    Single-file checkpoint converters chunk, split and concatenate weights along the output dimension (fused QKV,
    swapped scale/shift). Bundling keeps the per-row scales aligned with the rows through those operations; any other
    shape or dtype change is refused.
    """

    def __new__(cls, data, quant_type, weight_scale, weight_scale_2=None, input_scale=None):
        self = torch.Tensor._make_subclass(cls, data, False)
        self.quant_type = quant_type
        self.weight_scale = weight_scale
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scale
        return self

    def as_tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self, False)

    def _like(self, data, weight_scale):
        return BFLQuantizedParameter(data, self.quant_type, weight_scale, self.weight_scale_2, self.input_scale)

    def __repr__(self):
        return f"BFLQuantizedParameter(quant_type={self.quant_type!r}, data={self.as_tensor()!r})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.chunk, torch.Tensor.chunk, torch.split, torch.Tensor.split):
            source = args[0]
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            if dim != 0:
                raise NotImplementedError(f"{func.__name__} along dim={dim} is not supported on quantized weights.")
            with torch._C.DisableTorchFunctionSubclass():
                weights = func(*args, **kwargs)
                scales = func(source.weight_scale, *args[1:], **kwargs)
            return tuple(source._like(w, s) for w, s in zip(weights, scales))

        if func in (torch.cat, torch.concat, torch.concatenate):
            tensors = args[0]
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            source = tensors[0]
            compatible = all(
                isinstance(t, cls)
                and t.quant_type == source.quant_type
                and all(
                    (getattr(t, name) is None and getattr(source, name) is None)
                    or torch.equal(getattr(t, name), getattr(source, name))
                    for name in ("weight_scale_2", "input_scale")
                )
                for t in tensors
            )
            if dim != 0 or not compatible:
                raise NotImplementedError(
                    "Only quantized weights of the same scheme and global scales can be concatenated along dim=0."
                )
            with torch._C.DisableTorchFunctionSubclass():
                weight = torch.cat(tensors, dim=0)
                weight_scale = torch.cat([t.weight_scale for t in tensors], dim=0)
            return source._like(weight, weight_scale)

        source = next(a for a in torch.utils._pytree.tree_leaves(args) if isinstance(a, cls))
        with torch._C.DisableTorchFunctionSubclass():
            result = func(*args, **kwargs)
        if not isinstance(result, torch.Tensor) or func in torch.overrides.get_default_nowrap_functions():
            return result
        if result.shape != source.shape or result.dtype != source.dtype:
            raise NotImplementedError(
                f"{getattr(func, '__name__', func)} changes the shape or dtype of a quantized weight, which is not supported."
            )
        weight_scale = source.weight_scale.to(result.device)
        return source._like(result, weight_scale)


class FP8RLinear(nn.Linear):
    """Linear over an E4M3 weight `[N, K]` with fp32 per-row scales `[N]`.

    On CUDA devices with FP8 support (compute capability 8.9+) activations are quantized per row and the GEMM runs
    through `torch._scaled_mm`; elsewhere the weight is dequantized to `compute_dtype`.
    """

    def __init__(self, in_features, out_features, bias=True, compute_dtype=None, device=None):
        super().__init__(in_features, out_features, bias, device=device)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))
        self.compute_dtype = compute_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            inputs.is_cuda
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
            and torch.cuda.get_device_capability(inputs.device) >= (8, 9)
        ):
            x = inputs.reshape(-1, self.in_features).float()
            x_scale = (x.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
            x_q = (x / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            output = torch._scaled_mm(
                x_q, self.weight.T, x_scale, self.weight_scale[None, :], out_dtype=torch.bfloat16, use_fast_accum=True
            )
            output = output.reshape(*inputs.shape[:-1], self.out_features).to(self.compute_dtype)
            if self.bias is not None:
                output = output + self.bias
            return output

        weight = dequantize_fp8r(self.weight, self.weight_scale).to(self.compute_dtype)
        return F.linear(inputs, weight, self.bias)


class NVFP4Linear(nn.Linear):
    """Linear over a packed E2M1 weight `[N, K/2]`, E4M3 block scales `[N, K/16]`, the fp32 global weight scale
    `weight_scale_2` and the calibrated fp32 activation scale `input_scale`.

    On Blackwell GPUs with FlashInfer installed activations are quantized to NVFP4 with the static `input_scale` and
    the GEMM runs through `flashinfer.mm_fp4`; elsewhere the weight is dequantized to `compute_dtype`.
    """

    def __init__(self, in_features, out_features, bias=True, compute_dtype=None, device=None):
        super().__init__(in_features, out_features, bias, device=device)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device), requires_grad=False
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, in_features // NVFP4_BLOCK, dtype=torch.float8_e4m3fn, device=device),
        )
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))
        self.compute_dtype = compute_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.is_cuda and is_flashinfer_available() and torch.cuda.get_device_capability(inputs.device)[0] >= 10:
            x = inputs.reshape(-1, self.in_features).to(torch.bfloat16).contiguous()
            x_q, x_scale = flashinfer.nvfp4_quantize(
                x, (1.0 / self.input_scale).reshape(1), sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
            )
            rows, cols = self.weight_scale.shape
            weight_scale = flashinfer.block_scale_interleave(self.weight_scale.view(torch.uint8).contiguous())
            weight_scale = weight_scale.reshape(-(-rows // 128) * 128, -(-cols // 4) * 4).view(torch.float8_e4m3fn)
            output = flashinfer.mm_fp4(
                x_q,
                self.weight.T,
                x_scale,
                weight_scale.T,
                alpha=(self.input_scale * self.weight_scale_2).reshape(1),
                out_dtype=torch.bfloat16,
                backend="cutlass",
            )
            output = output.reshape(*inputs.shape[:-1], self.out_features).to(self.compute_dtype)
            if self.bias is not None:
                output = output + self.bias
            return output

        weight = dequantize_nvfp4(self.weight, self.weight_scale, self.weight_scale_2).to(self.compute_dtype)
        return F.linear(inputs, weight, self.bias)


QUANTIZED_LINEARS = {"fp8r": FP8RLinear, "nvfp4": NVFP4Linear}


def _dequantize_weight(param: BFLQuantizedParameter) -> torch.Tensor:
    if param.quant_type == "nvfp4":
        return dequantize_nvfp4(param.as_tensor(), param.weight_scale, param.weight_scale_2)
    return dequantize_fp8r(param.as_tensor(), param.weight_scale)


def _replace_with_bfl_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[]):
    for name, module in model.named_children():
        if name in modules_to_not_convert:
            continue
        module_prefix = prefix + name + "."
        _replace_with_bfl_linear(module, compute_dtype, state_dict, module_prefix, modules_to_not_convert)

        weight = state_dict.get(module_prefix + "weight")
        if isinstance(module, nn.Linear) and isinstance(weight, BFLQuantizedParameter):
            with init_empty_weights():
                model._modules[name] = QUANTIZED_LINEARS[weight.quant_type](
                    module.in_features, module.out_features, module.bias is not None, compute_dtype=compute_dtype
                )
    return model


# Copied from diffusers.quantizers.bitsandbytes.utils._create_accelerate_new_hook
def _create_accelerate_new_hook(old_hook):
    r"""
    Creates a new hook based on the old hook. Use it only if you know what you are doing ! This method is a copy of:
    https://github.com/huggingface/peft/blob/748f7968f3a31ec06a1c2b0328993319ad9a150a/src/peft/utils/other.py#L245 with
    some changes
    """
    old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
    old_hook_attr = old_hook.__dict__
    filtered_old_hook_attr = {}
    old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
    for k in old_hook_attr.keys():
        if k in old_hook_init_signature.parameters:
            filtered_old_hook_attr[k] = old_hook_attr[k]
    new_hook = old_hook_cls(**filtered_old_hook_attr)
    return new_hook


def _dequantize_bfl_and_restore_linear(model):
    for name, module in model.named_children():
        if isinstance(module, (FP8RLinear, NVFP4Linear)):
            if isinstance(module, NVFP4Linear):
                weight = dequantize_nvfp4(module.weight, module.weight_scale, module.weight_scale_2)
            else:
                weight = dequantize_fp8r(module.weight, module.weight_scale)
            device = module.weight.device
            with init_empty_weights():
                new_module = nn.Linear(module.in_features, module.out_features, module.bias is not None)
            new_module.weight = nn.Parameter(weight.to(module.compute_dtype), requires_grad=False)
            if module.bias is not None:
                new_module.bias = module.bias

            if hasattr(module, "_hf_hook"):
                new_hook = _create_accelerate_new_hook(module._hf_hook)
                remove_hook_from_module(module)
                add_hook_to_module(new_module, new_hook)

            model._modules[name] = new_module.to(device)
        else:
            _dequantize_bfl_and_restore_linear(module)
    return model
