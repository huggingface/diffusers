# Copyright 2025 The HuggingFace Team and City96. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import inspect
import os
from contextlib import nullcontext

import gguf
import torch
import torch.nn as nn

from ...utils import is_accelerate_available, is_kernels_available


if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module


can_use_cuda_kernels = (
    os.getenv("DIFFUSERS_GGUF_CUDA_KERNELS", "false").lower() in ["1", "true", "yes"]
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 7
)
if can_use_cuda_kernels and is_kernels_available():
    from kernels import get_kernel

    ops = get_kernel("Isotr0py/ggml")
else:
    ops = None

UNQUANTIZED_TYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
STANDARD_QUANT_TYPES = {
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q4_1,
    gguf.GGMLQuantizationType.Q5_0,
    gguf.GGMLQuantizationType.Q5_1,
    gguf.GGMLQuantizationType.Q8_0,
    gguf.GGMLQuantizationType.Q8_1,
}
KQUANT_TYPES = {
    gguf.GGMLQuantizationType.Q2_K,
    gguf.GGMLQuantizationType.Q3_K,
    gguf.GGMLQuantizationType.Q4_K,
    gguf.GGMLQuantizationType.Q5_K,
    gguf.GGMLQuantizationType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    gguf.GGMLQuantizationType.IQ1_M,
    gguf.GGMLQuantizationType.IQ1_S,
    gguf.GGMLQuantizationType.IQ2_XXS,
    gguf.GGMLQuantizationType.IQ2_XS,
    gguf.GGMLQuantizationType.IQ2_S,
    gguf.GGMLQuantizationType.IQ3_XXS,
    gguf.GGMLQuantizationType.IQ3_S,
    gguf.GGMLQuantizationType.IQ4_XS,
    gguf.GGMLQuantizationType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T

    # TODO(Isotr0py): GGUF's MMQ and MMVQ implementation are designed for
    # contiguous batching and inefficient with diffusers' batching,
    # so we disabled it now.

    # elif qweight_type in MMVQ_QUANT_TYPES:
    #     y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # elif qweight_type in MMQ_QUANT_TYPES:
    #     y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])

    # If there is no available MMQ kernel, fallback to dequantize
    if qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape)
        y = x @ weight.to(x.dtype).T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = gguf.GGMLQuantizationType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y.as_tensor()


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


def _replace_with_gguf_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[]):
    def _should_convert_to_gguf(state_dict, prefix):
        weight_key = prefix + "weight"
        return weight_key in state_dict and isinstance(state_dict[weight_key], GGUFParameter)

    has_children = list(model.children())
    if not has_children:
        return

    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        _replace_with_gguf_linear(module, compute_dtype, state_dict, module_prefix, modules_to_not_convert)

        if (
            isinstance(module, nn.Linear)
            and _should_convert_to_gguf(state_dict, module_prefix)
            and name not in modules_to_not_convert
        ):
            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                model._modules[name] = GGUFLinear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                )
            model._modules[name].source_cls = type(module)
            # Force requires_grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

    return model


def _dequantize_gguf_and_restore_linear(model, modules_to_not_convert=[]):
    for name, module in model.named_children():
        if isinstance(module, GGUFLinear) and name not in modules_to_not_convert:
            device = module.weight.device
            bias = getattr(module, "bias", None)

            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    device=device,
                )
            new_module.weight = nn.Parameter(dequantize_gguf_tensor(module.weight))
            if bias is not None:
                new_module.bias = bias

            # Create a new hook and attach it in case we use accelerate
            if hasattr(module, "_hf_hook"):
                old_hook = module._hf_hook
                new_hook = _create_accelerate_new_hook(old_hook)

                remove_hook_from_module(module)
                add_hook_to_module(new_module, new_hook)

            new_module.to(device)
            model._modules[name] = new_module

        has_children = list(module.children())
        if has_children:
            _dequantize_gguf_and_restore_linear(module, modules_to_not_convert)

    return model


# dequantize operations based on torch ports of GGUF dequantize_functions
# from City96
# more info: https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py


QK_K = 256
K_SCALE_SIZE = 12


def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = ql | (qh << 4)
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    (
        ql,
        qh,
        scales,
        d,
    ) = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(0, 8, device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 8, 1)
    )
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2, 1)
    )
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 2, 1)
    )
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.arange(0, 8, device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 8, 1)
    )
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# this part from calcuis (gguf.org)
# more info: https://github.com/calcuis/gguf-connector/blob/main/src/gguf_connector/quant2c.py


def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32,
        device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs


def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32,
        device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8
    ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)) & 0x03
    scales = (scales_l | (scales_h << 4)) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2], 16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)


GGML_QUANT_SIZES = gguf.GGML_QUANT_SIZES
dequantize_functions = {
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}
SUPPORTED_GGUF_QUANT_TYPES = list(dequantize_functions.keys())


def _quant_shape_from_byte_shape(shape, type_size, block_size):
    return (*shape[:-1], shape[-1] // type_size * block_size)


def dequantize_gguf_tensor(tensor):
    if not hasattr(tensor, "quant_type"):
        return tensor

    quant_type = tensor.quant_type
    dequant_fn = dequantize_functions[quant_type]

    block_size, type_size = GGML_QUANT_SIZES[quant_type]

    tensor = tensor.view(torch.uint8)
    shape = _quant_shape_from_byte_shape(tensor.shape, type_size, block_size)

    n_blocks = tensor.numel() // type_size
    blocks = tensor.reshape((n_blocks, type_size))

    dequant = dequant_fn(blocks, block_size, type_size)
    dequant = dequant.reshape(shape)

    return dequant.as_tensor()


class GGUFParameter(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=False, quant_type=None):
        data = data if data is not None else torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_type = quant_type
        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        self.quant_shape = _quant_shape_from_byte_shape(self.shape, type_size, block_size)

        return self

    def as_tensor(self):
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)

    @staticmethod
    def _extract_quant_type(args):
        # When converting from original format checkpoints we often use splits, cats etc on tensors
        # this method ensures that the returned tensor type from those operations remains GGUFParameter
        # so that we preserve quant_type information
        for arg in args:
            if isinstance(arg, list) and isinstance(arg[0], GGUFParameter):
                return arg[0].quant_type
            if isinstance(arg, GGUFParameter):
                return arg.quant_type
        return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, torch.Tensor):
            quant_type = cls._extract_quant_type(args)
            return cls(result, quant_type=quant_type)
        # Handle tuples and lists
        elif type(result) in (list, tuple):
            # Preserve the original type (tuple or list)
            quant_type = cls._extract_quant_type(args)
            wrapped = [cls(x, quant_type=quant_type) if isinstance(x, torch.Tensor) else x for x in result]
            return type(result)(wrapped)
        else:
            return result


class GGUFLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.device = device

    def forward(self, inputs: torch.Tensor):
        if ops is not None and self.weight.is_cuda and inputs.is_cuda:
            return self.forward_cuda(inputs)
        return self.forward_native(inputs)

    def forward_native(self, inputs: torch.Tensor):
        weight = dequantize_gguf_tensor(self.weight)
        weight = weight.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None

        output = torch.nn.functional.linear(inputs, weight, bias)
        return output

    def forward_cuda(self, inputs: torch.Tensor):
        quant_type = self.weight.quant_type
        output = _fused_mul_mat_gguf(inputs.to(self.compute_dtype), self.weight, quant_type)
        if self.bias is not None:
            output += self.bias.to(self.compute_dtype)
        return output
