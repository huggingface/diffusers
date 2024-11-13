import torch
from torch._prims_common import is_low_precision_dtype
import torch.nn as nn
import gguf

_GGUF_FILE_TYPE_MAPPING = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    4: "MOSTLY_Q4_1_SOME_F16",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
}

QK_K_BLOCKSIZE = 256
K_SCALE_SIZE = 12


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_Q2_K(blocks, dtype=None):
    n_blocks = blocks.shape[0]

    scales, quantized_values, delta, delta_min = split_block_dims(blocks, QK_K_BLOCKSIZE // 16, QK_K_BLOCKSIZE // 4, 2)
    delta = delta.view(torch.float16).to(dtype)
    delta_min = delta_min.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (delta * (scales & 0xF)).reshape((n_blocks, QK_K_BLOCKSIZE // 16, 1))
    ml = (delta_min * (scales >> 4)).reshape((n_blocks, QK_K_BLOCKSIZE // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=delta.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (quantized_values.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K_BLOCKSIZE // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


dequantize_fns = {
    "MOSTLY_Q2_K": dequantize_Q2_K,
}


def _replace_with_gguf_linear(model, compute_dtype, quant_type, qtypes=None):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            model._modules[name] = GGUFLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
            )
            model._modules[name].source_cls = type(module)
            # Force requires grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

        has_children = list(module.children())
        if has_children:
            _replace_with_gguf_linear(module, compute_dtype, quant_type)

    return model


class GGUFParameter(torch.nn.Parameter):
    def __new__(cls, data, requires_grad=False, tensor_type=None):
        data = data if data is not None else torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.tensor_type = tensor_type

        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)

        tensor_type = None
        for arg in args:
            if isinstance(arg, GGUFParameter):
                tensor_type = arg.tensor_type
                break
        if isinstance(result, torch.Tensor):
            return cls(result, tensor_type=tensor_type)
        # Handle tuples and lists
        elif isinstance(result, (tuple, list)):
            # Preserve the original type (tuple or list)
            wrapped = [cls(x, tensor_type=tensor_type) if isinstance(x, torch.Tensor) else x for x in result]
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
        quant_type=None,
        device=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self._dequant_fn = dequantize_fns[self.quant_type]

    def forward(self, inputs):
        is_gguf_quant = hasattr(self.weight, "tensor_type")
        if is_gguf_quant:
            weight = self._dequant_fn(self.weight, torch.uint8).to(self.compute_dtype)
        else:
            weight = self.weight
        __import__("ipdb").set_trace()
        return torch.nn.functional.linear(inputs, weight, self.bias)
