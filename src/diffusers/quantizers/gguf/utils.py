import torch
import torch.nn as nn
import gguf


QK_K_BLOCKSIZE = 256
K_SCALE_SIZE = 12


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K_BLOCKSIZE // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K_BLOCKSIZE // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K_BLOCKSIZE // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K_BLOCKSIZE // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


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
        self._dequant_fn = gguf.quants.dequantize
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type

    def forward(self, inputs):
        weight = self._dequant_fn(self.weight, self.quant_type).to(self.compute_dtype)
        bias = self._dequant_fn(self.bias, self.quant_type).to(self.compute_dtype)

        return torch.nn.functional.linear(inputs, weight, bias)
