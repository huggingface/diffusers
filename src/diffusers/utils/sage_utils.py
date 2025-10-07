"""
Copyright (c) 2024 by SageAttention, The HuggingFace team.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

"""
Modified from
https://github.com/thu-ml/SageAttention/blob/68de3797d163b89d28f9a38026c3b7313f6940d2/sageattention/core.py
"""


import torch  # noqa


SAGE_ATTENTION_DISPATCH = {
    "sm80": {
        "func": "sageattn_qk_int8_pv_fp16_cuda",
        "kwargs": {
            "tensor_layout": "NHD",
            "is_causal": False,
            "sm_scale": None,
            "return_lse": False,
            "pv_accum_dtype": "fp32",
        },
    },
    "sm89": {
        "func": "sageattn_qk_int8_pv_fp8_cuda",
        "kwargs": {
            "tensor_layout": "NHD",
            "is_causal": False,
            "sm_scale": None,
            "return_lse": False,
            "pv_accum_dtype": "fp32+fp16",
        },
    },
    "sm90": {
        "func": "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "kwargs": {
            "tensor_layout": "NHD",
            "is_causal": False,
            "sm_scale": None,
            "return_lse": False,
            "pv_accum_dtype": "fp32+fp32",
        },
    },
    "sm120": {
        "func": "sageattn_qk_int8_pv_fp8_cuda",
        "kwargs": {
            "tensor_layout": "NHD",
            "is_causal": False,
            "qk_quant_gran": "per_warp",
            "sm_scale": None,
            "return_lse": False,
            "pv_accum_dtype": "fp32+fp16",
        },
    },
}


def get_cuda_version():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major, minor
    else:
        raise EnvironmentError("CUDA not found.")


def get_cuda_arch_versions():
    if not torch.cuda.is_available():
        EnvironmentError("CUDA not found.")
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


# Unlike the actual implementation, we just maintain function names rather than actual
# implementations.
def _get_sage_attn_fn_for_device():
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute
    capability.

    Parameters ---------- q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD". Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len. Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns ------- torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor). Shape:
        ``[batch_size, num_qo_heads, qo_len]``. Only returned if `return_lse` is True.

    Note ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """
    device_index = torch.cuda.current_device()
    arch = get_cuda_arch_versions()[device_index]
    return SAGE_ATTENTION_DISPATCH[arch]
