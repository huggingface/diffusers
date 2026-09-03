import gc

import pytest
import torch
import torch.nn as nn

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    is_gguf,
    is_gguf_available,
    is_quantization,
    nightly,
    require_accelerate,
    require_accelerator,
    require_gguf_version_greater_or_equal,
    require_kernels_version_greater_or_equal,
    torch_device,
)


if is_gguf_available():
    import gguf

    from diffusers.quantizers.gguf.utils import GGUFParameter

enable_full_determinism()


# Model-level GGUF tests live in `tests/models/testing_utils/quantization.py` and pipeline-level
# ones in `tests/pipelines/testing_utils/quantization.py`. This module covers backend behavior that
# fits neither: CUDA kernel correctness.
@is_quantization
@is_gguf
@nightly
@require_accelerate
@require_accelerator
@require_gguf_version_greater_or_equal("0.10.0")
@require_kernels_version_greater_or_equal("0.9.0")
class TestGGUFCudaKernels:
    @pytest.fixture(autouse=True)
    def _setup_cuda_kernels(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_cuda_kernels_vs_native(self):
        if torch_device != "cuda":
            pytest.skip("CUDA kernels test requires CUDA device")

        from diffusers.quantizers.gguf.utils import GGUFLinear, can_use_cuda_kernels

        if not can_use_cuda_kernels:
            pytest.skip("CUDA kernels not available (compute capability < 7 or kernels not installed)")

        test_quant_types = ["Q4_0", "Q4_K"]
        test_shape = (1, 64, 512)  # batch, seq_len, hidden_dim
        compute_dtype = torch.bfloat16

        for quant_type in test_quant_types:
            qtype = getattr(gguf.GGMLQuantizationType, quant_type)
            in_features, out_features = 512, 512

            torch.manual_seed(42)
            float_weight = torch.randn(out_features, in_features, dtype=torch.float32)
            quantized_data = gguf.quants.quantize(float_weight.numpy(), qtype)
            weight_data = torch.from_numpy(quantized_data).to(device=torch_device)
            weight = GGUFParameter(weight_data, quant_type=qtype)

            x = torch.randn(test_shape, dtype=compute_dtype, device=torch_device)

            linear = GGUFLinear(in_features, out_features, bias=True, compute_dtype=compute_dtype)
            linear.weight = weight
            linear.bias = nn.Parameter(torch.randn(out_features, dtype=compute_dtype))
            linear = linear.to(torch_device)

            with torch.no_grad():
                output_native = linear.forward_native(x)
                output_cuda = linear.forward_cuda(x)

            assert torch.allclose(output_native, output_cuda, 1e-2), (
                f"GGUF CUDA Kernel Output is different from Native Output for {quant_type}"
            )


@is_quantization
@is_gguf
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class TestGGUFParameterRewrap:
    """Offloading rebuilds parameters without forwarding `quant_type`."""

    def _make_param(self):
        from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES

        quant_type = gguf.GGMLQuantizationType.Q8_0
        _, type_size = GGML_QUANT_SIZES[quant_type]
        data = torch.zeros((4, type_size), dtype=torch.uint8)

        return GGUFParameter(data, quant_type=quant_type), quant_type

    def test_rewrap_without_quant_type_inherits_it(self):
        param, quant_type = self._make_param()
        rewrapped = GGUFParameter(param, requires_grad=False)

        assert rewrapped.quant_type == quant_type
        assert rewrapped.quant_shape == param.quant_shape

    def test_set_module_tensor_to_device_preserves_quant_type(self):
        from accelerate.utils import set_module_tensor_to_device

        from diffusers.quantizers.gguf.utils import GGUFLinear

        param, quant_type = self._make_param()
        out_features, in_features = param.quant_shape
        module = GGUFLinear(in_features, out_features, bias=False, compute_dtype=torch.bfloat16)
        module.weight = param

        # Passing `value` (or moving across devices) is what makes `accelerate` rebuild the
        # parameter through `param_cls(new_value, requires_grad=...)`. A same-device move with
        # no value short-circuits before that branch and would not cover the regression.
        set_module_tensor_to_device(module, "weight", torch.device("cpu"), value=param)

        assert module.weight.quant_type == quant_type

    def test_untyped_construction_raises(self):
        with pytest.raises(ValueError):
            GGUFParameter(torch.zeros((4, 32), dtype=torch.uint8))
