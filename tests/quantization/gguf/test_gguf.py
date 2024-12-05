import gc
import unittest

import torch

from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers.utils.testing_utils import (
    is_gguf_available,
    nightly,
    require_accelerate,
    require_big_gpu_with_torch_cuda,
    require_gguf_version_greater_or_equal,
    torch_device,
)


if is_gguf_available():
    from diffusers.quantizers.gguf.utils import GGUFParameter


@nightly
@require_big_gpu_with_torch_cuda
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class GGUFSingleFileTests(unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
    torch_dtype = torch.bfloat16

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4096, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 768),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
            "img_ids": torch.randn((4096, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "txt_ids": torch.randn((512, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "guidance": torch.tensor([3.5]).to(torch_device, self.torch_dtype),
        }

    def test_gguf_parameters(self):
        quant_storage_type = torch.uint8
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = FluxTransformer2DModel.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for param_name, param in model.named_parameters():
            if isinstance(param, GGUFParameter):
                assert hasattr(param, "quant_type")
                assert param.dtype == quant_storage_type

    def test_gguf_linear_layers(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = FluxTransformer2DModel.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module.weight, "quant_type"):
                assert module.weight.dtype == torch.uint8

    def test_gguf_memory(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)

        model = FluxTransformer2DModel.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        model.to("cuda")
        assert (model.get_memory_footprint() / 1024**3) < 5
        inputs = self.get_dummy_inputs()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model(**inputs)
        max_memory = torch.cuda.max_memory_allocated()
        assert (max_memory / 1024**3) < 5

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
        FluxTransformer2DModel._keep_in_fp32_modules = ["proj_out"]

        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = FluxTransformer2DModel.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    assert module.weight.dtype == torch.float32

    def test_dtype_assignment(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = FluxTransformer2DModel.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        with self.assertRaises(ValueError):
            # Tries with a `dtype`
            model.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device` and `dtype`
            model.to(device="cuda:0", dtype=torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.float()

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.half()

        # This should work
        model.to("cuda")
