import gc
import unittest

import numpy as np
import torch
import torch.nn as nn

from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.utils.testing_utils import (
    is_gguf_available,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_big_gpu_with_torch_cuda,
    require_gguf_version_greater_or_equal,
    torch_device,
)


if is_gguf_available():
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter


@nightly
@require_big_gpu_with_torch_cuda
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class GGUFSingleFileTesterMixin:
    ckpt_path = None
    model_cls = None
    torch_dtype = torch.bfloat16
    expected_memory_use_in_gb = 5

    def test_gguf_parameters(self):
        quant_storage_type = torch.uint8
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for param_name, param in model.named_parameters():
            if isinstance(param, GGUFParameter):
                assert hasattr(param, "quant_type")
                assert param.dtype == quant_storage_type

    def test_gguf_linear_layers(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module.weight, "quant_type"):
                assert module.weight.dtype == torch.uint8
                if module.bias is not None:
                    assert module.bias.dtype == torch.float32

    def test_gguf_memory_usage(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)

        model = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        model.to("cuda")
        assert (model.get_memory_footprint() / 1024**3) < self.expected_memory_use_in_gb
        inputs = self.get_dummy_inputs()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model(**inputs)
        max_memory = torch.cuda.max_memory_allocated()
        assert (max_memory / 1024**3) < self.expected_memory_use_in_gb

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
        _keep_in_fp32_modules = self.model_cls._keep_in_fp32_modules
        self.model_cls._keep_in_fp32_modules = ["proj_out"]

        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    assert module.weight.dtype == torch.float32
        self.model_cls._keep_in_fp32_modules = _keep_in_fp32_modules

    def test_dtype_assignment(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)

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

    def test_dequantize_model(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        model = self.model_cls.from_single_file(self.ckpt_path, quantization_config=quantization_config)
        model.dequantize()

        def _check_for_gguf_linear(model):
            has_children = list(model.children())
            if not has_children:
                return

            for name, module in model.named_children():
                if isinstance(module, nn.Linear):
                    assert not isinstance(module, GGUFLinear), f"{name} is still GGUFLinear"
                    assert not isinstance(module.weight, GGUFParameter), f"{name} weight is still GGUFParameter"

        for name, module in model.named_children():
            _check_for_gguf_linear(module)


class FluxGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
    torch_dtype = torch.bfloat16
    model_cls = FluxTransformer2DModel
    expected_memory_use_in_gb = 5

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

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a cat holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.47265625,
                0.43359375,
                0.359375,
                0.47070312,
                0.421875,
                0.34375,
                0.46875,
                0.421875,
                0.34765625,
                0.46484375,
                0.421875,
                0.34179688,
                0.47070312,
                0.42578125,
                0.34570312,
                0.46875,
                0.42578125,
                0.3515625,
                0.45507812,
                0.4140625,
                0.33984375,
                0.4609375,
                0.41796875,
                0.34375,
                0.45898438,
                0.41796875,
                0.34375,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


class SD35LargeGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/blob/main/sd3.5_large-Q4_0.gguf"
    torch_dtype = torch.bfloat16
    model_cls = SD3Transformer2DModel
    expected_memory_use_in_gb = 5

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a cat holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.17578125,
                0.27539062,
                0.27734375,
                0.11914062,
                0.26953125,
                0.25390625,
                0.109375,
                0.25390625,
                0.25,
                0.15039062,
                0.26171875,
                0.28515625,
                0.13671875,
                0.27734375,
                0.28515625,
                0.12109375,
                0.26757812,
                0.265625,
                0.16210938,
                0.29882812,
                0.28515625,
                0.15625,
                0.30664062,
                0.27734375,
                0.14648438,
                0.29296875,
                0.26953125,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


class SD35MediumGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/stable-diffusion-3.5-medium-gguf/blob/main/sd3.5_medium-Q3_K_M.gguf"
    torch_dtype = torch.bfloat16
    model_cls = SD3Transformer2DModel
    expected_memory_use_in_gb = 2

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 16, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "pooled_projections": torch.randn(
                (1, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a cat holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.625,
                0.6171875,
                0.609375,
                0.65625,
                0.65234375,
                0.640625,
                0.6484375,
                0.640625,
                0.625,
                0.6484375,
                0.63671875,
                0.6484375,
                0.66796875,
                0.65625,
                0.65234375,
                0.6640625,
                0.6484375,
                0.6328125,
                0.6640625,
                0.6484375,
                0.640625,
                0.67578125,
                0.66015625,
                0.62109375,
                0.671875,
                0.65625,
                0.62109375,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


class AuraFlowGGUFSingleFileTests(GGUFSingleFileTesterMixin, unittest.TestCase):
    ckpt_path = "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf"
    torch_dtype = torch.bfloat16
    model_cls = AuraFlowTransformer2DModel
    expected_memory_use_in_gb = 4

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.torch_dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 2048),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.torch_dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.torch_dtype),
        }

    def test_pipeline_inference(self):
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch_dtype)
        transformer = self.model_cls.from_single_file(
            self.ckpt_path, quantization_config=quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow-v0.3", transformer=transformer, torch_dtype=self.torch_dtype
        )
        pipe.enable_model_cpu_offload()

        prompt = "a pony holding a sign that says hello"
        output = pipe(
            prompt=prompt, num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0), output_type="np"
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slice = np.array(
            [
                0.46484375,
                0.546875,
                0.64453125,
                0.48242188,
                0.53515625,
                0.59765625,
                0.47070312,
                0.5078125,
                0.5703125,
                0.42773438,
                0.50390625,
                0.5703125,
                0.47070312,
                0.515625,
                0.57421875,
                0.45898438,
                0.48632812,
                0.53515625,
                0.4453125,
                0.5078125,
                0.56640625,
                0.47851562,
                0.5234375,
                0.57421875,
                0.48632812,
                0.5234375,
                0.56640625,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4
