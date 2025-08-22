import gc
import tempfile
import unittest

from diffusers import NVIDIAModelOptConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.utils import is_nvidia_modelopt_available, is_torch_available
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_big_accelerator,
    require_modelopt_version_greater_or_equal,
    require_torch_cuda_compatibility,
    torch_device,
)


if is_nvidia_modelopt_available():
    import modelopt.torch.quantization as mtq

if is_torch_available():
    import torch

    from ..utils import LoRALayer, get_memory_consumption_stat

enable_full_determinism()


@nightly
@require_big_accelerator
@require_accelerate
@require_modelopt_version_greater_or_equal("0.33.1")
class ModelOptBaseTesterMixin:
    model_id = "hf-internal-testing/tiny-sd3-pipe"
    model_cls = SD3Transformer2DModel
    pipeline_cls = StableDiffusion3Pipeline
    torch_dtype = torch.bfloat16
    expected_memory_reduction = 0.0
    keep_in_fp32_module = ""
    modules_to_not_convert = ""
    _test_torch_compile = False

    def setUp(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def tearDown(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def get_dummy_init_kwargs(self):
        return {"quant_type": "FP8"}

    def get_dummy_model_init_kwargs(self):
        return {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self.torch_dtype,
            "quantization_config": NVIDIAModelOptConfig(**self.get_dummy_init_kwargs()),
            "subfolder": "transformer",
        }

    def test_modelopt_layers(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert mtq.utils.is_quantized(module)

    def test_modelopt_memory_usage(self):
        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=torch.bfloat16) for k, v in inputs.items() if not isinstance(v, bool)
        }

        unquantized_model = self.model_cls.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, subfolder="transformer"
        )
        unquantized_model.to(torch_device)
        unquantized_model_memory = get_memory_consumption_stat(unquantized_model, inputs)

        quantized_model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        quantized_model.to(torch_device)
        quantized_model_memory = get_memory_consumption_stat(quantized_model, inputs)

        assert unquantized_model_memory / quantized_model_memory >= self.expected_memory_reduction

    def test_keep_modules_in_fp32(self):
        _keep_in_fp32_modules = self.model_cls._keep_in_fp32_modules
        self.model_cls._keep_in_fp32_modules = self.keep_in_fp32_module

        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        model.to(torch_device)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    assert module.weight.dtype == torch.float32
        self.model_cls._keep_in_fp32_modules = _keep_in_fp32_modules

    def test_modules_to_not_convert(self):
        init_kwargs = self.get_dummy_model_init_kwargs()
        quantization_config_kwargs = self.get_dummy_init_kwargs()
        quantization_config_kwargs.update({"modules_to_not_convert": self.modules_to_not_convert})
        quantization_config = NVIDIAModelOptConfig(**quantization_config_kwargs)
        init_kwargs.update({"quantization_config": quantization_config})

        model = self.model_cls.from_pretrained(**init_kwargs)
        model.to(torch_device)

        for name, module in model.named_modules():
            if name in self.modules_to_not_convert:
                assert not mtq.utils.is_quantized(module)

    def test_dtype_assignment(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())

        with self.assertRaises(ValueError):
            model.to(torch.float16)

        with self.assertRaises(ValueError):
            device_0 = f"{torch_device}:0"
            model.to(device=device_0, dtype=torch.float16)

        with self.assertRaises(ValueError):
            model.float()

        with self.assertRaises(ValueError):
            model.half()

        model.to(torch_device)

    def test_serialization(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        inputs = self.get_dummy_inputs()

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            saved_model = self.model_cls.from_pretrained(
                tmp_dir,
                torch_dtype=torch.bfloat16,
            )

        saved_model.to(torch_device)
        with torch.no_grad():
            saved_model_output = saved_model(**inputs)

        assert torch.allclose(model_output.sample, saved_model_output.sample, rtol=1e-5, atol=1e-5)

    def test_torch_compile(self):
        if not self._test_torch_compile:
            return

        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        compiled_model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)

        model.to(torch_device)
        with torch.no_grad():
            model_output = model(**self.get_dummy_inputs()).sample

        compiled_model.to(torch_device)
        with torch.no_grad():
            compiled_model_output = compiled_model(**self.get_dummy_inputs()).sample

        model_output = model_output.detach().float().cpu().numpy()
        compiled_model_output = compiled_model_output.detach().float().cpu().numpy()

        max_diff = numpy_cosine_similarity_distance(model_output.flatten(), compiled_model_output.flatten())
        assert max_diff < 1e-3

    def test_device_map_error(self):
        with self.assertRaises(ValueError):
            _ = self.model_cls.from_pretrained(
                **self.get_dummy_model_init_kwargs(),
                device_map={0: "8GB", "cpu": "16GB"},
            )

    def get_dummy_inputs(self):
        batch_size = 1
        seq_len = 16
        height = width = 32
        num_latent_channels = 4
        caption_channels = 8

        torch.manual_seed(0)
        hidden_states = torch.randn((batch_size, num_latent_channels, height, width)).to(
            torch_device, dtype=torch.bfloat16
        )
        encoder_hidden_states = torch.randn((batch_size, seq_len, caption_channels)).to(
            torch_device, dtype=torch.bfloat16
        )
        timestep = torch.tensor([1.0]).to(torch_device, dtype=torch.bfloat16).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }

    def test_model_cpu_offload(self):
        init_kwargs = self.get_dummy_init_kwargs()
        transformer = self.model_cls.from_pretrained(
            self.model_id,
            quantization_config=NVIDIAModelOptConfig(**init_kwargs),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = self.pipeline_cls.from_pretrained(self.model_id, transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=torch_device)
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)

    def test_training(self):
        quantization_config = NVIDIAModelOptConfig(**self.get_dummy_init_kwargs())
        quantized_model = self.model_cls.from_pretrained(
            self.model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)

        for param in quantized_model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        for _, module in quantized_model.named_modules():
            if hasattr(module, "to_q"):
                module.to_q = LoRALayer(module.to_q, rank=4)
            if hasattr(module, "to_k"):
                module.to_k = LoRALayer(module.to_k, rank=4)
            if hasattr(module, "to_v"):
                module.to_v = LoRALayer(module.to_v, rank=4)

        with torch.amp.autocast(str(torch_device), dtype=torch.bfloat16):
            inputs = self.get_dummy_inputs()
            output = quantized_model(**inputs)[0]
            output.norm().backward()

        for module in quantized_model.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)


class SanaTransformerFP8WeightsTest(ModelOptBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 0.6

    def get_dummy_init_kwargs(self):
        return {"quant_type": "FP8"}


class SanaTransformerINT8WeightsTest(ModelOptBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 0.6
    _test_torch_compile = True

    def get_dummy_init_kwargs(self):
        return {"quant_type": "INT8"}


@require_torch_cuda_compatibility(8.0)
class SanaTransformerINT4WeightsTest(ModelOptBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {
            "quant_type": "INT4",
            "block_quantize": 128,
            "channel_quantize": -1,
            "disable_conv_quantization": True,
        }


@require_torch_cuda_compatibility(8.0)
class SanaTransformerNF4WeightsTest(ModelOptBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 0.65

    def get_dummy_init_kwargs(self):
        return {
            "quant_type": "NF4",
            "block_quantize": 128,
            "channel_quantize": -1,
            "scale_block_quantize": 8,
            "scale_channel_quantize": -1,
            "modules_to_not_convert": ["conv"],
        }


@require_torch_cuda_compatibility(8.0)
class SanaTransformerNVFP4WeightsTest(ModelOptBaseTesterMixin, unittest.TestCase):
    expected_memory_reduction = 0.65

    def get_dummy_init_kwargs(self):
        return {
            "quant_type": "NVFP4",
            "block_quantize": 128,
            "channel_quantize": -1,
            "scale_block_quantize": 8,
            "scale_channel_quantize": -1,
            "modules_to_not_convert": ["conv"],
        }
