import gc
import tempfile
import unittest

from diffusers import FluxPipeline, FluxTransformer2DModel, QuantoConfig
from diffusers.models.attention_processor import Attention
from diffusers.utils import is_optimum_quanto_available, is_torch_available

from ...testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_accelerator,
    require_torch_cuda_compatibility,
    torch_device,
)


if is_optimum_quanto_available():
    from optimum.quanto import QLinear

if is_torch_available():
    import torch

    from ..utils import LoRALayer, get_memory_consumption_stat

enable_full_determinism()


@nightly
@require_accelerator
@require_accelerate
class QuantoBaseTesterMixin:
    model_id = None
    pipeline_model_id = None
    model_cls = None
    torch_dtype = torch.bfloat16
    # the expected reduction in peak memory used compared to an unquantized model expressed as a percentage
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
        return {"weights_dtype": "float8"}

    def get_dummy_model_init_kwargs(self):
        return {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self.torch_dtype,
            "quantization_config": QuantoConfig(**self.get_dummy_init_kwargs()),
        }

    def test_quanto_layers(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert isinstance(module, QLinear)

    def test_quanto_memory_usage(self):
        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=torch.bfloat16) for k, v in inputs.items() if not isinstance(v, bool)
        }

        unquantized_model = self.model_cls.from_pretrained(self.model_id, torch_dtype=self.torch_dtype)
        unquantized_model.to(torch_device)
        unquantized_model_memory = get_memory_consumption_stat(unquantized_model, inputs)

        quantized_model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())
        quantized_model.to(torch_device)
        quantized_model_memory = get_memory_consumption_stat(quantized_model, inputs)

        assert unquantized_model_memory / quantized_model_memory >= self.expected_memory_reduction

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
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
        quantization_config = QuantoConfig(**quantization_config_kwargs)

        init_kwargs.update({"quantization_config": quantization_config})

        model = self.model_cls.from_pretrained(**init_kwargs)
        model.to(torch_device)

        for name, module in model.named_modules():
            if name in self.modules_to_not_convert:
                assert not isinstance(module, QLinear)

    def test_dtype_assignment(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())

        with self.assertRaises(ValueError):
            # Tries with a `dtype`
            model.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device` and `dtype`
            device_0 = f"{torch_device}:0"
            model.to(device=device_0, dtype=torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.float()

        with self.assertRaises(ValueError):
            # Tries with a cast
            model.half()

        # This should work
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
                **self.get_dummy_model_init_kwargs(), device_map={0: "8GB", "cpu": "16GB"}
            )


class FluxTransformerQuantoMixin(QuantoBaseTesterMixin):
    model_id = "hf-internal-testing/tiny-flux-transformer"
    model_cls = FluxTransformer2DModel
    pipeline_cls = FluxPipeline
    torch_dtype = torch.bfloat16
    keep_in_fp32_module = "proj_out"
    modules_to_not_convert = ["proj_out"]
    _test_torch_compile = False

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

    def get_dummy_training_inputs(self, device=None, seed: int = 0):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

        torch.manual_seed(seed)
        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(device, dtype=torch.bfloat16)

        torch.manual_seed(seed)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(
            device, dtype=torch.bfloat16
        )

        torch.manual_seed(seed)
        pooled_prompt_embeds = torch.randn((batch_size, embedding_dim)).to(device, dtype=torch.bfloat16)

        torch.manual_seed(seed)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(device, dtype=torch.bfloat16)

        torch.manual_seed(seed)
        image_ids = torch.randn((height * width, num_image_channels)).to(device, dtype=torch.bfloat16)

        timestep = torch.tensor([1.0]).to(device, dtype=torch.bfloat16).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": image_ids,
            "timestep": timestep,
        }

    def test_model_cpu_offload(self):
        init_kwargs = self.get_dummy_init_kwargs()
        transformer = self.model_cls.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            quantization_config=QuantoConfig(**init_kwargs),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = self.pipeline_cls.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe", transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)

    def test_training(self):
        quantization_config = QuantoConfig(**self.get_dummy_init_kwargs())
        quantized_model = self.model_cls.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)

        for param in quantized_model.parameters():
            # freeze the model as only adapter layers will be trained
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        for _, module in quantized_model.named_modules():
            if isinstance(module, Attention):
                module.to_q = LoRALayer(module.to_q, rank=4)
                module.to_k = LoRALayer(module.to_k, rank=4)
                module.to_v = LoRALayer(module.to_v, rank=4)

        with torch.amp.autocast(str(torch_device), dtype=torch.bfloat16):
            inputs = self.get_dummy_training_inputs(torch_device)
            output = quantized_model(**inputs)[0]
            output.norm().backward()

        for module in quantized_model.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)


class FluxTransformerFloat8WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_reduction = 0.6

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "float8"}


class FluxTransformerInt8WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_reduction = 0.6
    _test_torch_compile = True

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "int8"}


@require_torch_cuda_compatibility(8.0)
class FluxTransformerInt4WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "int4"}


@require_torch_cuda_compatibility(8.0)
class FluxTransformerInt2WeightsTest(FluxTransformerQuantoMixin, unittest.TestCase):
    expected_memory_reduction = 0.65

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "int2"}
