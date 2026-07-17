import gc

import pytest

from diffusers import FluxPipeline, FluxTransformer2DModel, QuantoConfig
from diffusers.utils import is_torch_available

from ...testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    require_accelerate,
    require_accelerator,
    require_torch_cuda_compatibility,
    torch_device,
)


if is_torch_available():
    import torch

    from ..utils import get_memory_consumption_stat

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

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()
        yield
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

    def test_dtype_assignment(self):
        model = self.model_cls.from_pretrained(**self.get_dummy_model_init_kwargs())

        with pytest.raises(ValueError):
            # Tries with a `dtype`
            model.to(torch.float16)

        with pytest.raises(ValueError):
            # Tries with a `device` and `dtype`
            device_0 = f"{torch_device}:0"
            model.to(device=device_0, dtype=torch.float16)

        with pytest.raises(ValueError):
            # Tries with a cast
            model.float()

        with pytest.raises(ValueError):
            # Tries with a cast
            model.half()

        # This should work
        model.to(torch_device)

    def test_device_map_error(self):
        with pytest.raises(ValueError):
            _ = self.model_cls.from_pretrained(
                **self.get_dummy_model_init_kwargs(), device_map={0: "8GB", "cpu": "16GB"}
            )


class FluxTransformerQuantoMixin(QuantoBaseTesterMixin):
    model_id = "hf-internal-testing/tiny-flux-transformer"
    model_cls = FluxTransformer2DModel
    pipeline_cls = FluxPipeline
    torch_dtype = torch.bfloat16

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


class TestFluxTransformerFloat8Weights(FluxTransformerQuantoMixin):
    expected_memory_reduction = 0.6

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "float8"}


class TestFluxTransformerInt8Weights(FluxTransformerQuantoMixin):
    expected_memory_reduction = 0.6

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "int8"}


@require_torch_cuda_compatibility(8.0)
class TestFluxTransformerInt4Weights(FluxTransformerQuantoMixin):
    expected_memory_reduction = 0.55

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "int4"}


@require_torch_cuda_compatibility(8.0)
class TestFluxTransformerInt2Weights(FluxTransformerQuantoMixin):
    expected_memory_reduction = 0.65

    def get_dummy_init_kwargs(self):
        return {"weights_dtype": "int2"}
