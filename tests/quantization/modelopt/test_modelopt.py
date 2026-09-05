import gc

import pytest

from diffusers import NVIDIAModelOptConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.utils import is_torch_available

from ...testing_utils import (
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    nightly,
    require_accelerate,
    require_big_accelerator,
    require_modelopt_version_greater_or_equal,
    require_torch_cuda_compatibility,
    torch_device,
)


if is_torch_available():
    import torch

enable_full_determinism()


# Model-level ModelOpt tests live in `tests/models/testing_utils/quantization.py`
# (`ModelOptTesterMixin` / `ModelOptCompileTesterMixin`), wired into model test files via concrete
# classes (e.g. `TestSD3TransformerModelOpt`). Only pipeline-level coverage remains here.
@nightly
@require_big_accelerator
@require_accelerate
@require_modelopt_version_greater_or_equal("0.33.1")
class ModelOptBaseTesterMixin:
    model_id = "hf-internal-testing/tiny-sd3-pipe"
    model_cls = SD3Transformer2DModel
    pipeline_cls = StableDiffusion3Pipeline
    torch_dtype = torch.bfloat16

    @pytest.fixture(autouse=True)
    def _setup(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()
        yield
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def get_dummy_init_kwargs(self):
        return {"quant_type": "FP8"}

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


class TestSanaTransformerFP8Weights(ModelOptBaseTesterMixin):
    def get_dummy_init_kwargs(self):
        return {"quant_type": "FP8"}


class TestSanaTransformerINT8Weights(ModelOptBaseTesterMixin):
    def get_dummy_init_kwargs(self):
        return {"quant_type": "INT8"}


@require_torch_cuda_compatibility(8.0)
class TestSanaTransformerINT4Weights(ModelOptBaseTesterMixin):
    def get_dummy_init_kwargs(self):
        return {
            "quant_type": "INT4",
            "block_quantize": 128,
            "channel_quantize": -1,
            "disable_conv_quantization": True,
        }


@require_torch_cuda_compatibility(8.0)
class TestSanaTransformerNF4Weights(ModelOptBaseTesterMixin):
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
class TestSanaTransformerNVFP4Weights(ModelOptBaseTesterMixin):
    def get_dummy_init_kwargs(self):
        return {
            "quant_type": "NVFP4",
            "block_quantize": 128,
            "channel_quantize": -1,
            "scale_block_quantize": 8,
            "scale_channel_quantize": -1,
            "modules_to_not_convert": ["conv"],
        }
