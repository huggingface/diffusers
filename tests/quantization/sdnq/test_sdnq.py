import gc
import tempfile

import pytest

from diffusers import FluxPipeline, FluxTransformer2DModel, SDNQConfig
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import is_torch_available

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    require_accelerate,
    require_accelerator,
    require_sdnq,
    torch_device,
)


if is_torch_available():
    import torch

enable_full_determinism()


@nightly
@require_sdnq
@require_accelerator
@require_accelerate
class TestSDNQ:
    model_id = "hf-internal-testing/tiny-flux-transformer"
    pipeline_model_id = "hf-internal-testing/tiny-flux-pipe"
    model_cls = FluxTransformer2DModel
    dtype = torch.bfloat16

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        backend_empty_cache(torch_device)
        gc.collect()
        yield
        backend_empty_cache(torch_device)
        gc.collect()

    def get_dummy_inputs(self):
        return {
            "hidden_states": torch.randn((1, 4096, 64), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.dtype
            ),
            "encoder_hidden_states": torch.randn(
                (1, 512, 4096),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.dtype),
            "pooled_projections": torch.randn(
                (1, 768),
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device, self.dtype),
            "timestep": torch.tensor([1]).to(torch_device, self.dtype),
            "img_ids": torch.randn((4096, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.dtype
            ),
            "txt_ids": torch.randn((512, 3), generator=torch.Generator("cpu").manual_seed(0)).to(
                torch_device, self.dtype
            ),
            "guidance": torch.tensor([3.5]).to(torch_device, self.dtype),
        }

    def get_quantized_model(self, **quant_kwargs):
        quant_kwargs = {"weights_dtype": "int8", **quant_kwargs}
        return self.model_cls.from_pretrained(
            self.model_id,
            quantization_config=SDNQConfig(**quant_kwargs),
            dtype=self.dtype,
        )

    def _num_sdnq_layers(self, model):
        return sum(1 for module in model.modules() if hasattr(module, "sdnq_dequantizer"))

    def test_quantize_on_load(self):
        model = self.get_quantized_model()
        assert self._num_sdnq_layers(model) > 0
        assert model.quantization_config.quant_method == "sdnq"

    def test_forward(self):
        model = self.get_quantized_model().to(torch_device)
        inputs = self.get_dummy_inputs()
        with torch.no_grad():
            output = model(**inputs)[0]
        assert output is not None and not output.isnan().any()

    def test_serialization_round_trip(self):
        model = self.get_quantized_model().to(torch_device)
        inputs = self.get_dummy_inputs()
        with torch.no_grad():
            output = model(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            del model
            gc.collect()
            # No quantization_config passed — it must be picked up from the saved config.json.
            loaded_model = self.model_cls.from_pretrained(tmp_dir, dtype=self.dtype).to(torch_device)

        assert self._num_sdnq_layers(loaded_model) > 0
        with torch.no_grad():
            loaded_output = loaded_model(**inputs)[0]
        # bf16 quantized outputs differ in the last bits between the fresh and reloaded model
        assert torch.allclose(output, loaded_output, atol=1e-2, rtol=1e-2)

    def test_pipeline_quant_config(self):
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_mapping={"transformer": SDNQConfig(weights_dtype="int8")}
        )
        pipe = FluxPipeline.from_pretrained(
            self.pipeline_model_id,
            quantization_config=pipeline_quant_config,
            dtype=self.dtype,
        ).to(torch_device)
        assert self._num_sdnq_layers(pipe.transformer) > 0
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)

    def test_uint4_svd(self):
        model = self.get_quantized_model(weights_dtype="uint4", use_svd=True)
        assert self._num_sdnq_layers(model) > 0

    def test_prequantized_pipeline_round_trip(self):
        # sdnq also registers itself with transformers, so text encoders can be quantized too.
        from sdnq import SDNQConfig as SDNQLibConfig
        from transformers import CLIPTextModel

        transformer = self.model_cls.from_pretrained(
            self.pipeline_model_id,
            subfolder="transformer",
            quantization_config=SDNQConfig(weights_dtype="int8"),
            dtype=self.dtype,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.pipeline_model_id,
            subfolder="text_encoder",
            quantization_config=SDNQLibConfig(weights_dtype="int8"),
            dtype=self.dtype,
        )
        pipe = FluxPipeline.from_pretrained(
            self.pipeline_model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            dtype=self.dtype,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe.save_pretrained(tmp_dir)
            del pipe, transformer, text_encoder
            gc.collect()
            # No quantization_config passed — both components must reload quantized from their config.json.
            loaded_pipe = FluxPipeline.from_pretrained(tmp_dir, dtype=self.dtype).to(torch_device)

        assert self._num_sdnq_layers(loaded_pipe.transformer) > 0
        assert self._num_sdnq_layers(loaded_pipe.text_encoder) > 0
        _ = loaded_pipe("a cat holding a sign that says hello", num_inference_steps=2)
