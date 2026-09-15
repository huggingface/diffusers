# coding=utf-8
# Copyright 2026 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline-level quantization tests.

Model-level quantization tests live in `tests/models/testing_utils/quantization.py` and are wired
into the individual model test files. Backend-level tests (config validation, loading error paths,
kernels) live in `tests/quantization/`. This module only covers behavior that needs a pipeline.

The module name intentionally does not match pytest's `test_*.py` discovery pattern: these tests
only run when this file is passed to pytest explicitly, as the nightly quantization CI jobs do.
Every class is marked with the `quantization` marker plus its backend marker (`bitsandbytes`,
`torchao`, `gguf`, `modelopt`) so CI can select per-backend subsets with `pytest -m`.
"""

import gc
import json
import tempfile

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from parameterized import parameterized
from PIL import Image

from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    AutoencoderKL,
    BitsAndBytesConfig,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    FluxControlPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    NVIDIAModelOptConfig,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    TorchAoConfig,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import is_accelerate_version, load_image, logging

from ...testing_utils import (
    CaptureLogger,
    Expectations,
    backend_empty_cache,
    backend_reset_peak_memory_stats,
    backend_synchronize,
    enable_full_determinism,
    is_bitsandbytes,
    is_bitsandbytes_available,
    is_gguf,
    is_gguf_available,
    is_modelopt,
    is_quantization,
    is_torchao,
    is_torchao_available,
    is_transformers_available,
    nightly,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_big_accelerator,
    require_bitsandbytes_version_greater,
    require_gguf_version_greater_or_equal,
    require_modelopt_version_greater_or_equal,
    require_peft_backend,
    require_peft_version_greater,
    require_torch,
    require_torch_accelerator,
    require_torch_version_greater,
    require_torch_version_greater_equal,
    require_torchao_version_greater_or_equal,
    require_transformers_version_greater,
    slow,
    torch_device,
)


if is_transformers_available():
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel
    from transformers import BitsAndBytesConfig as TranBitsAndBytesConfig
else:
    TranBitsAndBytesConfig = None

if is_bitsandbytes_available():
    pass

if is_torchao_available():
    from torchao.quantization import (
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8DynamicActivationIntxWeightConfig,
        Int8Tensor,
        Int8WeightOnlyConfig,
        IntxWeightOnlyConfig,
    )
    from torchao.utils import TorchAOBaseTensor

if is_gguf_available():
    pass


enable_full_determinism()


# ======================== Shared compile base ========================


@is_quantization
@require_torch_accelerator
@slow
class QuantCompileTests:
    @property
    def quantization_config(self):
        raise NotImplementedError(
            "This property should be implemented in the subclass to return the appropriate quantization config."
        )

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()
        yield
        gc.collect()
        backend_empty_cache(torch_device)
        torch.compiler.reset()

    def _init_pipeline(self, quantization_config, torch_dtype):
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        return pipe

    def _test_torch_compile_with_cpu_offload(self, torch_dtype=torch.bfloat16):
        pipe = self._init_pipeline(self.quantization_config, torch_dtype)
        pipe.enable_model_cpu_offload()
        # regional compilation is better for offloading.
        # see: https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
        if getattr(pipe.transformer, "_repeated_blocks"):
            pipe.transformer.compile_repeated_blocks(fullgraph=True)
        else:
            pipe.transformer.compile()

        # small resolutions to ensure speedy execution.
        pipe("a dog", num_inference_steps=2, max_sequence_length=16, height=256, width=256)

    def test_torch_compile_with_cpu_offload(self):
        self._test_torch_compile_with_cpu_offload()


# ======================== PipelineQuantizationConfig ========================


@is_quantization
@require_bitsandbytes_version_greater("0.43.2")
@require_torchao_version_greater_or_equal("0.16.0")
@require_accelerate
@require_torch
@require_torch_accelerator
@slow
class TestPipelineQuantization:
    model_name = "hf-internal-testing/tiny-flux-pipe"
    prompt = "a beautiful sunset amidst the mountains."
    num_inference_steps = 10
    seed = 0

    def test_quant_config_set_correctly_through_kwargs(self):
        components_to_quantize = ["transformer", "text_encoder_2"]
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=components_to_quantize,
        )
        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)
        for name, component in pipe.components.items():
            if name in components_to_quantize:
                assert getattr(component.config, "quantization_config", None) is not None
                quantization_config = component.config.quantization_config
                assert quantization_config.load_in_4bit
                assert quantization_config.quant_method == "bitsandbytes"

        _ = pipe(self.prompt, num_inference_steps=self.num_inference_steps)

    def test_quant_config_set_correctly_through_granular(self):
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig(Int8WeightOnlyConfig(version=2)),
                "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
            }
        )
        components_to_quantize = list(quant_config.quant_mapping.keys())
        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)
        for name, component in pipe.components.items():
            if name in components_to_quantize:
                assert getattr(component.config, "quantization_config", None) is not None
                quantization_config = component.config.quantization_config

                if name == "text_encoder_2":
                    assert quantization_config.load_in_4bit
                    assert quantization_config.quant_method == "bitsandbytes"
                else:
                    assert quantization_config.quant_method == "torchao"

        _ = pipe(self.prompt, num_inference_steps=self.num_inference_steps)

    def test_raises_error_for_invalid_config(self):
        with pytest.raises(ValueError) as err_context:
            _ = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": TorchAoConfig(Int8WeightOnlyConfig(version=2)),
                    "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
                },
                quant_backend="bitsandbytes_4bit",
            )

        assert (
            str(err_context.value) == "Both `quant_backend` and `quant_mapping` cannot be specified at the same time."
        )

    def test_validation_for_kwargs(self):
        components_to_quantize = ["transformer", "text_encoder_2"]
        with pytest.raises(ValueError) as err_context:
            _ = PipelineQuantizationConfig(
                quant_backend="torchao",
                quant_kwargs={"quant_type": Int8WeightOnlyConfig(version=2)},
                components_to_quantize=components_to_quantize,
            )

        assert "The signatures of the __init__ methods of the quantization config classes" in str(err_context.value)

    def test_raises_error_for_wrong_config_class(self):
        quant_config = {
            "transformer": TorchAoConfig(Int8WeightOnlyConfig(version=2)),
            "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
        }
        with pytest.raises(ValueError) as err_context:
            _ = DiffusionPipeline.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        assert str(err_context.value) == "`quantization_config` must be an instance of `PipelineQuantizationConfig`."

    def test_validation_for_mapping(self):
        with pytest.raises(ValueError) as err_context:
            _ = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusionPipeline(),
                    "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
                }
            )

        assert "Provided config for module_name=transformer could not be found" in str(err_context.value)

    def test_saving_loading(self):
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": TorchAoConfig(Int8WeightOnlyConfig(version=2)),
                "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
            }
        )
        components_to_quantize = list(quant_config.quant_mapping.keys())
        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)

        pipe_inputs = {"prompt": self.prompt, "num_inference_steps": self.num_inference_steps, "output_type": "latent"}
        output_1 = pipe(**pipe_inputs, generator=torch.manual_seed(self.seed)).images

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            loaded_pipe = DiffusionPipeline.from_pretrained(tmpdir, torch_dtype=torch.bfloat16).to(torch_device)
        for name, component in loaded_pipe.components.items():
            if name in components_to_quantize:
                assert getattr(component.config, "quantization_config", None) is not None
                quantization_config = component.config.quantization_config

                if name == "text_encoder_2":
                    assert quantization_config.load_in_4bit
                    assert quantization_config.quant_method == "bitsandbytes"
                else:
                    assert quantization_config.quant_method == "torchao"

        output_2 = loaded_pipe(**pipe_inputs, generator=torch.manual_seed(self.seed)).images

        assert torch.allclose(output_1, output_2)

    @parameterized.expand(["quant_kwargs", "quant_mapping"])
    def test_warn_invalid_component(self, method):
        invalid_component = "foo"
        if method == "quant_kwargs":
            components_to_quantize = ["transformer", invalid_component]
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=components_to_quantize,
            )
        else:
            quant_config = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": TorchAoConfig(Int8WeightOnlyConfig(version=2)),
                    invalid_component: TranBitsAndBytesConfig(load_in_8bit=True),
                }
            )

        logger = logging.get_logger("diffusers.pipelines.pipeline_loading_utils")
        logger.setLevel(logging.WARNING)
        with CaptureLogger(logger) as cap_logger:
            _ = DiffusionPipeline.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        assert invalid_component in cap_logger.out

    @parameterized.expand(["quant_kwargs", "quant_mapping"])
    def test_no_quantization_for_all_invalid_components(self, method):
        invalid_component = "foo"
        if method == "quant_kwargs":
            components_to_quantize = [invalid_component]
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=components_to_quantize,
            )
        else:
            quant_config = PipelineQuantizationConfig(
                quant_mapping={invalid_component: TranBitsAndBytesConfig(load_in_8bit=True)}
            )

        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        for name, component in pipe.components.items():
            if isinstance(component, torch.nn.Module):
                assert not hasattr(component.config, "quantization_config")

    @parameterized.expand(["quant_kwargs", "quant_mapping"])
    def test_quant_config_repr(self, method):
        component_name = "transformer"
        if method == "quant_kwargs":
            components_to_quantize = [component_name]
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=components_to_quantize,
            )
        else:
            quant_config = PipelineQuantizationConfig(
                quant_mapping={component_name: BitsAndBytesConfig(load_in_8bit=True)}
            )

        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        assert getattr(pipe, "quantization_config", None) is not None
        retrieved_config = pipe.quantization_config
        expected_config = """
transformer BitsAndBytesConfig {
  "_load_in_4bit": false,
  "_load_in_8bit": true,
  "bnb_4bit_compute_dtype": "float32",
  "bnb_4bit_quant_storage": "uint8",
  "bnb_4bit_quant_type": "fp4",
  "bnb_4bit_use_double_quant": false,
  "llm_int8_enable_fp32_cpu_offload": false,
  "llm_int8_has_fp16_weight": false,
  "llm_int8_skip_modules": null,
  "llm_int8_threshold": 6.0,
  "load_in_4bit": false,
  "load_in_8bit": true,
  "quant_method": "bitsandbytes"
}

"""
        expected_data = self._parse_config_string(expected_config)
        actual_data = self._parse_config_string(str(retrieved_config))
        assert actual_data == expected_data

    def _parse_config_string(self, config_string: str) -> tuple[str, dict]:
        first_brace = config_string.find("{")
        if first_brace == -1:
            raise ValueError("Could not find opening brace '{' in the string.")

        json_part = config_string[first_brace:]
        data = json.loads(json_part)

        return data

    def test_single_component_to_quantize(self):
        component_to_quantize = "transformer"
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=component_to_quantize,
        )
        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        for name, component in pipe.components.items():
            if name == component_to_quantize:
                assert hasattr(component.config, "quantization_config")


# ======================== BitsAndBytes ========================


# Model-level BitsAndBytes tests live in `tests/models/testing_utils/quantization.py`
# (`BitsAndBytesTesterMixin` / `BitsAndBytesCompileTesterMixin`), wired into model test files via
# concrete classes (e.g. `TestFluxTransformerBitsAndBytes`). Only pipeline-level coverage remains here.
@is_quantization
@is_bitsandbytes
@require_bitsandbytes_version_greater("0.43.2")
@require_accelerate
@require_torch
@require_torch_accelerator
@slow
class Base4bitTests:
    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only SD3 to test our module
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

    prompt = "a beautiful sunset amidst the mountains."
    num_inference_steps = 10
    seed = 0

    @pytest.fixture(autouse=True, scope="class")
    def _toggle_determinism(self):
        was_enabled = torch.are_deterministic_algorithms_enabled()
        if not was_enabled:
            torch.use_deterministic_algorithms(True)
        yield
        if not was_enabled:
            torch.use_deterministic_algorithms(False)


@require_transformers_version_greater("4.44.0")
class TestSlowBnb4Bit(Base4bitTests):
    @pytest.fixture(autouse=True)
    def _setup_slow(self):
        gc.collect()
        backend_empty_cache(torch_device)

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config, device_map=torch_device
        )
        self.pipeline_4bit = DiffusionPipeline.from_pretrained(
            self.model_name, transformer=model_4bit, torch_dtype=torch.float16
        )
        self.pipeline_4bit.enable_model_cpu_offload()
        yield
        del self.pipeline_4bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_quality(self):
        output = self.pipeline_4bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images

        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1123, 0.1296, 0.1609, 0.1042, 0.1230, 0.1274, 0.0928, 0.1165, 0.1216])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-2

    def test_generate_quality_dequantize(self):
        r"""
        Test that loading the model and unquantize it produce correct results.
        """
        self.pipeline_4bit.transformer.dequantize()
        output = self.pipeline_4bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images

        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1216, 0.1387, 0.1584, 0.1152, 0.1318, 0.1282, 0.1062, 0.1226, 0.1228])
        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3

        # Since we offloaded the `pipeline_4bit.transformer` to CPU (result of `enable_model_cpu_offload()), check
        # the following.
        assert self.pipeline_4bit.transformer.device.type == "cpu"
        # calling it again shouldn't be a problem
        _ = self.pipeline_4bit(
            prompt=self.prompt,
            num_inference_steps=2,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images

    def test_moving_to_cpu_throws_warning(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config, device_map=torch_device
        )

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            # Because `model.dtype` will return torch.float16 as SD3 transformer has
            # a conv layer as the first layer.
            _ = DiffusionPipeline.from_pretrained(
                self.model_name, transformer=model_4bit, torch_dtype=torch.float16
            ).to("cpu")

        assert "Pipelines loaded with `dtype=torch.float16`" in cap_logger.out

    @pytest.mark.xfail(
        condition=is_accelerate_version("<=", "1.1.1"),
        reason="Test will pass after https://github.com/huggingface/accelerate/pull/3223 is in a release.",
        strict=True,
    )
    def test_pipeline_cuda_placement_works_with_nf4(self):
        transformer_nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        transformer_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=transformer_nf4_config,
            torch_dtype=torch.float16,
            device_map=torch_device,
        )
        text_encoder_3_nf4_config = TranBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        text_encoder_3_4bit = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_3",
            quantization_config=text_encoder_3_nf4_config,
            torch_dtype=torch.float16,
            device_map=torch_device,
        )
        # CUDA device placement works.
        pipeline_4bit = DiffusionPipeline.from_pretrained(
            self.model_name,
            transformer=transformer_4bit,
            text_encoder_3=text_encoder_3_4bit,
            torch_dtype=torch.float16,
        ).to(torch_device)

        # Check if inference works.
        _ = pipeline_4bit(self.prompt, max_sequence_length=20, num_inference_steps=2)

        del pipeline_4bit


@require_transformers_version_greater("4.44.0")
class TestSlowBnb4BitFlux(Base4bitTests):
    @pytest.fixture(autouse=True)
    def _setup_flux(self):
        gc.collect()
        backend_empty_cache(torch_device)

        model_id = "hf-internal-testing/flux.1-dev-nf4-pkg"
        t5_4bit = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2")
        transformer_4bit = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer")
        self.pipeline_4bit = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder_2=t5_4bit,
            transformer=transformer_4bit,
            torch_dtype=torch.float16,
        )
        self.pipeline_4bit.enable_model_cpu_offload()
        yield
        del self.pipeline_4bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_quality(self):
        # keep the resolution and max tokens to a lower number for faster execution.
        output = self.pipeline_4bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            height=256,
            width=256,
            max_sequence_length=64,
            output_type="np",
        ).images

        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.0583, 0.0586, 0.0632, 0.0815, 0.0813, 0.0947, 0.1040, 0.1145, 0.1265])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3

    @require_peft_backend
    def test_lora_loading(self):
        self.pipeline_4bit.load_lora_weights(
            hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), adapter_name="hyper-sd"
        )
        self.pipeline_4bit.set_adapters("hyper-sd", adapter_weights=0.125)

        output = self.pipeline_4bit(
            prompt=self.prompt,
            height=256,
            width=256,
            max_sequence_length=64,
            output_type="np",
            num_inference_steps=8,
            generator=torch.Generator().manual_seed(42),
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.5347, 0.5342, 0.5283, 0.5093, 0.4988, 0.5093, 0.5044, 0.5015, 0.4946])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3


@require_transformers_version_greater("4.44.0")
@require_peft_backend
class TestSlowBnb4BitFluxControlWithLora(Base4bitTests):
    @pytest.fixture(autouse=True)
    def _setup_flux_control(self):
        gc.collect()
        backend_empty_cache(torch_device)

        self.pipeline_4bit = FluxControlPipeline.from_pretrained("eramth/flux-4bit", torch_dtype=torch.float16)
        self.pipeline_4bit.enable_model_cpu_offload()
        yield
        del self.pipeline_4bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_lora_loading(self):
        self.pipeline_4bit.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora")

        output = self.pipeline_4bit(
            prompt=self.prompt,
            control_image=Image.new(mode="RGB", size=(256, 256)),
            height=256,
            width=256,
            max_sequence_length=64,
            output_type="np",
            num_inference_steps=8,
            generator=torch.Generator().manual_seed(42),
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1636, 0.1675, 0.1982, 0.1743, 0.1809, 0.1936, 0.1743, 0.2095, 0.2139])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3, f"{out_slice=} != {expected_slice=}"


@is_quantization
@is_bitsandbytes
@require_torch_version_greater("2.7.1")
@require_bitsandbytes_version_greater("0.45.5")
class TestBnb4BitCompile(QuantCompileTests):
    @property
    def quantization_config(self):
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder_2"],
        )


@is_quantization
@is_bitsandbytes
@require_bitsandbytes_version_greater("0.43.2")
@require_accelerate
@require_torch
@require_torch_accelerator
@slow
class Base8bitTests:
    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only SD3 to test our module
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

    prompt = "a beautiful sunset amidst the mountains."
    num_inference_steps = 10
    seed = 0

    @pytest.fixture(autouse=True, scope="class")
    def _toggle_determinism(self):
        was_enabled = torch.are_deterministic_algorithms_enabled()
        if not was_enabled:
            torch.use_deterministic_algorithms(True)
        yield
        if not was_enabled:
            torch.use_deterministic_algorithms(False)


@require_transformers_version_greater("4.44.0")
class TestSlowBnb8bit(Base8bitTests):
    @pytest.fixture(autouse=True)
    def _setup_slow(self):
        gc.collect()
        backend_empty_cache(torch_device)

        mixed_int8_config = BitsAndBytesConfig(load_in_8bit=True)
        model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=mixed_int8_config, device_map=torch_device
        )
        self.pipeline_8bit = DiffusionPipeline.from_pretrained(
            self.model_name, transformer=model_8bit, torch_dtype=torch.float16
        )
        self.pipeline_8bit.enable_model_cpu_offload()
        yield
        del self.pipeline_8bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_quality(self):
        output = self.pipeline_8bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.0674, 0.0623, 0.0364, 0.0632, 0.0671, 0.0430, 0.0317, 0.0493, 0.0583])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-2

    def test_model_cpu_offload_raises_warning(self):
        model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=torch_device,
        )
        pipeline_8bit = DiffusionPipeline.from_pretrained(
            self.model_name, transformer=model_8bit, torch_dtype=torch.float16
        )
        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            pipeline_8bit.enable_model_cpu_offload()

        assert "has been loaded in `bitsandbytes` 8bit" in cap_logger.out

    def test_moving_to_cpu_throws_warning(self):
        model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=torch_device,
        )
        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            # Because `model.dtype` will return torch.float16 as SD3 transformer has
            # a conv layer as the first layer.
            _ = DiffusionPipeline.from_pretrained(
                self.model_name, transformer=model_8bit, torch_dtype=torch.float16
            ).to("cpu")

        assert "Pipelines loaded with `dtype=torch.float16`" in cap_logger.out

    def test_generate_quality_dequantize(self):
        r"""
        Test that loading the model and unquantize it produce correct results.
        """
        self.pipeline_8bit.transformer.dequantize()
        output = self.pipeline_8bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images

        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.0266, 0.0264, 0.0271, 0.0110, 0.0310, 0.0098, 0.0078, 0.0256, 0.0208])
        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-2

        # 8bit models cannot be offloaded to CPU.
        assert self.pipeline_8bit.transformer.device.type == torch_device
        # calling it again shouldn't be a problem
        _ = self.pipeline_8bit(
            prompt=self.prompt,
            num_inference_steps=2,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images

    @pytest.mark.xfail(
        condition=is_accelerate_version("<=", "1.1.1"),
        reason="Test will pass after https://github.com/huggingface/accelerate/pull/3223 is in a release.",
        strict=True,
    )
    def test_pipeline_cuda_placement_works_with_mixed_int8(self):
        transformer_8bit_config = BitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=transformer_8bit_config,
            torch_dtype=torch.float16,
            device_map=torch_device,
        )
        text_encoder_3_8bit_config = TranBitsAndBytesConfig(load_in_8bit=True)
        text_encoder_3_8bit = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_3",
            quantization_config=text_encoder_3_8bit_config,
            torch_dtype=torch.float16,
            device_map=torch_device,
        )

        # CUDA device placement works.
        device = torch_device if torch_device != "rocm" else "cuda"
        pipeline_8bit = DiffusionPipeline.from_pretrained(
            self.model_name,
            transformer=transformer_8bit,
            text_encoder_3=text_encoder_3_8bit,
            torch_dtype=torch.float16,
        ).to(device)

        # Check if inference works.
        _ = pipeline_8bit(self.prompt, max_sequence_length=20, num_inference_steps=2)

        del pipeline_8bit


@require_transformers_version_greater("4.44.0")
@require_big_accelerator
class TestSlowBnb8bitFlux(Base8bitTests):
    @pytest.fixture(autouse=True)
    def _setup_slow_flux(self):
        gc.collect()
        backend_empty_cache(torch_device)

        model_id = "hf-internal-testing/flux.1-dev-int8-pkg"
        t5_8bit = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2")
        transformer_8bit = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer")
        self.pipeline_8bit = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder_2=t5_8bit,
            transformer=transformer_8bit,
            torch_dtype=torch.float16,
        )
        self.pipeline_8bit.enable_model_cpu_offload()
        yield
        del self.pipeline_8bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_quality(self):
        # keep the resolution and max tokens to a lower number for faster execution.
        output = self.pipeline_8bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            height=256,
            width=256,
            max_sequence_length=64,
            output_type="np",
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.0574, 0.0554, 0.0581, 0.0686, 0.0676, 0.0759, 0.0757, 0.0803, 0.0930])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3

    @require_peft_version_greater("0.14.0")
    def test_lora_loading(self):
        self.pipeline_8bit.load_lora_weights(
            hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors"), adapter_name="hyper-sd"
        )
        self.pipeline_8bit.set_adapters("hyper-sd", adapter_weights=0.125)

        output = self.pipeline_8bit(
            prompt=self.prompt,
            height=256,
            width=256,
            max_sequence_length=64,
            output_type="np",
            num_inference_steps=8,
            generator=torch.manual_seed(42),
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()

        expected_slice = np.array([0.3916, 0.3916, 0.3887, 0.4243, 0.4155, 0.4233, 0.4570, 0.4531, 0.4248])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3


@require_transformers_version_greater("4.44.0")
@require_peft_backend
class TestSlowBnb8bitFluxControlWithLora(Base8bitTests):
    @pytest.fixture(autouse=True)
    def _setup_flux_control_lora(self):
        gc.collect()
        backend_empty_cache(torch_device)

        self.pipeline_8bit = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            quantization_config=PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["transformer", "text_encoder_2"],
            ),
            torch_dtype=torch.float16,
        )
        self.pipeline_8bit.enable_model_cpu_offload()
        yield
        del self.pipeline_8bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_lora_loading(self):
        self.pipeline_8bit.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora")

        output = self.pipeline_8bit(
            prompt=self.prompt,
            control_image=Image.new(mode="RGB", size=(256, 256)),
            height=256,
            width=256,
            max_sequence_length=64,
            output_type="np",
            num_inference_steps=8,
            generator=torch.Generator().manual_seed(42),
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()
        # Hardware-dependent: the Control LoRA dequantizes and expands `x_embedder`, and the error
        # accumulates over the 8 denoising steps enough that even different CUDA GPUs disagree, so
        # reference slices are stored per accelerator backend.
        expected_slices = Expectations(
            {
                (None, None): np.array([0.2029, 0.2136, 0.2268, 0.1921, 0.1997, 0.2185, 0.2021, 0.2183, 0.2292]),
                ("xpu", 5): np.array([0.0955, 0.1223, 0.1509, 0.0872, 0.1155, 0.1890, 0.0754, 0.1028, 0.2178]),
            }
        )
        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3, f"{out_slice=} != {expected_slice=}"


@is_quantization
@is_bitsandbytes
@require_torch_version_greater_equal("2.6.0")
@require_bitsandbytes_version_greater("0.48.0")
class TestBnb8BitCompile(QuantCompileTests):
    @property
    def quantization_config(self):
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=["transformer", "text_encoder_2"],
        )

    def test_torch_compile_with_cpu_offload(self):
        super()._test_torch_compile_with_cpu_offload(torch_dtype=torch.float16)


# ======================== TorchAO ========================


def _is_xpu_or_cuda_capability_atleast_8_9() -> bool:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major == 8:
            return minor >= 9
        return major >= 9
    elif torch.xpu.is_available():
        return True
    return False


# Model-level TorchAO tests live in `tests/models/testing_utils/quantization.py`
# (`TorchAoTesterMixin` / `TorchAoCompileTesterMixin`), wired into model test files via concrete
# classes (e.g. `TestFluxTransformerTorchAo`). Only pipeline-level coverage remains here.
# Slices for these tests have been obtained on our aws-g6e-xlarge-plus runners
@is_quantization
@is_torchao
@require_torch
@require_torch_accelerator
@require_torchao_version_greater_or_equal("0.15.0")
class TestTorchAo:
    @pytest.fixture(autouse=True)
    def _setup_torchao(self):
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_components(
        self, quantization_config: TorchAoConfig, model_id: str = "hf-internal-testing/tiny-flux-pipe"
    ):
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device: torch.device, seed: int = 0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        inputs = {
            "prompt": "an astronaut riding a horse in space",
            "height": 32,
            "width": 32,
            "num_inference_steps": 2,
            "output_type": "np",
            "generator": generator,
        }

        return inputs

    def _test_quant_type(self, quantization_config: TorchAoConfig, expected_slice: list[float], model_id: str):
        components = self.get_dummy_components(quantization_config, model_id)
        pipe = FluxPipeline(**components)
        pipe.to(device=torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]
        output_slice = output[-1, -1, -3:, -3:].flatten()

        assert np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3)

    def test_quantization(self):
        for model_id in ["hf-internal-testing/tiny-flux-pipe", "hf-internal-testing/tiny-flux-sharded"]:
            # fmt: off
            QUANTIZATION_TYPES_TO_TEST = [
                (Int4WeightOnlyConfig(version=2), np.array([0.4648, 0.5234, 0.5547, 0.4219, 0.4414, 0.6445, 0.4336, 0.4531, 0.5625])),
                (Int8DynamicActivationIntxWeightConfig(version=2), np.array([0.4688, 0.5195, 0.5547, 0.418, 0.4414, 0.6406, 0.4336, 0.4531, 0.5625])),
                (Int8WeightOnlyConfig(version=2), np.array([0.4648, 0.5195, 0.5547, 0.4199, 0.4414, 0.6445, 0.4316, 0.4531, 0.5625])),
                (Int8DynamicActivationInt8WeightConfig(version=2), np.array([0.4648, 0.5195, 0.5547, 0.4199, 0.4414, 0.6445, 0.4316, 0.4531, 0.5625])),
                (IntxWeightOnlyConfig(dtype=torch.uint4, group_size=16, version=2), np.array([0.4609, 0.5234, 0.5508, 0.4199, 0.4336, 0.6406, 0.4316, 0.4531, 0.5625])),
                (IntxWeightOnlyConfig(dtype=torch.uint7, group_size=16, version=2), np.array([0.4648, 0.5195, 0.5547, 0.4219, 0.4414, 0.6445, 0.4316, 0.4531, 0.5625])),
            ]

            if _is_xpu_or_cuda_capability_atleast_8_9():
                QUANTIZATION_TYPES_TO_TEST.extend([
                    (Float8WeightOnlyConfig(weight_dtype=torch.float8_e5m2), np.array([0.4590, 0.5273, 0.5547, 0.4219, 0.4375, 0.6406, 0.4316, 0.4512, 0.5625])),
                    (Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn), np.array([0.4648, 0.5234, 0.5547, 0.4219, 0.4414, 0.6406, 0.4316, 0.4531, 0.5625])),
                ])
            # fmt: on

            for quant_config, expected_slice in QUANTIZATION_TYPES_TO_TEST:
                quantization_config = TorchAoConfig(quant_type=quant_config, modules_to_not_convert=["x_embedder"])
                self._test_quant_type(quantization_config, expected_slice, model_id)

    def test_sequential_cpu_offload(self):
        r"""
        A test that checks if inference runs as expected when sequential cpu offloading is enabled.
        """
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_dummy_inputs(torch_device)
        _ = pipe(**inputs)

    @require_torchao_version_greater_or_equal("0.15.0")
    def test_aobase_config(self):
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        _ = pipe(**inputs)


@is_quantization
@is_torchao
@require_torchao_version_greater_or_equal("0.15.0")
class TestTorchAoCompile(QuantCompileTests):
    @property
    def quantization_config(self):
        return PipelineQuantizationConfig(
            quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig())},
        )

    def test_torch_compile_with_cpu_offload(self):
        pipe = self._init_pipeline(self.quantization_config, torch.bfloat16)
        pipe.enable_model_cpu_offload()
        # No compilation because it fails with:
        # RuntimeError: _apply(): Couldn't swap Linear.weight

        # small resolutions to ensure speedy execution.
        pipe("a dog", num_inference_steps=2, max_sequence_length=16, height=256, width=256)


# Slices for these tests have been obtained on our aws-g6e-xlarge-plus runners
@is_quantization
@is_torchao
@require_torch
@require_torch_accelerator
@require_torchao_version_greater_or_equal("0.15.0")
@slow
@nightly
class TestSlowTorchAo:
    @pytest.fixture(autouse=True)
    def _setup_slow_torchao(self):
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_components(self, quantization_config: TorchAoConfig):
        # This is just for convenience, so that we can modify it at one place for custom environments and locally testing
        cache_dir = None
        model_id = "black-forest-labs/FLUX.1-dev"
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
        tokenizer_2 = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", cache_dir=cache_dir)
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device: torch.device, seed: int = 0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        inputs = {
            "prompt": "an astronaut riding a horse in space",
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "output_type": "np",
            "generator": generator,
        }

        return inputs

    def _test_quant_type(self, quantization_config, expected_slice):
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.enable_model_cpu_offload()

        weight = pipe.transformer.transformer_blocks[0].ff.net[2].weight
        assert isinstance(weight, TorchAOBaseTensor)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0].flatten()
        output_slice = np.concatenate((output[:16], output[-16:]))
        assert np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3)

    def test_quantization(self):
        # fmt: off
        QUANTIZATION_TYPES_TO_TEST = [
            (Int8WeightOnlyConfig(), np.array([0.0505, 0.0742, 0.1367, 0.0429, 0.0585, 0.1386, 0.0585, 0.0703, 0.1367, 0.0566, 0.0703, 0.1464, 0.0546, 0.0703, 0.1425, 0.0546, 0.3535, 0.7578, 0.5000, 0.4062, 0.7656, 0.5117, 0.4121, 0.7656, 0.5117, 0.3984, 0.7578, 0.5234, 0.4023, 0.7382, 0.5390, 0.4570])),
            (Int8DynamicActivationInt8WeightConfig(), np.array([0.0546, 0.0761, 0.1386, 0.0488, 0.0644, 0.1425, 0.0605, 0.0742, 0.1406, 0.0625, 0.0722, 0.1523, 0.0625, 0.0742, 0.1503, 0.0605, 0.3886, 0.7968, 0.5507, 0.4492, 0.7890, 0.5351, 0.4316, 0.8007, 0.5390, 0.4179, 0.8281, 0.5820, 0.4531, 0.7812, 0.5703, 0.4921])),
        ]

        if _is_xpu_or_cuda_capability_atleast_8_9():
            QUANTIZATION_TYPES_TO_TEST.extend([
                (Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn), np.array([0.0546, 0.0722, 0.1328, 0.0468, 0.0585, 0.1367, 0.0605, 0.0703, 0.1328, 0.0625, 0.0703, 0.1445, 0.0585, 0.0703, 0.1406, 0.0605, 0.3496, 0.7109, 0.4843, 0.4042, 0.7226, 0.5000, 0.4160, 0.7031, 0.4824, 0.3886, 0.6757, 0.4667, 0.3710, 0.6679, 0.4902, 0.4238])),
            ])
        # fmt: on

        for quant_config, expected_slice in QUANTIZATION_TYPES_TO_TEST:
            quantization_config = TorchAoConfig(quant_type=quant_config, modules_to_not_convert=["x_embedder"])
            self._test_quant_type(quantization_config, expected_slice)
            gc.collect()
            backend_empty_cache(torch_device)
            backend_synchronize(torch_device)

    def test_serialization_int8wo(self):
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.enable_model_cpu_offload()

        weight = pipe.transformer.x_embedder.weight
        assert isinstance(weight, Int8Tensor)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0].flatten()[:128]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe.transformer.save_pretrained(tmp_dir, safe_serialization=False)
            pipe.remove_all_hooks()
            del pipe.transformer
            gc.collect()
            backend_empty_cache(torch_device)
            backend_synchronize(torch_device)
            transformer = FluxTransformer2DModel.from_pretrained(
                tmp_dir, torch_dtype=torch.bfloat16, use_safetensors=False
            )
            pipe.transformer = transformer
            pipe.enable_model_cpu_offload()

        weight = transformer.x_embedder.weight
        assert isinstance(weight, Int8Tensor)

        loaded_output = pipe(**inputs)[0].flatten()[:128]
        # Seems to require higher tolerance depending on which machine it is being run.
        # A difference of 0.06 in normalized pixel space (-1 to 1), corresponds to a difference of
        # 0.06 / 2 * 255 = 7.65 in pixel space (0 to 255). On our CI runners, the difference is about 0.04,
        # on DGX it is 0.06, and on audace it is 0.037. So, we are using a tolerance of 0.06 here.
        assert np.allclose(output, loaded_output, atol=0.06)


# ======================== GGUF ========================


# Model-level GGUF tests live in `tests/models/testing_utils/quantization.py`
# (`GGUFTesterMixin` / `GGUFCompileTesterMixin`) and backend-level ones (quantized parameter/layer
# inspection, memory use, CUDA kernels) in `tests/quantization/gguf/`. Only pipeline-level coverage
# remains here.
@is_quantization
@is_gguf
@nightly
@require_big_accelerator
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class GGUFPipelineTests:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)


class TestFluxGGUFPipeline(GGUFPipelineTests):
    ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
    model_cls = FluxTransformer2DModel
    torch_dtype = torch.bfloat16

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


class TestSD35LargeGGUFPipeline(GGUFPipelineTests):
    ckpt_path = "https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/blob/main/sd3.5_large-Q4_0.gguf"
    model_cls = SD3Transformer2DModel
    torch_dtype = torch.bfloat16

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
            prompt=prompt,
            num_inference_steps=2,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images[0]
        output_slice = output[:3, :3, :].flatten()
        expected_slices = Expectations(
            {
                ("xpu", 3): np.array(
                    [
                        0.16796875,
                        0.27929688,
                        0.28320312,
                        0.11328125,
                        0.27539062,
                        0.26171875,
                        0.10742188,
                        0.26367188,
                        0.26171875,
                        0.1484375,
                        0.2734375,
                        0.296875,
                        0.13476562,
                        0.2890625,
                        0.30078125,
                        0.1171875,
                        0.28125,
                        0.28125,
                        0.16015625,
                        0.31445312,
                        0.30078125,
                        0.15625,
                        0.32421875,
                        0.296875,
                        0.14453125,
                        0.30859375,
                        0.2890625,
                    ]
                ),
                ("cuda", 7): np.array(
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
                ),
            }
        )
        expected_slice = expected_slices.get_expectation()
        max_diff = numpy_cosine_similarity_distance(expected_slice, output_slice)
        assert max_diff < 1e-4


class TestSD35MediumGGUFPipeline(GGUFPipelineTests):
    ckpt_path = "https://huggingface.co/city96/stable-diffusion-3.5-medium-gguf/blob/main/sd3.5_medium-Q3_K_M.gguf"
    model_cls = SD3Transformer2DModel
    torch_dtype = torch.bfloat16

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


class TestAuraFlowGGUFPipeline(GGUFPipelineTests):
    ckpt_path = "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf"
    model_cls = AuraFlowTransformer2DModel
    torch_dtype = torch.bfloat16

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


@is_quantization
@is_gguf
@require_peft_backend
@nightly
@require_big_accelerator
@require_accelerate
@require_gguf_version_greater_or_equal("0.10.0")
class TestFluxControlLoRAGGUF:
    def test_lora_loading(self):
        ckpt_path = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)
        pipe.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora")

        prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
        control_image = load_image(
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/control_image_robot_canny.png"
        )

        output = pipe(
            prompt=prompt,
            control_image=control_image,
            height=256,
            width=256,
            num_inference_steps=10,
            guidance_scale=30.0,
            output_type="np",
            generator=torch.manual_seed(0),
        ).images

        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.8047, 0.8359, 0.8711, 0.6875, 0.7070, 0.7383, 0.5469, 0.5820, 0.6641])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        assert max_diff < 1e-3


@is_quantization
@is_gguf
@require_torch_version_greater("2.7.1")
class TestGGUFCompile(QuantCompileTests):
    torch_dtype = torch.bfloat16
    gguf_ckpt = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q2_K.gguf"

    @property
    def quantization_config(self):
        return GGUFQuantizationConfig(compute_dtype=self.torch_dtype)

    def _init_pipeline(self, *args, **kwargs):
        transformer = FluxTransformer2DModel.from_single_file(
            self.gguf_ckpt, quantization_config=self.quantization_config, torch_dtype=self.torch_dtype
        )
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.torch_dtype
        )
        return pipe


# ======================== NVIDIA ModelOpt ========================


# Model-level ModelOpt tests live in `tests/models/testing_utils/quantization.py`
# (`ModelOptTesterMixin` / `ModelOptCompileTesterMixin`), wired into model test files via concrete
# classes (e.g. `TestSD3TransformerModelOpt`). Only pipeline-level coverage remains here.
@is_quantization
@is_modelopt
@nightly
@require_big_accelerator
@require_accelerate
@require_modelopt_version_greater_or_equal("0.33.1")
class TestModelOptFP8:
    model_id = "hf-internal-testing/tiny-sd3-pipe"

    @pytest.fixture(autouse=True)
    def _setup(self):
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()
        yield
        backend_reset_peak_memory_stats(torch_device)
        backend_empty_cache(torch_device)
        gc.collect()

    def test_model_cpu_offload(self):
        transformer = SD3Transformer2DModel.from_pretrained(
            self.model_id,
            quantization_config=NVIDIAModelOptConfig(quant_type="FP8"),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        _ = pipe("a cat holding a sign that says hello", num_inference_steps=2)
