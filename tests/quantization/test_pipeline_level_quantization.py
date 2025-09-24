# coding=utf-8
# Copyright 2025 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import tempfile
import unittest

import torch
from parameterized import parameterized

from diffusers import BitsAndBytesConfig, DiffusionPipeline, QuantoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import logging

from ..testing_utils import (
    CaptureLogger,
    is_transformers_available,
    require_accelerate,
    require_bitsandbytes_version_greater,
    require_quanto,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_transformers_available():
    from transformers import BitsAndBytesConfig as TranBitsAndBytesConfig
else:
    TranBitsAndBytesConfig = None


@require_bitsandbytes_version_greater("0.43.2")
@require_quanto
@require_accelerate
@require_torch
@require_torch_accelerator
@slow
class PipelineQuantizationTests(unittest.TestCase):
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
                self.assertTrue(getattr(component.config, "quantization_config", None) is not None)
                quantization_config = component.config.quantization_config
                self.assertTrue(quantization_config.load_in_4bit)
                self.assertTrue(quantization_config.quant_method == "bitsandbytes")

        _ = pipe(self.prompt, num_inference_steps=self.num_inference_steps)

    def test_quant_config_set_correctly_through_granular(self):
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": QuantoConfig(weights_dtype="int8"),
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
                self.assertTrue(getattr(component.config, "quantization_config", None) is not None)
                quantization_config = component.config.quantization_config

                if name == "text_encoder_2":
                    self.assertTrue(quantization_config.load_in_4bit)
                    self.assertTrue(quantization_config.quant_method == "bitsandbytes")
                else:
                    self.assertTrue(quantization_config.quant_method == "quanto")

        _ = pipe(self.prompt, num_inference_steps=self.num_inference_steps)

    def test_raises_error_for_invalid_config(self):
        with self.assertRaises(ValueError) as err_context:
            _ = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": QuantoConfig(weights_dtype="int8"),
                    "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
                },
                quant_backend="bitsandbytes_4bit",
            )

        self.assertTrue(
            str(err_context.exception)
            == "Both `quant_backend` and `quant_mapping` cannot be specified at the same time."
        )

    def test_validation_for_kwargs(self):
        components_to_quantize = ["transformer", "text_encoder_2"]
        with self.assertRaises(ValueError) as err_context:
            _ = PipelineQuantizationConfig(
                quant_backend="quanto",
                quant_kwargs={"weights_dtype": "int8"},
                components_to_quantize=components_to_quantize,
            )

        self.assertTrue(
            "The signatures of the __init__ methods of the quantization config classes" in str(err_context.exception)
        )

    def test_raises_error_for_wrong_config_class(self):
        quant_config = {
            "transformer": QuantoConfig(weights_dtype="int8"),
            "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
        }
        with self.assertRaises(ValueError) as err_context:
            _ = DiffusionPipeline.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        self.assertTrue(
            str(err_context.exception) == "`quantization_config` must be an instance of `PipelineQuantizationConfig`."
        )

    def test_validation_for_mapping(self):
        with self.assertRaises(ValueError) as err_context:
            _ = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusionPipeline(),
                    "text_encoder_2": TranBitsAndBytesConfig(load_in_4bit=True, compute_dtype=torch.bfloat16),
                }
            )

        self.assertTrue("Provided config for module_name=transformer could not be found" in str(err_context.exception))

    def test_saving_loading(self):
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": QuantoConfig(weights_dtype="int8"),
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
                self.assertTrue(getattr(component.config, "quantization_config", None) is not None)
                quantization_config = component.config.quantization_config

                if name == "text_encoder_2":
                    self.assertTrue(quantization_config.load_in_4bit)
                    self.assertTrue(quantization_config.quant_method == "bitsandbytes")
                else:
                    self.assertTrue(quantization_config.quant_method == "quanto")

        output_2 = loaded_pipe(**pipe_inputs, generator=torch.manual_seed(self.seed)).images

        self.assertTrue(torch.allclose(output_1, output_2))

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
                    "transformer": QuantoConfig("int8"),
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
        self.assertTrue(invalid_component in cap_logger.out)

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
                self.assertTrue(not hasattr(component.config, "quantization_config"))

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
        self.assertTrue(getattr(pipe, "quantization_config", None) is not None)
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
        self.assertTrue(actual_data == expected_data)

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
                self.assertTrue(hasattr(component.config, "quantization_config"))
