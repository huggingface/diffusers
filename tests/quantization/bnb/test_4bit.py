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
import gc
import os
import tempfile
import unittest

import numpy as np
import pytest
import safetensors.torch
from huggingface_hub import hf_hub_download
from PIL import Image

from diffusers import (
    BitsAndBytesConfig,
    DiffusionPipeline,
    FluxControlPipeline,
    FluxTransformer2DModel,
    SD3Transformer2DModel,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import is_accelerate_version, logging

from ...testing_utils import (
    CaptureLogger,
    backend_empty_cache,
    is_bitsandbytes_available,
    is_torch_available,
    is_transformers_available,
    load_pt,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_bitsandbytes_version_greater,
    require_peft_backend,
    require_torch,
    require_torch_accelerator,
    require_torch_version_greater,
    require_transformers_version_greater,
    slow,
    torch_device,
)
from ..test_torch_compile_utils import QuantCompileTests


def get_some_linear_layer(model):
    if model.__class__.__name__ in ["SD3Transformer2DModel", "FluxTransformer2DModel"]:
        return model.transformer_blocks[0].attn.to_q
    else:
        return NotImplementedError("Don't know what layer to retrieve here.")


if is_transformers_available():
    from transformers import BitsAndBytesConfig as BnbConfig
    from transformers import T5EncoderModel

if is_torch_available():
    import torch

    from ..utils import LoRALayer, get_memory_consumption_stat


if is_bitsandbytes_available():
    import bitsandbytes as bnb

    from diffusers.quantizers.bitsandbytes.utils import replace_with_bnb_linear


@require_bitsandbytes_version_greater("0.43.2")
@require_accelerate
@require_torch
@require_torch_accelerator
@slow
class Base4bitTests(unittest.TestCase):
    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only SD3 to test our module
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

    # This was obtained on audace so the number might slightly change
    expected_rel_difference = 3.69

    expected_memory_saving_ratio = 0.8

    prompt = "a beautiful sunset amidst the mountains."
    num_inference_steps = 10
    seed = 0

    @classmethod
    def setUpClass(cls):
        cls.is_deterministic_enabled = torch.are_deterministic_algorithms_enabled()
        if not cls.is_deterministic_enabled:
            torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        if not cls.is_deterministic_enabled:
            torch.use_deterministic_algorithms(False)

    def get_dummy_inputs(self):
        prompt_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/prompt_embeds.pt",
            torch_device,
        )
        pooled_prompt_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/pooled_prompt_embeds.pt",
            torch_device,
        )
        latent_model_input = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/latent_model_input.pt",
            torch_device,
        )

        input_dict_for_transformer = {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": torch.Tensor([1.0]),
            "return_dict": False,
        }
        return input_dict_for_transformer


class BnB4BitBasicTests(Base4bitTests):
    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

        # Models
        self.model_fp16 = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=torch.float16
        )
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config, device_map=torch_device
        )

    def tearDown(self):
        if hasattr(self, "model_fp16"):
            del self.model_fp16
        if hasattr(self, "model_4bit"):
            del self.model_4bit

        gc.collect()
        backend_empty_cache(torch_device)

    def test_quantization_num_parameters(self):
        r"""
        Test if the number of returned parameters is correct
        """
        num_params_4bit = self.model_4bit.num_parameters()
        num_params_fp16 = self.model_fp16.num_parameters()

        self.assertEqual(num_params_4bit, num_params_fp16)

    def test_quantization_config_json_serialization(self):
        r"""
        A simple test to check if the quantization config is correctly serialized and deserialized
        """
        config = self.model_4bit.config

        self.assertTrue("quantization_config" in config)

        _ = config["quantization_config"].to_dict()
        _ = config["quantization_config"].to_diff_dict()

        _ = config["quantization_config"].to_json_string()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()

        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.expected_rel_difference, delta=1e-2)
        linear = get_some_linear_layer(self.model_4bit)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Params4bit)

    def test_model_memory_usage(self):
        # Delete to not let anything interfere.
        del self.model_4bit, self.model_fp16

        # Re-instantiate.
        inputs = self.get_dummy_inputs()
        inputs = {
            k: v.to(device=torch_device, dtype=torch.float16) for k, v in inputs.items() if not isinstance(v, bool)
        }
        model_fp16 = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=torch.float16
        ).to(torch_device)
        unquantized_model_memory = get_memory_consumption_stat(model_fp16, inputs)
        del model_fp16

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config, torch_dtype=torch.float16
        )
        quantized_model_memory = get_memory_consumption_stat(model_4bit, inputs)
        assert unquantized_model_memory / quantized_model_memory >= self.expected_memory_saving_ratio

    def test_original_dtype(self):
        r"""
        A simple test to check if the model successfully stores the original dtype
        """
        self.assertTrue("_pre_quantization_dtype" in self.model_4bit.config)
        self.assertFalse("_pre_quantization_dtype" in self.model_fp16.config)
        self.assertTrue(self.model_4bit.config["_pre_quantization_dtype"] == torch.float16)

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
        fp32_modules = SD3Transformer2DModel._keep_in_fp32_modules
        SD3Transformer2DModel._keep_in_fp32_modules = ["proj_out"]

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config, device_map=torch_device
        )

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    self.assertTrue(module.weight.dtype == torch.float32)
                else:
                    # 4-bit parameters are packed in uint8 variables
                    self.assertTrue(module.weight.dtype == torch.uint8)

        # test if inference works.
        with torch.no_grad() and torch.amp.autocast(torch_device, dtype=torch.float16):
            input_dict_for_transformer = self.get_dummy_inputs()
            model_inputs = {
                k: v.to(device=torch_device) for k, v in input_dict_for_transformer.items() if not isinstance(v, bool)
            }
            model_inputs.update({k: v for k, v in input_dict_for_transformer.items() if k not in model_inputs})
            _ = model(**model_inputs)

        SD3Transformer2DModel._keep_in_fp32_modules = fp32_modules

    def test_linear_are_4bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()

        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ["proj_out"]:
                    # 4-bit parameters are packed in uint8 variables
                    self.assertTrue(module.weight.dtype == torch.uint8)

    def test_config_from_pretrained(self):
        transformer_4bit = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/flux.1-dev-nf4-pkg", subfolder="transformer"
        )
        linear = get_some_linear_layer(transformer_4bit)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Params4bit)
        self.assertTrue(hasattr(linear.weight, "quant_state"))
        self.assertTrue(linear.weight.quant_state.__class__ == bnb.functional.QuantState)

    def test_device_assignment(self):
        mem_before = self.model_4bit.get_memory_footprint()

        # Move to CPU
        self.model_4bit.to("cpu")
        self.assertEqual(self.model_4bit.device.type, "cpu")
        self.assertAlmostEqual(self.model_4bit.get_memory_footprint(), mem_before)

        # Move back to CUDA device
        for device in [0, f"{torch_device}", f"{torch_device}:0", "call()"]:
            if device == "call()":
                self.model_4bit.to(f"{torch_device}:0")
            else:
                self.model_4bit.to(device)
            self.assertEqual(self.model_4bit.device, torch.device(0))
            self.assertAlmostEqual(self.model_4bit.get_memory_footprint(), mem_before)
            self.model_4bit.to("cpu")

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after converting it in 4-bit will throw an error.
        Checks also if other models are casted correctly. Device placement, however, is supported.
        """
        with self.assertRaises(ValueError):
            # Tries with a `dtype`
            self.model_4bit.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device` and `dtype`
            self.model_4bit.to(device=f"{torch_device}:0", dtype=torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a cast
            self.model_4bit.float()

        with self.assertRaises(ValueError):
            # Tries with a cast
            self.model_4bit.half()

        # This should work
        self.model_4bit.to(torch_device)

        # Test if we did not break anything
        self.model_fp16 = self.model_fp16.to(dtype=torch.float32, device=torch_device)
        input_dict_for_transformer = self.get_dummy_inputs()
        model_inputs = {
            k: v.to(dtype=torch.float32, device=torch_device)
            for k, v in input_dict_for_transformer.items()
            if not isinstance(v, bool)
        }
        model_inputs.update({k: v for k, v in input_dict_for_transformer.items() if k not in model_inputs})
        with torch.no_grad():
            _ = self.model_fp16(**model_inputs)

        # Check this does not throw an error
        _ = self.model_fp16.to("cpu")

        # Check this does not throw an error
        _ = self.model_fp16.half()

        # Check this does not throw an error
        _ = self.model_fp16.float()

        # Check that this does not throw an error
        _ = self.model_fp16.to(torch_device)

    def test_bnb_4bit_wrong_config(self):
        r"""
        Test whether creating a bnb config with unsupported values leads to errors.
        """
        with self.assertRaises(ValueError):
            _ = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_storage="add")

    def test_bnb_4bit_errors_loading_incorrect_state_dict(self):
        r"""
        Test if loading with an incorrect state dict raises an error.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            nf4_config = BitsAndBytesConfig(load_in_4bit=True)
            model_4bit = SD3Transformer2DModel.from_pretrained(
                self.model_name, subfolder="transformer", quantization_config=nf4_config, device_map=torch_device
            )
            model_4bit.save_pretrained(tmpdirname)
            del model_4bit

            with self.assertRaises(ValueError) as err_context:
                state_dict = safetensors.torch.load_file(
                    os.path.join(tmpdirname, "diffusion_pytorch_model.safetensors")
                )

                # corrupt the state dict
                key_to_target = "context_embedder.weight"  # can be other keys too.
                compatible_param = state_dict[key_to_target]
                corrupted_param = torch.randn(compatible_param.shape[0] - 1, 1)
                state_dict[key_to_target] = bnb.nn.Params4bit(corrupted_param, requires_grad=False)
                safetensors.torch.save_file(
                    state_dict, os.path.join(tmpdirname, "diffusion_pytorch_model.safetensors")
                )

                _ = SD3Transformer2DModel.from_pretrained(tmpdirname)

            assert key_to_target in str(err_context.exception)

    def test_bnb_4bit_logs_warning_for_no_quantization(self):
        model_with_no_linear = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3), torch.nn.ReLU())
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        logger = logging.get_logger("diffusers.quantizers.bitsandbytes.utils")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            _ = replace_with_bnb_linear(model_with_no_linear, quantization_config=quantization_config)
        assert (
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            in cap_logger.out
        )


class BnB4BitTrainingTests(Base4bitTests):
    def setUp(self):
        gc.collect()
        backend_empty_cache(torch_device)

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config, device_map=torch_device
        )

    def test_training(self):
        # Step 1: freeze all parameters
        for param in self.model_4bit.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        # Step 2: add adapters
        for _, module in self.model_4bit.named_modules():
            if "Attention" in repr(type(module)):
                module.to_k = LoRALayer(module.to_k, rank=4)
                module.to_q = LoRALayer(module.to_q, rank=4)
                module.to_v = LoRALayer(module.to_v, rank=4)

        # Step 3: dummy batch
        input_dict_for_transformer = self.get_dummy_inputs()
        model_inputs = {
            k: v.to(device=torch_device) for k, v in input_dict_for_transformer.items() if not isinstance(v, bool)
        }
        model_inputs.update({k: v for k, v in input_dict_for_transformer.items() if k not in model_inputs})

        # Step 4: Check if the gradient is not None
        with torch.amp.autocast(torch_device, dtype=torch.float16):
            out = self.model_4bit(**model_inputs)[0]
            out.norm().backward()

        for module in self.model_4bit.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)
                self.assertTrue(module.adapter[1].weight.grad.norm().item() > 0)


@require_transformers_version_greater("4.44.0")
class SlowBnb4BitTests(Base4bitTests):
    def setUp(self) -> None:
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

    def tearDown(self):
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
        self.assertTrue(max_diff < 1e-2)

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
        self.assertTrue(max_diff < 1e-3)

        # Since we offloaded the `pipeline_4bit.transformer` to CPU (result of `enable_model_cpu_offload()), check
        # the following.
        self.assertTrue(self.pipeline_4bit.transformer.device.type == "cpu")
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
        text_encoder_3_nf4_config = BnbConfig(
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

    def test_device_map(self):
        """
        Test if the quantized model is working properly with "auto".
        cpu/disk offloading as well doesn't work with bnb.
        """

        def get_dummy_tensor_inputs(device=None, seed: int = 0):
            batch_size = 1
            num_latent_channels = 4
            num_image_channels = 3
            height = width = 4
            sequence_length = 48
            embedding_dim = 32

            torch.manual_seed(seed)
            hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(
                device, dtype=torch.bfloat16
            )
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

        inputs = get_dummy_tensor_inputs(torch_device)
        expected_slice = np.array(
            [0.47070312, 0.00390625, -0.03662109, -0.19628906, -0.53125, 0.5234375, -0.17089844, -0.59375, 0.578125]
        )

        # non sharded
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        weight = quantized_model.transformer_blocks[0].ff.net[2].weight
        self.assertTrue(isinstance(weight, bnb.nn.modules.Params4bit))

        output = quantized_model(**inputs)[0]
        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(numpy_cosine_similarity_distance(output_slice, expected_slice) < 1e-3)

        # sharded

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-sharded",
            subfolder="transformer",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        weight = quantized_model.transformer_blocks[0].ff.net[2].weight
        self.assertTrue(isinstance(weight, bnb.nn.modules.Params4bit))

        output = quantized_model(**inputs)[0]
        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()

        self.assertTrue(numpy_cosine_similarity_distance(output_slice, expected_slice) < 1e-3)


@require_transformers_version_greater("4.44.0")
class SlowBnb4BitFluxTests(Base4bitTests):
    def setUp(self) -> None:
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

    def tearDown(self):
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
        self.assertTrue(max_diff < 1e-3)

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
        self.assertTrue(max_diff < 1e-3)


@require_transformers_version_greater("4.44.0")
@require_peft_backend
class SlowBnb4BitFluxControlWithLoraTests(Base4bitTests):
    def setUp(self) -> None:
        gc.collect()
        backend_empty_cache(torch_device)

        self.pipeline_4bit = FluxControlPipeline.from_pretrained("eramth/flux-4bit", torch_dtype=torch.float16)
        self.pipeline_4bit.enable_model_cpu_offload()

    def tearDown(self):
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
        self.assertTrue(max_diff < 1e-3, msg=f"{out_slice=} != {expected_slice=}")


@slow
class BaseBnb4BitSerializationTests(Base4bitTests):
    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_serialization(self, quant_type="nf4", double_quant=True, safe_serialization=True):
        r"""
        Test whether it is possible to serialize a model in 4-bit. Uses most typical params as default.
        See ExtendedSerializationTest class for more params combinations.
        """

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_0 = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=self.quantization_config,
            device_map=torch_device,
        )
        self.assertTrue("_pre_quantization_dtype" in model_0.config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_0.save_pretrained(tmpdirname, safe_serialization=safe_serialization)

            config = SD3Transformer2DModel.load_config(tmpdirname)
            self.assertTrue("quantization_config" in config)
            self.assertTrue("_pre_quantization_dtype" not in config)

            model_1 = SD3Transformer2DModel.from_pretrained(tmpdirname)

        # checking quantized linear module weight
        linear = get_some_linear_layer(model_1)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Params4bit)
        self.assertTrue(hasattr(linear.weight, "quant_state"))
        self.assertTrue(linear.weight.quant_state.__class__ == bnb.functional.QuantState)

        # checking memory footpring
        self.assertAlmostEqual(model_0.get_memory_footprint() / model_1.get_memory_footprint(), 1, places=2)

        # Matching all parameters and their quant_state items:
        d0 = dict(model_0.named_parameters())
        d1 = dict(model_1.named_parameters())
        self.assertTrue(d0.keys() == d1.keys())

        for k in d0.keys():
            self.assertTrue(d0[k].shape == d1[k].shape)
            self.assertTrue(d0[k].device.type == d1[k].device.type)
            self.assertTrue(d0[k].device == d1[k].device)
            self.assertTrue(d0[k].dtype == d1[k].dtype)
            self.assertTrue(torch.equal(d0[k], d1[k].to(d0[k].device)))

            if isinstance(d0[k], bnb.nn.modules.Params4bit):
                for v0, v1 in zip(
                    d0[k].quant_state.as_dict().values(),
                    d1[k].quant_state.as_dict().values(),
                ):
                    if isinstance(v0, torch.Tensor):
                        self.assertTrue(torch.equal(v0, v1.to(v0.device)))
                    else:
                        self.assertTrue(v0 == v1)

        # comparing forward() outputs
        dummy_inputs = self.get_dummy_inputs()
        inputs = {k: v.to(torch_device) for k, v in dummy_inputs.items() if isinstance(v, torch.Tensor)}
        inputs.update({k: v for k, v in dummy_inputs.items() if k not in inputs})
        out_0 = model_0(**inputs)[0]
        out_1 = model_1(**inputs)[0]
        self.assertTrue(torch.equal(out_0, out_1))


class ExtendedSerializationTest(BaseBnb4BitSerializationTests):
    """
    tests more combinations of parameters
    """

    def test_nf4_single_unsafe(self):
        self.test_serialization(quant_type="nf4", double_quant=False, safe_serialization=False)

    def test_nf4_single_safe(self):
        self.test_serialization(quant_type="nf4", double_quant=False, safe_serialization=True)

    def test_nf4_double_unsafe(self):
        self.test_serialization(quant_type="nf4", double_quant=True, safe_serialization=False)

    # nf4 double safetensors quantization is tested in test_serialization() method from the parent class

    def test_fp4_single_unsafe(self):
        self.test_serialization(quant_type="fp4", double_quant=False, safe_serialization=False)

    def test_fp4_single_safe(self):
        self.test_serialization(quant_type="fp4", double_quant=False, safe_serialization=True)

    def test_fp4_double_unsafe(self):
        self.test_serialization(quant_type="fp4", double_quant=True, safe_serialization=False)

    def test_fp4_double_safe(self):
        self.test_serialization(quant_type="fp4", double_quant=True, safe_serialization=True)


@require_torch_version_greater("2.7.1")
@require_bitsandbytes_version_greater("0.45.5")
class Bnb4BitCompileTests(QuantCompileTests, unittest.TestCase):
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

    @require_bitsandbytes_version_greater("0.46.1")
    def test_torch_compile(self):
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        super().test_torch_compile()

    def test_torch_compile_with_group_offload_leaf(self):
        super()._test_torch_compile_with_group_offload_leaf(use_stream=True)
