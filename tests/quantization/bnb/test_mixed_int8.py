# coding=utf-8
# Copyright 2024 The HuggingFace Team Inc.
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
import tempfile
import unittest

import numpy as np
import pytest

from diffusers import BitsAndBytesConfig, DiffusionPipeline, FluxTransformer2DModel, SD3Transformer2DModel, logging
from diffusers.utils import is_accelerate_version
from diffusers.utils.testing_utils import (
    CaptureLogger,
    is_bitsandbytes_available,
    is_torch_available,
    is_transformers_available,
    load_pt,
    numpy_cosine_similarity_distance,
    require_accelerate,
    require_bitsandbytes_version_greater,
    require_torch,
    require_torch_gpu,
    require_transformers_version_greater,
    slow,
    torch_device,
)


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
    import torch.nn as nn

    class LoRALayer(nn.Module):
        """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only

        Taken from
        https://github.com/huggingface/transformers/blob/566302686a71de14125717dea9a6a45b24d42b37/tests/quantization/bnb/test_8bit.py#L62C5-L78C77
        """

        def __init__(self, module: nn.Module, rank: int):
            super().__init__()
            self.module = module
            self.adapter = nn.Sequential(
                nn.Linear(module.in_features, rank, bias=False),
                nn.Linear(rank, module.out_features, bias=False),
            )
            small_std = (2.0 / (5 * min(module.in_features, module.out_features))) ** 0.5
            nn.init.normal_(self.adapter[0].weight, std=small_std)
            nn.init.zeros_(self.adapter[1].weight)
            self.adapter.to(module.weight.device)

        def forward(self, input, *args, **kwargs):
            return self.module(input, *args, **kwargs) + self.adapter(input)


if is_bitsandbytes_available():
    import bitsandbytes as bnb


@require_bitsandbytes_version_greater("0.43.2")
@require_accelerate
@require_torch
@require_torch_gpu
@slow
class Base8bitTests(unittest.TestCase):
    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only SD3 to test our module
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

    # This was obtained on audace so the number might slightly change
    expected_rel_difference = 1.94

    prompt = "a beautiful sunset amidst the mountains."
    num_inference_steps = 10
    seed = 0

    def get_dummy_inputs(self):
        prompt_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/prompt_embeds.pt"
        )
        pooled_prompt_embeds = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/pooled_prompt_embeds.pt"
        )
        latent_model_input = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/latent_model_input.pt"
        )

        input_dict_for_transformer = {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": torch.Tensor([1.0]),
            "return_dict": False,
        }
        return input_dict_for_transformer


class BnB8bitBasicTests(Base8bitTests):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        # Models
        self.model_fp16 = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=torch.float16
        )
        mixed_int8_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=mixed_int8_config
        )

    def tearDown(self):
        del self.model_fp16
        del self.model_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_quantization_num_parameters(self):
        r"""
        Test if the number of returned parameters is correct
        """
        num_params_8bit = self.model_8bit.num_parameters()
        num_params_fp16 = self.model_fp16.num_parameters()

        self.assertEqual(num_params_8bit, num_params_fp16)

    def test_quantization_config_json_serialization(self):
        r"""
        A simple test to check if the quantization config is correctly serialized and deserialized
        """
        config = self.model_8bit.config

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
        mem_8bit = self.model_8bit.get_memory_footprint()

        self.assertAlmostEqual(mem_fp16 / mem_8bit, self.expected_rel_difference, delta=1e-2)
        linear = get_some_linear_layer(self.model_8bit)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Int8Params)

    def test_original_dtype(self):
        r"""
        A simple test to check if the model succesfully stores the original dtype
        """
        self.assertTrue("_pre_quantization_dtype" in self.model_8bit.config)
        self.assertFalse("_pre_quantization_dtype" in self.model_fp16.config)
        self.assertTrue(self.model_8bit.config["_pre_quantization_dtype"] == torch.float16)

    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32.
        Also ensures if inference works.
        """
        fp32_modules = SD3Transformer2DModel._keep_in_fp32_modules
        SD3Transformer2DModel._keep_in_fp32_modules = ["proj_out"]

        mixed_int8_config = BitsAndBytesConfig(load_in_8bit=True)
        model = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=mixed_int8_config
        )

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name in model._keep_in_fp32_modules:
                    self.assertTrue(module.weight.dtype == torch.float32)
                else:
                    # 8-bit parameters are packed in int8 variables
                    self.assertTrue(module.weight.dtype == torch.int8)

        # test if inference works.
        with torch.no_grad() and torch.amp.autocast("cuda", dtype=torch.float16):
            input_dict_for_transformer = self.get_dummy_inputs()
            model_inputs = {
                k: v.to(device=torch_device) for k, v in input_dict_for_transformer.items() if not isinstance(v, bool)
            }
            model_inputs.update({k: v for k, v in input_dict_for_transformer.items() if k not in model_inputs})
            _ = model(**model_inputs)

        SD3Transformer2DModel._keep_in_fp32_modules = fp32_modules

    def test_linear_are_8bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        self.model_fp16.get_memory_footprint()
        self.model_8bit.get_memory_footprint()

        for name, module in self.model_8bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in ["proj_out"]:
                    # 8-bit parameters are packed in int8 variables
                    self.assertTrue(module.weight.dtype == torch.int8)

    def test_llm_skip(self):
        r"""
        A simple test to check if `llm_int8_skip_modules` works as expected
        """
        config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["proj_out"])
        model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=config
        )
        linear = get_some_linear_layer(model_8bit)
        self.assertTrue(linear.weight.dtype == torch.int8)
        self.assertTrue(isinstance(linear, bnb.nn.Linear8bitLt))

        self.assertTrue(isinstance(model_8bit.proj_out, nn.Linear))
        self.assertTrue(model_8bit.proj_out.weight.dtype != torch.int8)

    def test_config_from_pretrained(self):
        transformer_8bit = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/flux.1-dev-int8-pkg", subfolder="transformer"
        )
        linear = get_some_linear_layer(transformer_8bit)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Int8Params)
        self.assertTrue(hasattr(linear.weight, "SCB"))

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after converting it in 8-bit will throw an error.
        Checks also if other models are casted correctly.
        """
        with self.assertRaises(ValueError):
            # Tries with `str`
            self.model_8bit.to("cpu")

        with self.assertRaises(ValueError):
            # Tries with a `dtype``
            self.model_8bit.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_8bit.to(torch.device("cuda:0"))

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_8bit.float()

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_8bit.half()

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
        _ = self.model_fp16.cuda()


class BnB8bitTrainingTests(Base8bitTests):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        mixed_int8_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=mixed_int8_config
        )

    def test_training(self):
        # Step 1: freeze all parameters
        for param in self.model_8bit.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        # Step 2: add adapters
        for _, module in self.model_8bit.named_modules():
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
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = self.model_8bit(**model_inputs)[0]
            out.norm().backward()

        for module in self.model_8bit.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)
                self.assertTrue(module.adapter[1].weight.grad.norm().item() > 0)


@require_transformers_version_greater("4.44.0")
class SlowBnb8bitTests(Base8bitTests):
    def setUp(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

        mixed_int8_config = BitsAndBytesConfig(load_in_8bit=True)
        model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=mixed_int8_config
        )
        self.pipeline_8bit = DiffusionPipeline.from_pretrained(
            self.model_name, transformer=model_8bit, torch_dtype=torch.float16
        )
        self.pipeline_8bit.enable_model_cpu_offload()

    def tearDown(self):
        del self.pipeline_8bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_quality(self):
        output = self.pipeline_8bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images
        out_slice = output[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.0376, 0.0359, 0.0015, 0.0449, 0.0479, 0.0098, 0.0083, 0.0295, 0.0295])

        max_diff = numpy_cosine_similarity_distance(expected_slice, out_slice)
        self.assertTrue(max_diff < 1e-2)

    def test_model_cpu_offload_raises_warning(self):
        model_8bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
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
            self.model_name, subfolder="transformer", quantization_config=BitsAndBytesConfig(load_in_8bit=True)
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
        self.assertTrue(max_diff < 1e-2)

        # 8bit models cannot be offloaded to CPU.
        self.assertTrue(self.pipeline_8bit.transformer.device.type == "cuda")
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
        )
        text_encoder_3_8bit_config = BnbConfig(load_in_8bit=True)
        text_encoder_3_8bit = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_3",
            quantization_config=text_encoder_3_8bit_config,
            torch_dtype=torch.float16,
        )
        # CUDA device placement works.
        pipeline_8bit = DiffusionPipeline.from_pretrained(
            self.model_name,
            transformer=transformer_8bit,
            text_encoder_3=text_encoder_3_8bit,
            torch_dtype=torch.float16,
        ).to("cuda")

        # Check if inference works.
        _ = pipeline_8bit("table", max_sequence_length=20, num_inference_steps=2)

        del pipeline_8bit


@require_transformers_version_greater("4.44.0")
class SlowBnb8bitFluxTests(Base8bitTests):
    def setUp(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

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

    def tearDown(self):
        del self.pipeline_8bit

        gc.collect()
        torch.cuda.empty_cache()

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
        self.assertTrue(max_diff < 1e-3)


@slow
class BaseBnb8bitSerializationTests(Base8bitTests):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        self.model_0 = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=quantization_config
        )

    def tearDown(self):
        del self.model_0

        gc.collect()
        torch.cuda.empty_cache()

    def test_serialization(self):
        r"""
        Test whether it is possible to serialize a model in 8-bit. Uses most typical params as default.
        """
        self.assertTrue("_pre_quantization_dtype" in self.model_0.config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_0.save_pretrained(tmpdirname)

            config = SD3Transformer2DModel.load_config(tmpdirname)
            self.assertTrue("quantization_config" in config)
            self.assertTrue("_pre_quantization_dtype" not in config)

            model_1 = SD3Transformer2DModel.from_pretrained(tmpdirname)

        # checking quantized linear module weight
        linear = get_some_linear_layer(model_1)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Int8Params)
        self.assertTrue(hasattr(linear.weight, "SCB"))

        # checking memory footpring
        self.assertAlmostEqual(self.model_0.get_memory_footprint() / model_1.get_memory_footprint(), 1, places=2)

        # Matching all parameters and their quant_state items:
        d0 = dict(self.model_0.named_parameters())
        d1 = dict(model_1.named_parameters())
        self.assertTrue(d0.keys() == d1.keys())

        # comparing forward() outputs
        dummy_inputs = self.get_dummy_inputs()
        inputs = {k: v.to(torch_device) for k, v in dummy_inputs.items() if isinstance(v, torch.Tensor)}
        inputs.update({k: v for k, v in dummy_inputs.items() if k not in inputs})
        out_0 = self.model_0(**inputs)[0]
        out_1 = model_1(**inputs)[0]
        self.assertTrue(torch.equal(out_0, out_1))

    def test_serialization_sharded(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model_0.save_pretrained(tmpdirname, max_shard_size="200MB")

            config = SD3Transformer2DModel.load_config(tmpdirname)
            self.assertTrue("quantization_config" in config)
            self.assertTrue("_pre_quantization_dtype" not in config)

            model_1 = SD3Transformer2DModel.from_pretrained(tmpdirname)

        # checking quantized linear module weight
        linear = get_some_linear_layer(model_1)
        self.assertTrue(linear.weight.__class__ == bnb.nn.Int8Params)
        self.assertTrue(hasattr(linear.weight, "SCB"))

        # comparing forward() outputs
        dummy_inputs = self.get_dummy_inputs()
        inputs = {k: v.to(torch_device) for k, v in dummy_inputs.items() if isinstance(v, torch.Tensor)}
        inputs.update({k: v for k, v in dummy_inputs.items() if k not in inputs})
        out_0 = self.model_0(**inputs)[0]
        out_1 = model_1(**inputs)[0]
        self.assertTrue(torch.equal(out_0, out_1))
