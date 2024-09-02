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

from diffusers import BitsAndBytesConfig, DiffusionPipeline, SD3Transformer2DModel
from diffusers.utils.testing_utils import (
    is_bitsandbytes_available,
    is_torch_available,
    load_pt,
    print_tensor_test,
    require_accelerate,
    require_bitsandbytes,
    require_torch_gpu,
    slow,
    torch_device,
)


def get_some_linear_layer(model):
    if model.__class__.__name__ == "SD3Transformer2DModel":
        return model.transformer_blocks[0].attn.to_q
    else:
        return NotImplementedError("Don't know what layer to retrieve here.")


if is_torch_available():
    import torch


if is_bitsandbytes_available():
    import bitsandbytes as bnb


@require_bitsandbytes
@require_accelerate
@require_torch_gpu
class Base4bitTests(unittest.TestCase):
    # We need to test on relatively large models (aka >1b parameters otherwise the quantiztion may not work as expected)
    # Therefore here we use only bloom-1b3 to test our module
    model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

    prompt = "a beautiful sunset amidst the mountains."
    num_inference_steps = 10
    seed = 0

    prompt_embeds = load_pt(
        "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/prompt_embeds.pt"
    ).to(torch_device)
    pooled_prompt_embeds = load_pt(
        "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/pooled_prompt_embeds.pt"
    ).to(torch_device)
    latent_model_input = load_pt(
        "https://huggingface.co/datasets/hf-internal-testing/bnb-diffusers-testing-artifacts/resolve/main/latent_model_input.pt"
    ).to(torch_device)
    input_dict_for_transformer = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "latent_model_input": latent_model_input,
        "timestep": torch.Tensor([1.0]),
        "return_dict": False,
    }


class BnB4BitBasicTests(Base4bitTests):
    def setUp(self):
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
            self.model_name, subfolder="transformer", quantization_config=nf4_config
        )

    def tearDown(self):
        del self.model_fp16
        del self.model_4bit

        gc.collect()
        torch.cuda.empty_cache()

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

        self.assertTrue(hasattr(config, "quantization_config"))

        _ = config.to_dict()
        _ = config.to_diff_dict()

        _ = config.to_json_string()

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        from bitsandbytes.nn import Params4bit

        mem_fp16 = self.model_fp16.get_memory_footprint()
        mem_4bit = self.model_4bit.get_memory_footprint()

        self.assertAlmostEqual(mem_fp16 / mem_4bit, self.EXPECTED_RELATIVE_DIFFERENCE)
        linear = get_some_linear_layer(self.model_4bit)
        self.assertTrue(linear.weight.__class__ == Params4bit)

    def test_original_dtype(self):
        r"""
        A simple test to check if the model succesfully stores the original dtype
        """
        self.assertTrue(hasattr(self.model_4bit.config, "_pre_quantization_dtype"))
        self.assertFalse(hasattr(self.model_fp16.config, "_pre_quantization_dtype"))
        self.assertTrue(self.model_4bit.config._pre_quantization_dtype == torch.float16)

    def test_linear_are_4bit(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        self.model_fp16.get_memory_footprint()
        self.model_4bit.get_memory_footprint()

        for name, module in self.model_4bit.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name not in self.model_fp16._keep_in_fp32_modules:
                    # 4-bit parameters are packed in uint8 variables
                    self.assertTrue(module.weight.dtype == torch.uint8)

    def test_device_and_dtype_assignment(self):
        r"""
        Test whether trying to cast (or assigning a device to) a model after converting it in 8-bit will throw an error.
        Checks also if other models are casted correctly.
        """
        with self.assertRaises(ValueError):
            # Tries with `str`
            self.model_4bit.to("cpu")

        with self.assertRaises(ValueError):
            # Tries with a `dtype``
            self.model_4bit.to(torch.float16)

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_4bit.to(torch.device("cuda:0"))

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_4bit.float()

        with self.assertRaises(ValueError):
            # Tries with a `device`
            self.model_4bit.half()

        # Test if we did not break anything

        self.model_fp16 = self.model_fp16.to(torch.float32)
        model_inputs = {k: v.to(torch.float32) for k, v in self.input_dict_for_transformer.items()}
        with torch.no_grad():
            _ = self.model_fp16(**model_inputs)

        # Check this does not throw an error
        _ = self.model_fp16.to("cpu")

        # Check this does not throw an error
        _ = self.model_fp16.half()

        # Check this does not throw an error
        _ = self.model_fp16.float()

    def test_bnb_4bit_wrong_config(self):
        r"""
        Test whether creating a bnb config with unsupported values leads to errors.
        """
        with self.assertRaises(ValueError):
            _ = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_storage="add")


@slow
class SlowBnb4BitTests(Base4bitTests):
    def setUp(self) -> None:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_4bit = SD3Transformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", quantization_config=nf4_config
        )
        self.pipeline_4bit = DiffusionPipeline.from_pretrained(
            self.model_name, transformer=model_4bit, torch_dtype=torch.float16
        )
        self.pipeline_4bit.enable_model_cpu_offload()

    def tearDown(self):
        del self.pipeline_4bit

        gc.collect()
        torch.cuda.empty_cache()

    def test_quality(self):
        output = self.pipeline_4bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images
        print_tensor_test(output, limit_to_slices=True)

        assert output is None

    def test_generate_quality_dequantize(self):
        r"""
        Test that loading the model and unquantize it produce correct results
        """
        self.pipeline_4bit.dequantize()
        output = self.pipeline_4bit(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            generator=torch.manual_seed(self.seed),
            output_type="np",
        ).images
        print_tensor_test(output, limit_to_slices=True)

        assert output is None


@slow
class BaseBnb4BitSerializationTests(Base4bitTests):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

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
        model_0 = SD3Transformer2DModel.from_pretrained(self.model_name, quantization_config=self.quantization_config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_0.save_pretrained(tmpdirname, safe_serialization=safe_serialization)

            config = SD3Transformer2DModel.load_config(tmpdirname)
            self.assertTrue("quantization_config" in config)

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
        with torch.no_grad():
            out_0 = model_0(**self.input_dict_for_transformer)[0]
            out_1 = model_1(**self.input_dict_for_transformer)[0]
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
