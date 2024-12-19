# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import gc
import tempfile
import unittest
from typing import List

import numpy as np
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    TorchAoConfig,
)
from diffusers.models.attention_processor import Attention
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    is_torch_available,
    is_torchao_available,
    nightly,
    require_torch,
    require_torch_gpu,
    require_torchao_version_greater,
    slow,
    torch_device,
)


enable_full_determinism()


if is_torch_available():
    import torch
    import torch.nn as nn

    class LoRALayer(nn.Module):
        """Wraps a linear layer with LoRA-like adapter - Used for testing purposes only

        Taken from
        https://github.com/huggingface/transformers/blob/566302686a71de14125717dea9a6a45b24d42b37/tests/quantization/bnb/test_4bit.py#L62C5-L78C77
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


if is_torchao_available():
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.dtypes.affine_quantized_tensor import TensorCoreTiledLayoutType
    from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor


@require_torch
@require_torch_gpu
@require_torchao_version_greater("0.6.0")
class TorchAoConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = TorchAoConfig("int4_weight_only")
        torchao_orig_config = quantization_config.to_dict()

        for key in torchao_orig_config:
            self.assertEqual(getattr(quantization_config, key), torchao_orig_config[key])

    def test_post_init_check(self):
        """
        Test kwargs validations in TorchAoConfig
        """
        _ = TorchAoConfig("int4_weight_only")
        with self.assertRaisesRegex(ValueError, "is not supported yet"):
            _ = TorchAoConfig("uint8")

        with self.assertRaisesRegex(ValueError, "does not support the following keyword arguments"):
            _ = TorchAoConfig("int4_weight_only", group_size1=32)

    def test_repr(self):
        """
        Check that there is no error in the repr
        """
        quantization_config = TorchAoConfig("int4_weight_only", modules_to_not_convert=["conv"], group_size=8)
        expected_repr = """TorchAoConfig {
            "modules_to_not_convert": [
                "conv"
            ],
            "quant_method": "torchao",
            "quant_type": "int4_weight_only",
            "quant_type_kwargs": {
                "group_size": 8
            }
        }""".replace(" ", "").replace("\n", "")
        quantization_repr = repr(quantization_config).replace(" ", "").replace("\n", "")
        self.assertEqual(quantization_repr, expected_repr)


# Slices for these tests have been obtained on our aws-g6e-xlarge-plus runners
@require_torch
@require_torch_gpu
@require_torchao_version_greater("0.6.0")
class TorchAoTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_components(self, quantization_config: TorchAoConfig):
        model_id = "hf-internal-testing/tiny-flux-pipe"
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
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

    def get_dummy_tensor_inputs(self, device=None, seed: int = 0):
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

    def _test_quant_type(self, quantization_config: TorchAoConfig, expected_slice: List[float]):
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.to(device=torch_device, dtype=torch.bfloat16)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]
        output_slice = output[-1, -1, -3:, -3:].flatten()

        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_quantization(self):
        # fmt: off
        QUANTIZATION_TYPES_TO_TEST = [
            ("int4wo", np.array([0.4648, 0.5234, 0.5547, 0.4219, 0.4414, 0.6445, 0.4336, 0.4531, 0.5625])),
            ("int4dq", np.array([0.4688, 0.5195, 0.5547, 0.418, 0.4414, 0.6406, 0.4336, 0.4531, 0.5625])),
            ("int8wo", np.array([0.4648, 0.5195, 0.5547, 0.4199, 0.4414, 0.6445, 0.4316, 0.4531, 0.5625])),
            ("int8dq", np.array([0.4648, 0.5195, 0.5547, 0.4199, 0.4414, 0.6445, 0.4316, 0.4531, 0.5625])),
            ("uint4wo", np.array([0.4609, 0.5234, 0.5508, 0.4199, 0.4336, 0.6406, 0.4316, 0.4531, 0.5625])),
            ("uint7wo", np.array([0.4648, 0.5195, 0.5547, 0.4219, 0.4414, 0.6445, 0.4316, 0.4531, 0.5625])),
        ]

        if TorchAoConfig._is_cuda_capability_atleast_8_9():
            QUANTIZATION_TYPES_TO_TEST.extend([
                ("float8wo_e5m2", np.array([0.4590, 0.5273, 0.5547, 0.4219, 0.4375, 0.6406, 0.4316, 0.4512, 0.5625])),
                ("float8wo_e4m3", np.array([0.4648, 0.5234, 0.5547, 0.4219, 0.4414, 0.6406, 0.4316, 0.4531, 0.5625])),
                # =====
                # The following lead to an internal torch error:
                #    RuntimeError: mat2 shape (32x4 must be divisible by 16
                # Skip these for now; TODO(aryan): investigate later
                # ("float8dq_e4m3", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                # ("float8dq_e4m3_tensor", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                # =====
                # Cutlass fails to initialize for below
                # ("float8dq_e4m3_row", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                # =====
                ("fp4", np.array([0.4668, 0.5195, 0.5547, 0.4199, 0.4434, 0.6445, 0.4316, 0.4531, 0.5625])),
                ("fp6", np.array([0.4668, 0.5195, 0.5547, 0.4199, 0.4434, 0.6445, 0.4316, 0.4531, 0.5625])),
            ])
        # fmt: on

        for quantization_name, expected_slice in QUANTIZATION_TYPES_TO_TEST:
            quant_kwargs = {}
            if quantization_name in ["uint4wo", "uint7wo"]:
                # The dummy flux model that we use has smaller dimensions. This imposes some restrictions on group_size here
                quant_kwargs.update({"group_size": 16})
            quantization_config = TorchAoConfig(
                quant_type=quantization_name, modules_to_not_convert=["x_embedder"], **quant_kwargs
            )
            self._test_quant_type(quantization_config, expected_slice)

    def test_int4wo_quant_bfloat16_conversion(self):
        """
        Tests whether the dtype of model will be modified to bfloat16 for int4 weight-only quantization.
        """
        quantization_config = TorchAoConfig("int4_weight_only", group_size=64)
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        weight = quantized_model.transformer_blocks[0].ff.net[2].weight
        self.assertTrue(isinstance(weight, AffineQuantizedTensor))
        self.assertEqual(weight.quant_min, 0)
        self.assertEqual(weight.quant_max, 15)
        self.assertTrue(isinstance(weight.layout_type, TensorCoreTiledLayoutType))

    def test_offload(self):
        """
        Test if the quantized model int4 weight-only is working properly with cpu/disk offload. Also verifies
        that the device map is correctly set (in the `hf_device_map` attribute of the model).
        """

        device_map_offload = {
            "time_text_embed": torch_device,
            "context_embedder": torch_device,
            "x_embedder": torch_device,
            "transformer_blocks.0": "cpu",
            "single_transformer_blocks.0": "disk",
            "norm_out": torch_device,
            "proj_out": "cpu",
        }

        inputs = self.get_dummy_tensor_inputs(torch_device)

        with tempfile.TemporaryDirectory() as offload_folder:
            quantization_config = TorchAoConfig("int4_weight_only", group_size=64)
            quantized_model = FluxTransformer2DModel.from_pretrained(
                "hf-internal-testing/tiny-flux-pipe",
                subfolder="transformer",
                quantization_config=quantization_config,
                device_map=device_map_offload,
                torch_dtype=torch.bfloat16,
                offload_folder=offload_folder,
            )

            self.assertTrue(quantized_model.hf_device_map == device_map_offload)

            output = quantized_model(**inputs)[0]
            output_slice = output.flatten()[-9:].detach().float().cpu().numpy()

            expected_slice = np.array([0.3457, -0.0366, 0.0105, -0.2275, -0.4941, 0.4395, -0.166, -0.6641, 0.4375])
            self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_modules_to_not_convert(self):
        quantization_config = TorchAoConfig("int8_weight_only", modules_to_not_convert=["transformer_blocks.0"])
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        unquantized_layer = quantized_model.transformer_blocks[0].ff.net[2]
        self.assertTrue(isinstance(unquantized_layer, torch.nn.Linear))
        self.assertFalse(isinstance(unquantized_layer.weight, AffineQuantizedTensor))
        self.assertEqual(unquantized_layer.weight.dtype, torch.bfloat16)

        quantized_layer = quantized_model.proj_out
        self.assertTrue(isinstance(quantized_layer.weight, AffineQuantizedTensor))
        self.assertEqual(quantized_layer.weight.layout_tensor.data.dtype, torch.int8)

    def test_training(self):
        quantization_config = TorchAoConfig("int8_weight_only")
        quantized_model = FluxTransformer2DModel.from_pretrained(
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
            inputs = self.get_dummy_tensor_inputs(torch_device)
            output = quantized_model(**inputs)[0]
            output.norm().backward()

        for module in quantized_model.modules():
            if isinstance(module, LoRALayer):
                self.assertTrue(module.adapter[1].weight.grad is not None)
                self.assertTrue(module.adapter[1].weight.grad.norm().item() > 0)

    @nightly
    def test_torch_compile(self):
        r"""Test that verifies if torch.compile works with torchao quantization."""
        quantization_config = TorchAoConfig("int8_weight_only")
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.to(device=torch_device, dtype=torch.bfloat16)

        inputs = self.get_dummy_inputs(torch_device)
        normal_output = pipe(**inputs)[0].flatten()[-32:]

        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True, dynamic=False)
        inputs = self.get_dummy_inputs(torch_device)
        compile_output = pipe(**inputs)[0].flatten()[-32:]

        # Note: Seems to require higher tolerance
        self.assertTrue(np.allclose(normal_output, compile_output, atol=1e-2, rtol=1e-3))

    @staticmethod
    def _get_memory_footprint(module):
        quantized_param_memory = 0.0
        unquantized_param_memory = 0.0

        for param in module.parameters():
            if param.__class__.__name__ == "AffineQuantizedTensor":
                data, scale, zero_point = param.layout_tensor.get_plain()
                quantized_param_memory += data.numel() + data.element_size()
                quantized_param_memory += scale.numel() + scale.element_size()
                quantized_param_memory += zero_point.numel() + zero_point.element_size()
            else:
                unquantized_param_memory += param.data.numel() * param.data.element_size()

        total_memory = quantized_param_memory + unquantized_param_memory
        return total_memory, quantized_param_memory, unquantized_param_memory

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        transformer_int4wo = self.get_dummy_components(TorchAoConfig("int4wo"))["transformer"]
        transformer_int4wo_gs32 = self.get_dummy_components(TorchAoConfig("int4wo", group_size=32))["transformer"]
        transformer_int8wo = self.get_dummy_components(TorchAoConfig("int8wo"))["transformer"]
        transformer_bf16 = self.get_dummy_components(None)["transformer"]

        total_int4wo, quantized_int4wo, unquantized_int4wo = self._get_memory_footprint(transformer_int4wo)
        total_int4wo_gs32, quantized_int4wo_gs32, unquantized_int4wo_gs32 = self._get_memory_footprint(
            transformer_int4wo_gs32
        )
        total_int8wo, quantized_int8wo, unquantized_int8wo = self._get_memory_footprint(transformer_int8wo)
        total_bf16, quantized_bf16, unquantized_bf16 = self._get_memory_footprint(transformer_bf16)

        self.assertTrue(quantized_bf16 == 0 and total_bf16 == unquantized_bf16)
        # int4wo_gs32 has smaller group size, so more groups -> more scales and zero points
        self.assertTrue(total_int8wo < total_bf16 < total_int4wo_gs32)
        # int4 with default group size quantized very few linear layers compared to a smaller group size of 32
        self.assertTrue(quantized_int4wo < quantized_int4wo_gs32 and unquantized_int4wo > unquantized_int4wo_gs32)
        # int8 quantizes more layers compare to int4 with default group size
        self.assertTrue(quantized_int8wo < quantized_int4wo)

    def test_wrong_config(self):
        with self.assertRaises(ValueError):
            self.get_dummy_components(TorchAoConfig("int42"))


# This class is not to be run as a test by itself. See the tests that follow this class
@require_torch
@require_torch_gpu
@require_torchao_version_greater("0.6.0")
class TorchAoSerializationTest(unittest.TestCase):
    model_name = "hf-internal-testing/tiny-flux-pipe"
    quant_method, quant_method_kwargs = None, None
    device = "cuda"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_model(self, device=None):
        quantization_config = TorchAoConfig(self.quant_method, **self.quant_method_kwargs)
        quantized_model = FluxTransformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        return quantized_model.to(device)

    def get_dummy_tensor_inputs(self, device=None, seed: int = 0):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

        torch.manual_seed(seed)
        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(device, dtype=torch.bfloat16)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(
            device, dtype=torch.bfloat16
        )
        pooled_prompt_embeds = torch.randn((batch_size, embedding_dim)).to(device, dtype=torch.bfloat16)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(device, dtype=torch.bfloat16)
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

    def test_original_model_expected_slice(self):
        quantized_model = self.get_dummy_model(torch_device)
        inputs = self.get_dummy_tensor_inputs(torch_device)
        output = quantized_model(**inputs)[0]
        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(np.allclose(output_slice, self.expected_slice, atol=1e-3, rtol=1e-3))

    def check_serialization_expected_slice(self, expected_slice):
        quantized_model = self.get_dummy_model(self.device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir, safe_serialization=False)
            loaded_quantized_model = FluxTransformer2DModel.from_pretrained(
                tmp_dir, torch_dtype=torch.bfloat16, device_map=torch_device, use_safetensors=False
            )

        inputs = self.get_dummy_tensor_inputs(torch_device)
        output = loaded_quantized_model(**inputs)[0]

        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(
            isinstance(
                loaded_quantized_model.proj_out.weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)
            )
        )
        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_serialization_expected_slice(self):
        self.check_serialization_expected_slice(self.serialized_expected_slice)


class TorchAoSerializationINTA8W8Test(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_dynamic_activation_int8_weight", {}
    expected_slice = np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])
    serialized_expected_slice = expected_slice
    device = "cuda"


class TorchAoSerializationINTA16W8Test(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_weight_only", {}
    expected_slice = np.array([0.3613, -0.127, -0.0223, -0.2539, -0.459, 0.4961, -0.1357, -0.6992, 0.4551])
    serialized_expected_slice = expected_slice
    device = "cuda"


class TorchAoSerializationINTA8W8CPUTest(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_dynamic_activation_int8_weight", {}
    expected_slice = np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])
    serialized_expected_slice = expected_slice
    device = "cpu"


class TorchAoSerializationINTA16W8CPUTest(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_weight_only", {}
    expected_slice = np.array([0.3613, -0.127, -0.0223, -0.2539, -0.459, 0.4961, -0.1357, -0.6992, 0.4551])
    serialized_expected_slice = expected_slice
    device = "cpu"


# Slices for these tests have been obtained on our aws-g6e-xlarge-plus runners
@require_torch
@require_torch_gpu
@require_torchao_version_greater("0.6.0")
@slow
@nightly
class SlowTorchAoTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_components(self, quantization_config: TorchAoConfig):
        model_id = "black-forest-labs/FLUX.1-dev"
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        text_encoder_2 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_2")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        tokenizer_2 = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
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
        pipe = FluxPipeline(**components).to(dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0].flatten()
        output_slice = np.concatenate((output[:16], output[-16:]))

        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_quantization(self):
        # fmt: off
        QUANTIZATION_TYPES_TO_TEST = [
            ("int8wo", np.array([0.0505, 0.0742, 0.1367, 0.0429, 0.0585, 0.1386, 0.0585, 0.0703, 0.1367, 0.0566, 0.0703, 0.1464, 0.0546, 0.0703, 0.1425, 0.0546, 0.3535, 0.7578, 0.5000, 0.4062, 0.7656, 0.5117, 0.4121, 0.7656, 0.5117, 0.3984, 0.7578, 0.5234, 0.4023, 0.7382, 0.5390, 0.4570])),
            ("int8dq", np.array([0.0546, 0.0761, 0.1386, 0.0488, 0.0644, 0.1425, 0.0605, 0.0742, 0.1406, 0.0625, 0.0722, 0.1523, 0.0625, 0.0742, 0.1503, 0.0605, 0.3886, 0.7968, 0.5507, 0.4492, 0.7890, 0.5351, 0.4316, 0.8007, 0.5390, 0.4179, 0.8281, 0.5820, 0.4531, 0.7812, 0.5703, 0.4921])),
        ]

        if TorchAoConfig._is_cuda_capability_atleast_8_9():
            QUANTIZATION_TYPES_TO_TEST.extend([
                ("float8wo_e4m3", np.array([0.0546, 0.0722, 0.1328, 0.0468, 0.0585, 0.1367, 0.0605, 0.0703, 0.1328, 0.0625, 0.0703, 0.1445, 0.0585, 0.0703, 0.1406, 0.0605, 0.3496, 0.7109, 0.4843, 0.4042, 0.7226, 0.5000, 0.4160, 0.7031, 0.4824, 0.3886, 0.6757, 0.4667, 0.3710, 0.6679, 0.4902, 0.4238])),
                ("fp5_e3m1", np.array([0.0527, 0.0742, 0.1289, 0.0449, 0.0625, 0.1308, 0.0585, 0.0742, 0.1269, 0.0585, 0.0722, 0.1328, 0.0566, 0.0742, 0.1347, 0.0585, 0.3691, 0.7578, 0.5429, 0.4355, 0.7695, 0.5546, 0.4414, 0.7578, 0.5468, 0.4179, 0.7265, 0.5273, 0.3945, 0.6992, 0.5234, 0.4316])),
            ])
        # fmt: on

        for quantization_name, expected_slice in QUANTIZATION_TYPES_TO_TEST:
            quantization_config = TorchAoConfig(quant_type=quantization_name, modules_to_not_convert=["x_embedder"])
            self._test_quant_type(quantization_config, expected_slice)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
