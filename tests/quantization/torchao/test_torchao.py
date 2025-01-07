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
    require_torchao_version_greater_or_equal,
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
    from torchao.quantization.linear_activation_quantized_tensor import LinearActivationQuantizedTensor
    from torchao.utils import get_model_size_in_bytes


@require_torch
@require_torch_gpu
@require_torchao_version_greater_or_equal("0.7.0")
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
@require_torchao_version_greater_or_equal("0.7.0")
class TorchAoTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

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

    def _test_quant_type(self, quantization_config: TorchAoConfig, expected_slice: List[float], model_id: str):
        components = self.get_dummy_components(quantization_config, model_id)
        pipe = FluxPipeline(**components)
        pipe.to(device=torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]
        output_slice = output[-1, -1, -3:, -3:].flatten()

        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_quantization(self):
        for model_id in ["hf-internal-testing/tiny-flux-pipe", "hf-internal-testing/tiny-flux-sharded"]:
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
                self._test_quant_type(quantization_config, expected_slice, model_id)

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

    def test_device_map(self):
        # Note: We were not checking if the weight tensor's were AffineQuantizedTensor's before. If we did
        # it would have errored out. Now, we do. So, device_map basically never worked with or without
        # sharded checkpoints. This will need to be supported in the future (TODO(aryan))
        """
        Test if the quantized model int4 weight-only is working properly with "auto" and custom device maps.
        The custom device map performs cpu/disk offloading as well. Also verifies that the device map is
        correctly set (in the `hf_device_map` attribute of the model).
        """
        custom_device_map_dict = {
            "time_text_embed": torch_device,
            "context_embedder": torch_device,
            "x_embedder": torch_device,
            "transformer_blocks.0": "cpu",
            "single_transformer_blocks.0": "disk",
            "norm_out": torch_device,
            "proj_out": "cpu",
        }
        device_maps = ["auto", custom_device_map_dict]

        # inputs = self.get_dummy_tensor_inputs(torch_device)
        # expected_slice = np.array([0.3457, -0.0366, 0.0105, -0.2275, -0.4941, 0.4395, -0.166, -0.6641, 0.4375])

        for device_map in device_maps:
            # device_map_to_compare = {"": 0} if device_map == "auto" else device_map

            # Test non-sharded model - should work
            with self.assertRaises(NotImplementedError):
                with tempfile.TemporaryDirectory() as offload_folder:
                    quantization_config = TorchAoConfig("int4_weight_only", group_size=64)
                    _ = FluxTransformer2DModel.from_pretrained(
                        "hf-internal-testing/tiny-flux-pipe",
                        subfolder="transformer",
                        quantization_config=quantization_config,
                        device_map=device_map,
                        torch_dtype=torch.bfloat16,
                        offload_folder=offload_folder,
                    )

                    # weight = quantized_model.transformer_blocks[0].ff.net[2].weight
                    # self.assertTrue(quantized_model.hf_device_map == device_map_to_compare)
                    # self.assertTrue(isinstance(weight, AffineQuantizedTensor))

                    # output = quantized_model(**inputs)[0]
                    # output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
                    # self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

            # Test sharded model - should not work
            with self.assertRaises(NotImplementedError):
                with tempfile.TemporaryDirectory() as offload_folder:
                    quantization_config = TorchAoConfig("int4_weight_only", group_size=64)
                    _ = FluxTransformer2DModel.from_pretrained(
                        "hf-internal-testing/tiny-flux-sharded",
                        subfolder="transformer",
                        quantization_config=quantization_config,
                        device_map=device_map,
                        torch_dtype=torch.bfloat16,
                        offload_folder=offload_folder,
                    )

                    # weight = quantized_model.transformer_blocks[0].ff.net[2].weight
                    # self.assertTrue(quantized_model.hf_device_map == device_map_to_compare)
                    # self.assertTrue(isinstance(weight, AffineQuantizedTensor))

                    # output = quantized_model(**inputs)[0]
                    # output_slice = output.flatten()[-9:].detach().float().cpu().numpy()

                    # self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_modules_to_not_convert(self):
        quantization_config = TorchAoConfig("int8_weight_only", modules_to_not_convert=["transformer_blocks.0"])
        quantized_model_with_not_convert = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        unquantized_layer = quantized_model_with_not_convert.transformer_blocks[0].ff.net[2]
        self.assertTrue(isinstance(unquantized_layer, torch.nn.Linear))
        self.assertFalse(isinstance(unquantized_layer.weight, AffineQuantizedTensor))
        self.assertEqual(unquantized_layer.weight.dtype, torch.bfloat16)

        quantized_layer = quantized_model_with_not_convert.proj_out
        self.assertTrue(isinstance(quantized_layer.weight, AffineQuantizedTensor))

        quantization_config = TorchAoConfig("int8_weight_only")
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        size_quantized_with_not_convert = get_model_size_in_bytes(quantized_model_with_not_convert)
        size_quantized = get_model_size_in_bytes(quantized_model)

        self.assertTrue(size_quantized < size_quantized_with_not_convert)

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
        for model_id in ["hf-internal-testing/tiny-flux-pipe", "hf-internal-testing/tiny-flux-sharded"]:
            quantization_config = TorchAoConfig("int8_weight_only")
            components = self.get_dummy_components(quantization_config, model_id=model_id)
            pipe = FluxPipeline(**components)
            pipe.to(device=torch_device)

            inputs = self.get_dummy_inputs(torch_device)
            normal_output = pipe(**inputs)[0].flatten()[-32:]

            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True, dynamic=False)
            inputs = self.get_dummy_inputs(torch_device)
            compile_output = pipe(**inputs)[0].flatten()[-32:]

            # Note: Seems to require higher tolerance
            self.assertTrue(np.allclose(normal_output, compile_output, atol=1e-2, rtol=1e-3))

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model and the class type of the linear layers of the converted models
        """
        for model_id in ["hf-internal-testing/tiny-flux-pipe", "hf-internal-testing/tiny-flux-sharded"]:
            transformer_int4wo = self.get_dummy_components(TorchAoConfig("int4wo"), model_id=model_id)["transformer"]
            transformer_int4wo_gs32 = self.get_dummy_components(
                TorchAoConfig("int4wo", group_size=32), model_id=model_id
            )["transformer"]
            transformer_int8wo = self.get_dummy_components(TorchAoConfig("int8wo"), model_id=model_id)["transformer"]
            transformer_bf16 = self.get_dummy_components(None, model_id=model_id)["transformer"]

            # Will not quantized all the layers by default due to the model weights shapes not being divisible by group_size=64
            for block in transformer_int4wo.transformer_blocks:
                self.assertTrue(isinstance(block.ff.net[2].weight, AffineQuantizedTensor))
                self.assertTrue(isinstance(block.ff_context.net[2].weight, AffineQuantizedTensor))

            # Will quantize all the linear layers except x_embedder
            for name, module in transformer_int4wo_gs32.named_modules():
                if isinstance(module, nn.Linear) and name not in ["x_embedder"]:
                    self.assertTrue(isinstance(module.weight, AffineQuantizedTensor))

            # Will quantize all the linear layers
            for module in transformer_int8wo.modules():
                if isinstance(module, nn.Linear):
                    self.assertTrue(isinstance(module.weight, AffineQuantizedTensor))

            total_int4wo = get_model_size_in_bytes(transformer_int4wo)
            total_int4wo_gs32 = get_model_size_in_bytes(transformer_int4wo_gs32)
            total_int8wo = get_model_size_in_bytes(transformer_int8wo)
            total_bf16 = get_model_size_in_bytes(transformer_bf16)

            # TODO: refactor to align with other quantization tests
            # Latter has smaller group size, so more groups -> more scales and zero points
            self.assertTrue(total_int4wo < total_int4wo_gs32)
            # int8 quantizes more layers compare to int4 with default group size
            self.assertTrue(total_int8wo < total_int4wo)
            # int4wo does not quantize too many layers because of default group size, but for the layers it does
            # there is additional overhead of scales and zero points
            self.assertTrue(total_bf16 < total_int4wo)

    def test_wrong_config(self):
        with self.assertRaises(ValueError):
            self.get_dummy_components(TorchAoConfig("int42"))


# Slices for these tests have been obtained on our aws-g6e-xlarge-plus runners
@require_torch
@require_torch_gpu
@require_torchao_version_greater_or_equal("0.7.0")
class TorchAoSerializationTest(unittest.TestCase):
    model_name = "hf-internal-testing/tiny-flux-pipe"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_model(self, quant_method, quant_method_kwargs, device=None):
        quantization_config = TorchAoConfig(quant_method, **quant_method_kwargs)
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

    def _test_original_model_expected_slice(self, quant_method, quant_method_kwargs, expected_slice):
        quantized_model = self.get_dummy_model(quant_method, quant_method_kwargs, torch_device)
        inputs = self.get_dummy_tensor_inputs(torch_device)
        output = quantized_model(**inputs)[0]
        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        weight = quantized_model.transformer_blocks[0].ff.net[2].weight
        self.assertTrue(isinstance(weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)))
        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def _check_serialization_expected_slice(self, quant_method, quant_method_kwargs, expected_slice, device):
        quantized_model = self.get_dummy_model(quant_method, quant_method_kwargs, device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir, safe_serialization=False)
            loaded_quantized_model = FluxTransformer2DModel.from_pretrained(
                tmp_dir, torch_dtype=torch.bfloat16, use_safetensors=False
            ).to(device=torch_device)

        inputs = self.get_dummy_tensor_inputs(torch_device)
        output = loaded_quantized_model(**inputs)[0]

        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(
            isinstance(
                loaded_quantized_model.proj_out.weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)
            )
        )
        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_int_a8w8_cuda(self):
        quant_method, quant_method_kwargs = "int8_dynamic_activation_int8_weight", {}
        expected_slice = np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])
        device = "cuda"
        self._test_original_model_expected_slice(quant_method, quant_method_kwargs, expected_slice)
        self._check_serialization_expected_slice(quant_method, quant_method_kwargs, expected_slice, device)

    def test_int_a16w8_cuda(self):
        quant_method, quant_method_kwargs = "int8_weight_only", {}
        expected_slice = np.array([0.3613, -0.127, -0.0223, -0.2539, -0.459, 0.4961, -0.1357, -0.6992, 0.4551])
        device = "cuda"
        self._test_original_model_expected_slice(quant_method, quant_method_kwargs, expected_slice)
        self._check_serialization_expected_slice(quant_method, quant_method_kwargs, expected_slice, device)

    def test_int_a8w8_cpu(self):
        quant_method, quant_method_kwargs = "int8_dynamic_activation_int8_weight", {}
        expected_slice = np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])
        device = "cpu"
        self._test_original_model_expected_slice(quant_method, quant_method_kwargs, expected_slice)
        self._check_serialization_expected_slice(quant_method, quant_method_kwargs, expected_slice, device)

    def test_int_a16w8_cpu(self):
        quant_method, quant_method_kwargs = "int8_weight_only", {}
        expected_slice = np.array([0.3613, -0.127, -0.0223, -0.2539, -0.459, 0.4961, -0.1357, -0.6992, 0.4551])
        device = "cpu"
        self._test_original_model_expected_slice(quant_method, quant_method_kwargs, expected_slice)
        self._check_serialization_expected_slice(quant_method, quant_method_kwargs, expected_slice, device)


# Slices for these tests have been obtained on our aws-g6e-xlarge-plus runners
@require_torch
@require_torch_gpu
@require_torchao_version_greater_or_equal("0.7.0")
@slow
@nightly
class SlowTorchAoTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

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
        self.assertTrue(isinstance(weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor)))

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
                ("fp5_e3m1", np.array([0.0527, 0.0762, 0.1309, 0.0449, 0.0645, 0.1328, 0.0566, 0.0723, 0.125, 0.0566, 0.0703, 0.1328, 0.0566, 0.0742, 0.1348, 0.0566, 0.3633, 0.7617, 0.5273, 0.4277, 0.7891, 0.5469, 0.4375, 0.8008, 0.5586, 0.4336, 0.7383, 0.5156, 0.3906, 0.6992, 0.5156, 0.4375])),
            ])
        # fmt: on

        for quantization_name, expected_slice in QUANTIZATION_TYPES_TO_TEST:
            quantization_config = TorchAoConfig(quant_type=quantization_name, modules_to_not_convert=["x_embedder"])
            self._test_quant_type(quantization_config, expected_slice)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_serialization_int8wo(self):
        quantization_config = TorchAoConfig("int8wo")
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.enable_model_cpu_offload()

        weight = pipe.transformer.x_embedder.weight
        self.assertTrue(isinstance(weight, AffineQuantizedTensor))

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0].flatten()[:128]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe.transformer.save_pretrained(tmp_dir, safe_serialization=False)
            pipe.remove_all_hooks()
            del pipe.transformer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            transformer = FluxTransformer2DModel.from_pretrained(
                tmp_dir, torch_dtype=torch.bfloat16, use_safetensors=False
            )
            pipe.transformer = transformer
            pipe.enable_model_cpu_offload()

        weight = transformer.x_embedder.weight
        self.assertTrue(isinstance(weight, AffineQuantizedTensor))

        loaded_output = pipe(**inputs)[0].flatten()[:128]
        # Seems to require higher tolerance depending on which machine it is being run.
        # A difference of 0.06 in normalized pixel space (-1 to 1), corresponds to a difference of
        # 0.06 / 2 * 255 = 7.65 in pixel space (0 to 255). On our CI runners, the difference is about 0.04,
        # on DGX it is 0.06, and on audace it is 0.037. So, we are using a tolerance of 0.06 here.
        self.assertTrue(np.allclose(output, loaded_output, atol=0.06))

    def test_memory_footprint_int4wo(self):
        # The original checkpoints are in bf16 and about 24 GB
        expected_memory_in_gb = 6.0
        quantization_config = TorchAoConfig("int4wo")
        cache_dir = None
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        int4wo_memory_in_gb = get_model_size_in_bytes(transformer) / 1024**3
        self.assertTrue(int4wo_memory_in_gb < expected_memory_in_gb)

    def test_memory_footprint_int8wo(self):
        # The original checkpoints are in bf16 and about 24 GB
        expected_memory_in_gb = 12.0
        quantization_config = TorchAoConfig("int8wo")
        cache_dir = None
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        int8wo_memory_in_gb = get_model_size_in_bytes(transformer) / 1024**3
        self.assertTrue(int8wo_memory_in_gb < expected_memory_in_gb)


@require_torch
@require_torch_gpu
@require_torchao_version_greater_or_equal("0.7.0")
@slow
@nightly
class SlowTorchAoPreserializedModelTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

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

    def test_transformer_int8wo(self):
        # fmt: off
        expected_slice = np.array([0.0566, 0.0781, 0.1426, 0.0488, 0.0684, 0.1504, 0.0625, 0.0781, 0.1445, 0.0625, 0.0781, 0.1562, 0.0547, 0.0723, 0.1484, 0.0566, 0.5703, 0.8867, 0.7266, 0.5742, 0.875, 0.7148, 0.5586, 0.875, 0.7148, 0.5547, 0.8633, 0.7109, 0.5469, 0.8398, 0.6992, 0.5703])
        # fmt: on

        # This is just for convenience, so that we can modify it at one place for custom environments and locally testing
        cache_dir = None
        transformer = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/FLUX.1-Dev-TorchAO-int8wo-transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
            cache_dir=cache_dir,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        pipe.enable_model_cpu_offload()

        # Verify that all linear layer weights are quantized
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, nn.Linear):
                self.assertTrue(isinstance(module.weight, AffineQuantizedTensor))

        # Verify outputs match expected slice
        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0].flatten()
        output_slice = np.concatenate((output[:16], output[-16:]))
        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))
