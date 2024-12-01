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
    is_torch_available,
    is_torchao_available,
    require_torch,
    require_torch_gpu,
    require_torchao_version_greater,
    torch_device,
)


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
        repr(quantization_config)


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

    def get_dummy_tensor_inputs(self, device=None):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

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

    def _test_quant_type(self, quantization_config: TorchAoConfig, expected_slice: List[float]):
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)
        pipe.to(device=torch_device, dtype=torch.bfloat16)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]
        output_slice = output[-1, -1, -3:, -3:].flatten()

        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_quantization(self):
        # TODO(aryan): update these values from our CI
        QUANTIZATION_TYPES_TO_TEST = [
            ("int4wo", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("int4dq", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("int8wo", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("int8dq", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("uint4wo", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("int_a8w8", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
            ("uint_a16w7", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ]

        if TorchAoConfig._is_cuda_capability_atleast_8_9():
            QUANTIZATION_TYPES_TO_TEST.extend(
                [
                    ("float8wo_e5m2", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ("float8wo_e4m3", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ("float8dq_e4m3", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ("float8dq_e4m3_tensor", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ("float8dq_e4m3_row", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ("fp4wo", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    ("fp6", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                ]
            )

        for quantization_name, expected_slice in QUANTIZATION_TYPES_TO_TEST:
            quantization_config = TorchAoConfig(quant_type=quantization_name)
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
        Test if the quantized model int4 weight-only is working properly with cpu/disk offload.
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

            output = quantized_model(**inputs)[0]

            output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
            # TODO(aryan): get slice from CI
            expected_slice = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
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

    def get_dummy_tensor_inputs(self, device=None):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

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
        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_serialization_expected_slice(self):
        self.check_serialization_expected_slice(self.serialized_expected_slice)


class TorchAoSerializationINTA8W8Test(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_dynamic_activation_int8_weight", {}
    expected_slice = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    serialized_expected_slice = expected_slice
    device = "cuda"


class TorchAoSerializationINTA16W8Test(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_weight_only", {}
    expected_slice = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    serialized_expected_slice = expected_slice
    device = "cuda"


class TorchAoSerializationINTA8W8CPUTest(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_dynamic_activation_int8_weight", {}
    expected_slice = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    serialized_expected_slice = expected_slice
    device = "cpu"


class TorchAoSerializationINTA16W8CPUTest(TorchAoSerializationTest):
    quant_method, quant_method_kwargs = "int8_weight_only", {}
    expected_slice = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    serialized_expected_slice = expected_slice
    device = "cpu"
