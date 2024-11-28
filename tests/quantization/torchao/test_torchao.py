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
from diffusers.utils.testing_utils import (
    is_torch_available,
    is_torchao_available,
    require_torch,
    require_torch_gpu,
    require_torchao_version_greater,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

if is_torchao_available():
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.dtypes.affine_quantized_tensor import TensorCoreTiledLayoutType


def check_forward(test_module, model, batch_size=1, context_size=1024):
    # Test forward pass
    with torch.no_grad():
        out = model(torch.zeros([batch_size, context_size], device=model.device, dtype=torch.int32)).logits
    test_module.assertEqual(out.shape[0], batch_size)
    test_module.assertEqual(out.shape[1], context_size)


@require_torch
@require_torch_gpu
@require_torchao_version_greater("0.6.0")
@slow
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
@slow
class TorchAoTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

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

        qlayer = quantized_model.transformer_blocks[0].attn.to_q
        weight = qlayer.weight
        self.assertTrue(isinstance(weight, AffineQuantizedTensor))
        self.assertEqual(weight.quant_min, 0)
        self.assertEqual(weight.quant_max, 15)
        self.assertTrue(isinstance(weight.layout_type, TensorCoreTiledLayoutType))

    def test_int4wo_offload(self):
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

        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(
            torch_device, dtype=torch.bfloat16
        )
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(
            torch_device, dtype=torch.bfloat16
        )
        pooled_prompt_embeds = torch.randn((batch_size, embedding_dim)).to(torch_device, dtype=torch.bfloat16)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(torch_device, dtype=torch.bfloat16)
        image_ids = torch.randn((height * width, num_image_channels)).to(torch_device, dtype=torch.bfloat16)
        timestep = torch.tensor([1.0]).to(torch_device, dtype=torch.bfloat16).expand(batch_size)

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

            output = quantized_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                img_ids=image_ids,
                txt_ids=text_ids,
                pooled_projections=pooled_prompt_embeds,
                timestep=timestep,
            )

            output_slice = output.flatten()[-9:].detach().cpu().numpy()
            # TODO(aryan): get slice from CI
            expected_slice = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))


# @require_torch_gpu
# @require_torchao
# class TorchAoSerializationTest(unittest.TestCase):
#     input_text = "What are we having for dinner?"
#     max_new_tokens = 10
#     ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n- 1. What is the temperature outside"
#     # TODO: investigate why we don't have the same output as the original model for this test
#     SERIALIZED_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
#     model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     quant_scheme, quant_scheme_kwargs = "int4_weight_only", {"group_size": 32}
#     device = "cuda:0"

#     # called only once for all test in this class
#     @classmethod
#     def setUpClass(cls):
#         cls.quant_config = TorchAoConfig(cls.quant_scheme, **cls.quant_scheme_kwargs)
#         cls.quantized_model = AutoModelForCausalLM.from_pretrained(
#             cls.model_name,
#             torch_dtype=torch.bfloat16,
#             device_map=cls.device,
#             quantization_config=cls.quant_config,
#         )
#         cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

#     def tearDown(self):
#         gc.collect()
#         torch.cuda.empty_cache()
#         gc.collect()

#     def test_original_model_expected_output(self):
#         input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device)
#         output = self.quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)

#         self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), self.ORIGINAL_EXPECTED_OUTPUT)

#     def check_serialization_expected_output(self, device, expected_output):
#         """
#         Test if we can serialize and load/infer the model again on the same device
#         """
#         with tempfile.TemporaryDirectory() as tmpdirname:
#             self.quantized_model.save_pretrained(tmpdirname, safe_serialization=False)
#             loaded_quantized_model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
#             )
#             input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(self.device)

#             output = loaded_quantized_model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
#             self.assertEqual(self.tokenizer.decode(output[0], skip_special_tokens=True), expected_output)

#     def test_serialization_expected_output(self):
#         self.check_serialization_expected_output(self.device, self.SERIALIZED_EXPECTED_OUTPUT)


# class TorchAoSerializationW8A8Test(TorchAoSerializationTest):
#     quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
#     ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
#     SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
#     device = "cuda:0"


# class TorchAoSerializationW8Test(TorchAoSerializationTest):
#     quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
#     ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
#     SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
#     device = "cuda:0"


# class TorchAoSerializationW8A8CPUTest(TorchAoSerializationTest):
#     quant_scheme, quant_scheme_kwargs = "int8_dynamic_activation_int8_weight", {}
#     ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
#     SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
#     device = "cpu"

#     def test_serialization_expected_output_cuda(self):
#         """
#         Test if we can serialize on device (cpu) and load/infer the model on cuda
#         """
#         new_device = "cuda:0"
#         self.check_serialization_expected_output(new_device, self.SERIALIZED_EXPECTED_OUTPUT)


# class TorchAoSerializationW8CPUTest(TorchAoSerializationTest):
#     quant_scheme, quant_scheme_kwargs = "int8_weight_only", {}
#     ORIGINAL_EXPECTED_OUTPUT = "What are we having for dinner?\n\nJessica: (smiling)"
#     SERIALIZED_EXPECTED_OUTPUT = ORIGINAL_EXPECTED_OUTPUT
#     device = "cpu"

#     def test_serialization_expected_output_cuda(self):
#         """
#         Test if we can serialize on device (cpu) and load/infer the model on cuda
#         """
#         new_device = "cuda:0"
#         self.check_serialization_expected_output(new_device, self.SERIALIZED_EXPECTED_OUTPUT)


if __name__ == "__main__":
    unittest.main()
