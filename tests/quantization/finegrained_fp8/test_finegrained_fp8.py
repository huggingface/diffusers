# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
    nightly,
    numpy_cosine_similarity_distance,
    require_torch,
    require_torch_gpu,
    require_torchao_version_greater_or_equal,
    slow,
    torch_device,
)

from diffusers.quantizers.quantization_config import FinegrainedFP8Config
from diffusers.quantizers.finegrained_fp8.utils import FP8Linear


enable_full_determinism()


if is_torch_available():
    import torch
    import torch.nn as nn

    from ..utils import LoRALayer, get_memory_consumption_stat


@require_torch
@require_torch_gpu
class FinegrainedFP8ConfigTest(unittest.TestCase):
    def test_to_dict(self):
        """
        Makes sure the config format is properly set
        """
        quantization_config = FinegrainedFP8Config()
        finegrained_fp8_orig_config = quantization_config.to_dict()

        for key in finegrained_fp8_orig_config:
            self.assertEqual(getattr(quantization_config, key), finegrained_fp8_orig_config[key])

    def test_post_init_check(self):
        """
        Test kwargs validations in FinegrainedFP8Config
        """
        _ = FinegrainedFP8Config()
        with self.assertRaisesRegex(ValueError, "weight_block_size must be a tuple of two integers"):
            _ = FinegrainedFP8Config(weight_block_size=(1, 32, 32))

        with self.assertRaisesRegex(ValueError, "weight_block_size must be a tuple of two positive integers"):
            _ = FinegrainedFP8Config(weight_block_size=(0, 32))


@require_torch
@require_torch_gpu
class FinegrainedFP8Test(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_components(
        self, quantization_config: FinegrainedFP8Config, model_id: str = "hf-internal-testing/tiny-flux-pipe"
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
    def get_model_size_in_bytes(self, model, ignore_embeddings=False):
        """
        Returns the model size in bytes. The option to ignore embeddings
        is useful for models with disproportionately large embeddings compared
        to other model parameters that get quantized/sparsified.
        """
        model_size = 0
        for name, child in model.named_children():
            if not (isinstance(child, torch.nn.Embedding) and ignore_embeddings):
                for p in child.parameters(recurse=False):
                    model_size += p.numel() * p.element_size()
                for b in child.buffers(recurse=False):
                    model_size += b.numel() * b.element_size()
                model_size += self.get_model_size_in_bytes(child, ignore_embeddings)
        return model_size
    
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

    def _test_quantization_output(self, quantization_config: FinegrainedFP8Config, expected_slice: List[float], model_id: str):
        components = self.get_dummy_components(quantization_config, model_id)
        pipe = FluxPipeline(**components)
        pipe.to(device=torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]
        output_slice = output[-1, -1, -3:, -3:].flatten()

        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_quantization(self):
        expected_slice = [np.array([0.34179688, -0.03613281, 0.01428223, -0.22949219, -0.49609375, 0.4375, -0.1640625, -0.66015625, 0.43164062]), np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])]
        for index, model_id in enumerate(["hf-internal-testing/tiny-flux-pipe", "hf-internal-testing/tiny-flux-sharded"]):
            quantization_config = FinegrainedFP8Config(
                modules_to_not_convert=["x_embedder", "proj_out"], 
                weight_block_size=(32, 32)
            )
            self._test_quantization_output(quantization_config, model_id, expected_slice[index])

    def test_dtype(self):
        """
        Tests whether the dtype of the weight and weight_scale_inv are correct
        """
        quantization_config = FinegrainedFP8Config(
            modules_to_not_convert=["x_embedder", "proj_out"], 
            weight_block_size=(32, 32)
        )
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        weight = quantized_model.transformer_blocks[0].ff.net[2].weight
        weight_scale_inv = quantized_model.transformer_blocks[0].ff.net[2].weight_scale_inv
        self.assertTrue(isinstance(weight, FP8Linear))
        self.assertEqual(weight_scale_inv.dtype, torch.bfloat16)
        self.assertEqual(weight.weight.dtype, torch.float8_e4m3fn)

    def test_device_map_auto(self):
        """
        Test if the quantized model is working properly with "auto"
        """

        inputs = self.get_dummy_tensor_inputs(torch_device)
        # requires with different expected slices since models are different due to offload (we don't quantize modules offloaded to cpu/disk)
        expected_slice_auto = np.array(
            [
                0.34179688,
                -0.03613281,
                0.01428223,
                -0.22949219,
                -0.49609375,
                0.4375,
                -0.1640625,
                -0.66015625,
                0.43164062,
            ]
        )

        quantization_config = FinegrainedFP8Config(
            modules_to_not_convert=["x_embedder", "proj_out"], 
            weight_block_size=(32, 32)
        )
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        output = quantized_model(**inputs)[0]
        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(numpy_cosine_similarity_distance(output_slice, expected_slice_auto) < 1e-3)


    def test_modules_to_not_convert(self):
        quantization_config = FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["transformer_blocks.0", "proj_out", "x_embedder"])
        quantized_model_with_not_convert = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        unquantized_layer = quantized_model_with_not_convert.transformer_blocks[0].ff.net[2]
        self.assertTrue(isinstance(unquantized_layer, torch.nn.Linear))
        self.assertEqual(unquantized_layer.weight.dtype, torch.bfloat16)

        quantization_config = FinegrainedFP8Config(
            modules_to_not_convert=["x_embedder", "proj_out"], 
            weight_block_size=(32, 32)
        )
        quantized_model = FluxTransformer2DModel.from_pretrained(
            "hf-internal-testing/tiny-flux-pipe",
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        size_quantized_with_not_convert = self.get_model_size_in_bytes(quantized_model_with_not_convert)
        size_quantized = self.get_model_size_in_bytes(quantized_model)

        self.assertTrue(size_quantized < size_quantized_with_not_convert)

    def test_training(self):
        quantization_config = FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"])
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
            quantization_config = FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"])
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
            transformer_quantized = self.get_dummy_components(FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"]), model_id=model_id)["transformer"]
            transformer_bf16 = self.get_dummy_components(None, model_id=model_id)["transformer"]

            for name, module in transformer_quantized.named_modules():
                if isinstance(module, nn.Linear) and name not in ["x_embedder", "proj_out"]:
                    self.assertTrue(isinstance(module.weight, FP8Linear))


            total_quantized = self.get_model_size_in_bytes(transformer_quantized)
            total_bf16 = self.get_model_size_in_bytes(transformer_bf16)

            self.assertTrue(total_quantized < total_bf16)

    def test_model_memory_usage(self):
        model_id = "hf-internal-testing/tiny-flux-pipe"
        expected_memory_saving_ratio = 2.0

        inputs = self.get_dummy_tensor_inputs(device=torch_device)

        transformer_bf16 = self.get_dummy_components(None, model_id=model_id)["transformer"]
        transformer_bf16.to(torch_device)
        unquantized_model_memory = get_memory_consumption_stat(transformer_bf16, inputs)
        del transformer_bf16

        transformer_quantized = self.get_dummy_components(FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"]), model_id=model_id)["transformer"]
        transformer_quantized.to(torch_device)
        quantized_model_memory = get_memory_consumption_stat(transformer_quantized, inputs)
        self.assertTrue(unquantized_model_memory / quantized_model_memory >= expected_memory_saving_ratio)

    def test_exception_of_cpu_in_device_map(self):
        r"""
        A test that checks if inference runs as expected when sequential cpu offloading is enabled.
            """
        quantization_config = FinegrainedFP8Config(
            modules_to_not_convert=["x_embedder", "proj_out"], 
            weight_block_size=(32, 32)
        )
        device_map = {"transformer_blocks.0": "cpu"}

        with self.assertRaisesRegex(ValueError, "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."):
            _ = FluxTransformer2DModel.from_pretrained(
                "hf-internal-testing/tiny-flux-pipe",
                subfolder="transformer",
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )



@require_torch
@require_torch_gpu
class TorchAoSerializationTest(unittest.TestCase):
    model_name = "hf-internal-testing/tiny-flux-pipe"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_model(self, device=None):
        quantization_config = FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"])
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

    def _test_original_model_expected_slice(self, expected_slice):
        quantized_model = self.get_dummy_model(torch_device)
        inputs = self.get_dummy_tensor_inputs(torch_device)
        output = quantized_model(**inputs)[0]
        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(numpy_cosine_similarity_distance(output_slice, expected_slice) < 1e-3)

    def _check_serialization_expected_slice(self, expected_slice, device):
        quantized_model = self.get_dummy_model(device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir, safe_serialization=False)
            loaded_quantized_model = FluxTransformer2DModel.from_pretrained(
                tmp_dir, torch_dtype=torch.bfloat16, use_safetensors=False
            ).to(device=torch_device)

        inputs = self.get_dummy_tensor_inputs(torch_device)
        output = loaded_quantized_model(**inputs)[0]

        output_slice = output.flatten()[-9:].detach().float().cpu().numpy()
        self.assertTrue(numpy_cosine_similarity_distance(output_slice, expected_slice) < 1e-3)

    def test_slice_output(self):
        expected_slice = np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])
        device = "cuda"
        self._test_original_model_expected_slice(expected_slice)
        self._check_serialization_expected_slice(expected_slice, device)

@require_torch
@require_torch_gpu
@slow
@nightly
class SlowTorchAoTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_dummy_components(self, quantization_config: FinegrainedFP8Config):
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

    def _test_quant_output(self, quantization_config, expected_slice):
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0].flatten()
        output_slice = np.concatenate((output[:16], output[-16:]))
        self.assertTrue(np.allclose(output_slice, expected_slice, atol=1e-3, rtol=1e-3))

    def test_quantization(self):
        # fmt: off
        expected_slice = np.array([0.3633, -0.1357, -0.0188, -0.249, -0.4688, 0.5078, -0.1289, -0.6914, 0.4551])
        # fmt: on

        quantization_config = FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"])
        self._test_quant_output(quantization_config, expected_slice)

    def test_serialization(self):
        quantization_config = FinegrainedFP8Config(weight_block_size=(32, 32), modules_to_not_convert=["x_embedder", "proj_out"])
        components = self.get_dummy_components(quantization_config)
        pipe = FluxPipeline(**components)

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

        loaded_output = pipe(**inputs)[0].flatten()[:128]
        # Seems to require higher tolerance depending on which machine it is being run.
        # A difference of 0.06 in normalized pixel space (-1 to 1), corresponds to a difference of
        # 0.06 / 2 * 255 = 7.65 in pixel space (0 to 255). On our CI runners, the difference is about 0.04,
        # on DGX it is 0.06, and on audace it is 0.037. So, we are using a tolerance of 0.06 here.
        self.assertTrue(np.allclose(output, loaded_output, atol=0.06))