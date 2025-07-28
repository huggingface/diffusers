# Copyright 2024 Bria AI and The HuggingFace Team. All rights reserved.
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
import unittest
import tempfile

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    BriaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.bria import BriaPipeline
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)


# from ..test_pipelines_common import PipelineTesterMixin, check_qkv_fused_layers_exist
from tests.pipelines.test_pipelines_common import PipelineTesterMixin, check_qkv_fused_layers_exist

enable_full_determinism()



class BriaPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BriaPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds"])
    batch_params = frozenset(["prompt"])
    test_xformers_attention = False

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True
    
    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BriaTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=None,
            axes_dims_rope=[0, 4, 4],
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=32,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "bad, ugly",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs
    
    def test_bria_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]
        max_diff = np.abs(output_same_prompt - output_different_prompts).max()
        assert max_diff > 1e-6

    

    
    def test_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_inference(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 32, 32, 3))
        expected_slice = np.array(
            [0.5361328, 0.5253906, 0.5234375, 0.5292969, 0.5214844, 0.5185547, 0.5283203, 0.5205078, 0.519043]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice.flatten())
        self.assertLess(max_diff, 1e-4)

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # check that all modules are float32 except for text_encoder
        for name, module in pipe.components.items():
            if not hasattr(module, "dtype"):
                continue

            if name == "text_encoder":
                self.assertEqual(module.dtype, torch.float16)
            else:
                self.assertEqual(module.dtype, torch.float32)

        pipe.to(torch.bfloat16)

        # check that all modules are bfloat16 except for text_encoder (float16) and vae (float32)
        for name, module in pipe.components.items():
            if not hasattr(module, "dtype"):
                continue

            if name == "text_encoder":
                self.assertEqual(module.dtype, torch.float16)
            elif name == "vae":
                self.assertEqual(module.dtype, torch.float32)
            else:
                self.assertEqual(module.dtype, torch.bfloat16)

    
    def test_bria_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(16, 16), (32, 32), (64, 64)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)


    def test_torch_dtype_dict(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            torch_dtype_dict = {"transformer": torch.bfloat16, "default": torch.float16}
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype_dict)

            self.assertEqual(loaded_pipe.transformer.dtype, torch.bfloat16)
            self.assertEqual(loaded_pipe.text_encoder.dtype, torch.float16)
            self.assertEqual(loaded_pipe.vae.dtype, torch.float32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            torch_dtype_dict = {"default": torch.float16}
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype_dict)

            self.assertEqual(loaded_pipe.transformer.dtype, torch.float16)
            self.assertEqual(loaded_pipe.text_encoder.dtype, torch.float16)
            self.assertEqual(loaded_pipe.vae.dtype, torch.float32)

    


@slow
@require_torch_gpu
class BriaPipelineSlowTests(unittest.TestCase):
    pipeline_class = BriaPipeline
    repo_id = "briaai/BRIA-3.2"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        prompt_embeds = torch.load(
            hf_hub_download(
                repo_id="diffusers/test-slices", repo_type="dataset", filename="bria_prompt_embeds.pt"
            )
        ).to(device)
        return {
            "prompt_embeds": prompt_embeds,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "output_type": "np",
            "generator": generator,
        }

    def test_bria_inference_bf16(self):
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None
        )
        pipe.to(torch_device)

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10, 0].flatten()

        expected_slice = np.array(
            [
                0.3242,
                0.3203,
                0.3164,
                0.3164,
                0.3125,
                0.3125,
                0.3281,
                0.3242,
                0.3203,
                0.3301,
                0.3262,
                0.3242,
                0.3281,
                0.3242,
                0.3203,
                0.3262,
                0.3262,
                0.3164,
                0.3262,
                0.3281,
                0.3184,
                0.3281,
                0.3281,
                0.3203,
                0.3281,
                0.3281,
                0.3164,
                0.332,
                0.332,
                0.3203,
            ]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice, image_slice)
        self.assertLess(max_diff, 1e-4, f"Image slice is different from expected slice: {max_diff:.4f}")


@nightly
@require_torch_gpu
class BriaPipelineNightlyTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_bria_inference(self):
        pipe = BriaPipeline.from_pretrained("briaai/BRIA-3.2", torch_dtype=torch.bfloat16)
        pipe.to(torch_device)

        prompt = "a close-up of a smiling cat, high quality, realistic"
        image = pipe(prompt=prompt, num_inference_steps=5, output_type="np").images[0]

        image_slice = image[0, :10, :10, 0].flatten()
        expected_slice = np.array(
            [
                0.668,
                0.668,
                0.6641,
                0.6602,
                0.6602,
                0.6562,
                0.6523,
                0.6484,
                0.6523,
                0.6562,
                0.668,
                0.668,
                0.6641,
                0.6641,
                0.6602,
                0.6562,
                0.6523,
                0.6484,
                0.6523,
                0.6562,
                0.668,
                0.668,
                0.668,
                0.6641,
                0.6602,
                0.6562,
                0.6523,
                0.6484,
                0.6523,
                0.6562,
            ]
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice, image_slice)
        self.assertLess(max_diff, 1e-4, f"Image slice is different from expected slice: {max_diff:.4f}")

