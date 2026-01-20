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

import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config, SmolLM3ForCausalLM

from diffusers import (
    AutoencoderKLWan,
    BriaFiboEditPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
from tests.pipelines.test_pipelines_common import PipelineTesterMixin

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)


enable_full_determinism()


class BriaFiboPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BriaFiboEditPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_params = frozenset(["prompt"])
    test_xformers_attention = False
    test_layerwise_casting = False
    test_group_offloading = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BriaFiboTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=64,
            text_encoder_dim=32,
            pooled_projection_dim=None,
            axes_dims_rope=[0, 4, 4],
        )

        vae = AutoencoderKLWan(
            base_dim=80,
            decoder_base_dim=128,
            dim_mult=[1, 2, 4, 4],
            dropout=0.0,
            in_channels=12,
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
            is_residual=True,
            num_res_blocks=2,
            out_channels=12,
            patch_size=2,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            temperal_downsample=[False, True, True],
            z_dim=16,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        text_encoder = SmolLM3ForCausalLM(SmolLM3Config(hidden_size=32))
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            "prompt": '{"text": "A painting of a squirrel eating a burger","edit_instruction": "A painting of a squirrel eating a burger"}',
            "negative_prompt": "bad, ugly",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 192,
            "width": 336,
            "output_type": "np",
        }
        image = Image.new("RGB", (336, 192), (255, 255, 255))
        inputs["image"] = image
        return inputs

    @unittest.skip(reason="will not be supported due to dim-fusion")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip(reason="Batching is not supported yet")
    def test_num_images_per_prompt(self):
        pass

    @unittest.skip(reason="Batching is not supported yet")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip(reason="Batching is not supported yet")
    def test_inference_batch_single_identical(self):
        pass

    def test_bria_fibo_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe = pipe.to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = {"edit_instruction": "a different prompt"}
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()
        assert max_diff > 1e-6

    def test_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe = pipe.to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (64, 64), (32, 64)]
        for height, width in height_width_pairs:
            expected_height = height
            expected_width = width

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_bria_fibo_edit_mask(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe = pipe.to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        mask = Image.fromarray((np.ones((192, 336)) * 255).astype(np.uint8), mode="L")

        inputs.update({"mask": mask})
        output = pipe(**inputs).images[0]

        assert output.shape == (192, 336, 3)

    def test_bria_fibo_edit_mask_image_size_mismatch(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe = pipe.to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        mask = Image.fromarray((np.ones((64, 64)) * 255).astype(np.uint8), mode="L")

        inputs.update({"mask": mask})
        with self.assertRaisesRegex(ValueError, "Mask and image must have the same size"):
            pipe(**inputs)

    def test_bria_fibo_edit_mask_no_image(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe = pipe.to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        mask = Image.fromarray((np.ones((32, 32)) * 255).astype(np.uint8), mode="L")

        # Remove image from inputs if it's there (it shouldn't be by default from get_dummy_inputs)
        inputs.pop("image", None)
        inputs.update({"mask": mask})

        with self.assertRaisesRegex(ValueError, "If mask is provided, image must also be provided"):
            pipe(**inputs)
