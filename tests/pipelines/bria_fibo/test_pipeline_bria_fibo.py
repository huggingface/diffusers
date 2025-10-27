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

import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config, SmolLM3ForCausalLM

from diffusers import (
    AutoencoderKLWan,
    BriaFiboPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
from tests.pipelines.test_pipelines_common import PipelineTesterMixin, to_np

from ...testing_utils import (
    enable_full_determinism,
    require_torch_accelerator,
    torch_device,
)


enable_full_determinism()


class BriaFiboPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BriaFiboPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_params = frozenset(["prompt"])
    test_xformers_attention = False
    test_layerwise_casting = False
    test_group_offloading = False

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

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=160,
            decoder_base_dim=256,
            num_res_blocks=2,
            out_channels=12,
            patch_size=2,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            temperal_downsample=[False, True, True],
            z_dim=16,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
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
            "prompt": "{'text': 'A painting of a squirrel eating a burger'}",
            "negative_prompt": "bad, ugly",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        return inputs

    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_bria_fibo_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        pipe = pipe.to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = "a different prompt"
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

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_torch_accelerator
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if name == "vae":
                continue
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )

    # def test_to_dtype(self):
    #     components = self.get_dummy_components()
    #     pipe = self.pipeline_class(**components)
    #     pipe.set_progress_bar_config(disable=None)

    #     model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
    #     self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

    #     pipe.to(dtype=torch.float16)
    #     model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
    #     self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))

    @unittest.skip("")
    def test_save_load_dduf(self):
        pass
