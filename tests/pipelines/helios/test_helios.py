# Copyright 2025 The HuggingFace Team.
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

import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, HeliosPipeline, HeliosScheduler, HeliosTransformer3DModel

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class HeliosPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = HeliosPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = HeliosScheduler(stage_range=[0, 1], stages=1, use_dynamic_shifting=True)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = HeliosTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_dim=(4, 4, 4),
            has_multi_term_memory_patch=True,
            guidance_cross_attn=True,
            zero_history_timestep=True,
            is_amplify_history=False,
        )

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames
        generated_video = video[0]
        self.assertEqual(generated_video.shape, (33, 3, 16, 16))

        # fmt: off
        expected_slice = torch.tensor([0.4529, 0.4527, 0.4499, 0.4542, 0.4528, 0.4524, 0.4531, 0.4534, 0.5328,
        0.5340, 0.5012, 0.5135, 0.5322, 0.5203, 0.5144, 0.5101])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-3))

    # Override to set a more lenient max diff threshold.
    def test_save_load_float16(self):
        super().test_save_load_float16(expected_max_diff=0.03)

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("Optional components not applicable for Helios")
    def test_save_load_optional_components(self):
        pass


@slow
@require_torch_accelerator
class HeliosPipelineIntegrationTests(unittest.TestCase):
    prompt = "A painting of a squirrel eating a burger."

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    @unittest.skip("TODO: test needs to be implemented")
    def test_helios(self):
        pass
