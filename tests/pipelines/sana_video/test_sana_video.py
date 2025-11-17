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
import tempfile
import unittest

import numpy as np
import torch
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizer

from diffusers import AutoencoderKLWan, DPMSolverMultistepScheduler, SanaVideoPipeline, SanaVideoTransformer3DModel

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


class SanaVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SanaVideoPipeline
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
        scheduler = DPMSolverMultistepScheduler()

        torch.manual_seed(0)
        text_encoder_config = Gemma2Config(
            head_dim=16,
            hidden_size=8,
            initializer_range=0.02,
            intermediate_size=64,
            max_position_embeddings=8192,
            model_type="gemma2",
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=2,
            vocab_size=8,
            attn_implementation="eager",
        )
        text_encoder = Gemma2Model(text_encoder_config)
        tokenizer = GemmaTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        torch.manual_seed(0)
        transformer = SanaVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            num_attention_heads=2,
            attention_head_dim=12,
            num_layers=2,
            num_cross_attention_heads=2,
            cross_attention_head_dim=12,
            cross_attention_dim=24,
            caption_channels=8,
            mlp_ratio=2.5,
            dropout=0.0,
            attention_bias=False,
            sample_size=8,
            patch_size=(1, 2, 2),
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
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
            "prompt": "",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "frames": 9,
            "max_sequence_length": 16,
            "output_type": "pt",
            "complex_human_instruction": [],
            "use_resolution_binning": False,
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
        self.assertEqual(generated_video.shape, (9, 3, 32, 32))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    def test_save_load_local(self, expected_max_difference=5e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output.detach().cpu().numpy() - output_loaded.detach().cpu().numpy()).max()
        self.assertLess(max_diff, expected_max_difference)

    # TODO(aryan): Create a dummy gemma model with smol vocab size
    @unittest.skip(
        "A very small vocab size is used for fast tests. So, any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip(
        "A very small vocab size is used for fast tests. So, any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_single_identical(self):
        pass

    def test_float16_inference(self):
        # Requires higher tolerance as model seems very sensitive to dtype
        super().test_float16_inference(expected_max_diff=0.08)

    def test_save_load_float16(self):
        # Requires higher tolerance as model seems very sensitive to dtype
        super().test_save_load_float16(expected_max_diff=0.2)


@slow
@require_torch_accelerator
class SanaVideoPipelineIntegrationTests(unittest.TestCase):
    prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest."

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    @unittest.skip("TODO: test needs to be implemented")
    def test_sana_video_480p(self):
        pass
