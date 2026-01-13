# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import torch

from diffusers import F5ConditioningEncoder, F5DiTModel, F5FlowPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_xformers_available

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class F5TTSPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = F5FlowPipeline
    params = frozenset(
        [
            "ref_text",
            "gen_text",
            "guidance_scale",
        ]
    )
    batch_params = frozenset(["ref_text", "gen_text", "speed"])
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "return_dict",
        ]
    )
    # There is not xformers version of the F5FlowPipeline custom attention processor
    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)

        transformer = F5DiTModel(
            dim=1024,
            depth=2,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_num_embeds=256,
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=4,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
            checkpoint_activations=False,
        )

        conditioning_encoder = F5ConditioningEncoder(
            dim=1024,
            text_num_embeds=2546,
            text_dim=512,
            text_mask_padding=True,
            conv_layers=4,
            mel_dim=100,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()
        torch.manual_seed(0)

        vocab_char_map = {chr(i + 97): i for i in range(26)}

        components = {
            "transformer": transformer.eval(),
            "scheduler": scheduler,
            "conditioning_encoder": conditioning_encoder.eval(),
            "vocab_char_map": vocab_char_map,
        }
        return components

    def test_components_function(self):
        # Override to filter out dict (vocab_char_map) which can't be registered as a module
        # TODO vocab map needs some better handling
        init_components = self.get_dummy_components()
        init_components = {k: v for k, v in init_components.items() if not isinstance(v, (str, int, float, dict))}

        pipe = self.pipeline_class(**self.get_dummy_components())

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        torch.manual_seed(0)
        ref_audio = torch.randn(1, 16000).to(torch_device)
        speed = torch.tensor([1], device=torch_device)
        inputs = {
            "ref_text": "This is a test sentence",
            "gen_text": "This is another test sentence",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "ref_audio": ref_audio,
            "speed": speed,
        }
        return inputs

    def test_save_load_local(self):
        # increase tolerance from 1e-4 -> 7e-3 to account for large composite model
        super().test_save_load_local(expected_max_difference=7e-3)

    def test_save_load_optional_components(self):
        # increase tolerance from 1e-4 -> 7e-3 to account for large composite model
        super().test_save_load_optional_components(expected_max_difference=7e-3)

    def test_f5tts_forward_pass(self):
        components = self.get_dummy_components()
        f5tts_pipe = F5FlowPipeline(**components)
        f5tts_pipe = f5tts_pipe.to(torch_device)
        f5tts_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = f5tts_pipe(**inputs)
        audio = output.audios[0]

        assert audio.ndim == 2
        assert audio.shape == (100, 204)

    def test_attention_slicing_forward_pass(self):
        self._test_attention_slicing_forward_pass(test_mean_pixel_difference=False)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=5e-4)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(test_mean_pixel_difference=False)

    @unittest.skip("Not supported yet")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @unittest.skip("Not supported yet")
    def test_sequential_offload_forward_pass_twice(self):
        pass

    @unittest.skip("Test not supported because `rotary_embed_dim` doesn't have any sensible default.")
    def test_encode_prompt_works_in_isolation(self):
        pass
