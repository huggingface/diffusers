# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from unittest import mock

import torch
from transformers import AutoTokenizer

from diffusers import AutoencoderKLWan, Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Cosmos3OmniPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Cosmos3OmniPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs", "negative_prompt_embeds", "prompt_embeds"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = False
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Cosmos3OmniTransformer(
            head_dim=6,
            hidden_act="relu2",
            hidden_size=6,
            intermediate_size=12,
            latent_channel=16,
            latent_patch_size=1,
            num_attention_heads=1,
            num_hidden_layers=1,
            num_key_value_heads=1,
            patch_latent_dim=16,
            qk_norm_for_text=False,
            rms_norm_eps=1e-5,
            rope_axes_dim=[1, 1, 1],
            vocab_size=151657,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        text_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-cosmos3-modular-pipe", subfolder="text_tokenizer"
        )

        return {
            "transformer": transformer,
            "text_tokenizer": text_tokenizer,
            "vae": vae,
            "scheduler": UniPCMultistepScheduler(),
            "sound_tokenizer": None,
            # The inherited components test omits config flags and needs a non-None safety checker.
            "safety_checker": mock.Mock(spec=["check_text_safety", "check_video_safety"]),
            "enable_safety_checker": False,
        }

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return {
            "prompt": "a dog",
            "negative_prompt": "bad quality",
            "height": 16,
            "width": 16,
            "num_frames": 1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "generator": generator,
            "output_type": "np",
            "use_system_prompt": False,
            "add_resolution_template": False,
            "add_duration_template": False,
        }

    def test_inference(self):
        pipeline = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipeline.set_progress_bar_config(disable=None)

        video = pipeline(**self.get_dummy_inputs(torch_device)).video

        self.assertEqual(video.shape, (1, 16, 16, 3))

    def test_cosmos3_tokenize_prompt_uses_checkpoint_system_prompt_default(self):
        components = self.get_dummy_components()
        components["default_use_system_prompt"] = False
        pipeline = self.pipeline_class(**components)

        with mock.patch.object(
            pipeline.text_tokenizer,
            "apply_chat_template",
            wraps=pipeline.text_tokenizer.apply_chat_template,
        ) as apply_chat_template:
            pipeline.tokenize_prompt("A prompt", num_frames=1, add_resolution_template=False)

        assert all(call.args[0][0]["role"] == "user" for call in apply_chat_template.call_args_list)

    @unittest.skip("Cosmos3 currently supports one prompt per pipeline call.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("Cosmos3 currently supports one prompt per pipeline call.")
    def test_inference_batch_single_identical(self):
        pass
