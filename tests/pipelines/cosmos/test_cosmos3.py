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

from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from diffusers import AutoencoderKLWan, Cosmos3OmniPipeline, Cosmos3OmniTransformer, UniPCMultistepScheduler
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _preprocess_conditioning_image

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class Cosmos3OmniPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Cosmos3OmniPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 16, 16)
    # Cosmos3 Omni generates one video per call, so it exposes neither `num_images_per_prompt` (the base default)
    # nor the `num_videos_per_prompt` the other Cosmos pipelines take.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])

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

    def get_dummy_inputs(self):
        return {
            "prompt": "a dog",
            "negative_prompt": "bad quality",
            "height": 16,
            "width": 16,
            "num_frames": 1,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "generator": self.get_generator(0),
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "use_system_prompt": False,
            "add_resolution_template": False,
            "add_duration_template": False,
        }


class TestCosmos3OmniPipeline(Cosmos3OmniPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        video = pipe(**self.get_dummy_inputs()).video

        assert video[0].shape == self.output_shape

    def test_cosmos3_tokenize_prompt_uses_checkpoint_system_prompt_default(self):
        components = self.get_dummy_components()
        components["default_use_system_prompt"] = False
        pipeline = self.get_pipeline(**components)

        with mock.patch.object(
            pipeline.text_tokenizer,
            "apply_chat_template",
            wraps=pipeline.text_tokenizer.apply_chat_template,
        ) as apply_chat_template:
            pipeline.tokenize_prompt("A prompt", num_frames=1, add_resolution_template=False)

        assert all(call.args[0][0]["role"] == "user" for call in apply_chat_template.call_args_list)

    def test_i2v_image_preprocessing_preserves_aspect_ratio(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[:, 0] = [255, 0, 0]
        image[:, 1] = [0, 255, 0]
        image[:, 2] = [0, 0, 255]
        image[:, 3] = [255, 255, 255]

        actual = _preprocess_conditioning_image(Image.fromarray(image), height=2, width=2)
        expected_pixels = torch.tensor(
            [[[[0, 0], [0, 0]], [[255, 0], [255, 0]], [[0, 255], [0, 255]]]], dtype=torch.float32
        )
        expected = expected_pixels / 127.5 - 1.0

        torch.testing.assert_close(actual, expected)

    def test_i2v_pipeline_uses_native_preprocessing(self):
        pipeline = self.get_pipeline().to(torch_device)

        image = np.zeros((16, 32, 3), dtype=np.uint8)
        image[:, :8] = [255, 0, 0]
        image[:, 8:24] = [0, 255, 0]
        image[:, 24:] = [0, 0, 255]
        center_crop = Image.fromarray(image[:, 8:24])
        inputs = self.get_dummy_inputs()
        inputs.update(image=Image.fromarray(image), num_frames=5, output_type="latent")

        wide_output = pipeline(**inputs).video
        inputs.update(image=center_crop, generator=self.get_generator(0))
        crop_output = pipeline(**inputs).video

        torch.testing.assert_close(wide_output, crop_output)

    @pytest.mark.skip("Cosmos3 currently supports one prompt per pipeline call.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("Cosmos3 currently supports one prompt per pipeline call.")
    def test_inference_batch_single_identical(self):
        pass


class TestCosmos3OmniPipelineMemory(Cosmos3OmniPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Cosmos3 Omni pipeline."""
