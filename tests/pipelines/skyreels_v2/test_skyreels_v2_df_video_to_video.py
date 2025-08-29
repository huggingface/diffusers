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

import inspect
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingVideoToVideoPipeline,
    SkyReelsV2Transformer3DModel,
    UniPCMultistepScheduler,
)

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineTesterMixin,
)


enable_full_determinism()


class SkyReelsV2DiffusionForcingVideoToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SkyReelsV2DiffusionForcingVideoToVideoPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = frozenset(["video", "prompt", "negative_prompt"])
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
        scheduler = UniPCMultistepScheduler(flow_shift=5.0, use_flow_sigmas=True)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = SkyReelsV2Transformer3DModel(
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

        video = [Image.new("RGB", (16, 16))] * 7
        inputs = {
            "video": video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "generator": generator,
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 16,
            "output_type": "pt",
            "overlap_history": 3,
            "num_frames": 17,
            "base_num_frames": 5,
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

        total_frames = len(inputs["video"]) + inputs["num_frames"]
        expected_shape = (total_frames, 3, 16, 16)
        self.assertEqual(generated_video.shape, expected_shape)
        expected_video = torch.randn(*expected_shape)
        max_diff = np.abs(generated_video - expected_video).max()
        self.assertLessEqual(max_diff, 1e10)

    def test_callback_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        if "guidance_scale" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        # Track the number of callback calls for diffusion forcing pipelines
        callback_call_count = [0]  # Use list to make it mutable in closure

        def callback_increase_guidance(pipe, i, t, callback_kwargs):
            pipe._guidance_scale += 1.0
            callback_call_count[0] += 1
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # use cfg guidance because some pipelines modify the shape of the latents
        # outside of the denoising loop
        inputs["guidance_scale"] = 2.0
        inputs["callback_on_step_end"] = callback_increase_guidance
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

        # For diffusion forcing pipelines, use the actual callback count
        # since they run multiple iterations with nested denoising loops
        expected_guidance_scale = inputs["guidance_scale"] + callback_call_count[0]

        assert pipe.guidance_scale == expected_guidance_scale

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip(
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline has to run in mixed precision. Casting the entire pipeline will result in errors"
    )
    def test_float16_inference(self):
        pass

    @unittest.skip(
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline has to run in mixed precision. Save/Load the entire pipeline in FP16 will result in errors"
    )
    def test_save_load_float16(self):
        pass
