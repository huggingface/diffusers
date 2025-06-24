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

import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.utils.testing_utils import enable_full_determinism

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class WanVACEPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WanVACEPipeline
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
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanVACETransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=3,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
            vace_layers=[0, 2],
            vace_in_channels=96,
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

        num_frames = 17
        height = 16
        width = 16

        video = [Image.new("RGB", (height, width))] * num_frames
        mask = [Image.new("L", (height, width), 0)] * num_frames

        inputs = {
            "video": video,
            "mask": mask,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": num_frames,
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
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = [0.4523, 0.45198, 0.44872, 0.45326, 0.45211, 0.45258, 0.45344, 0.453, 0.52431, 0.52572, 0.50701, 0.5118, 0.53717, 0.53093, 0.50557, 0.51402]
        # fmt: on

        video_slice = video.flatten()
        video_slice = torch.cat([video_slice[:8], video_slice[-8:]])
        video_slice = [round(x, 5) for x in video_slice.tolist()]
        self.assertTrue(np.allclose(video_slice, expected_slice, atol=1e-3))

    def test_inference_with_single_reference_image(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["reference_images"] = Image.new("RGB", (16, 16))
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = [0.45247, 0.45214, 0.44874, 0.45314, 0.45171, 0.45299, 0.45428, 0.45317, 0.51378, 0.52658, 0.53361, 0.52303, 0.46204, 0.50435, 0.52555, 0.51342]
        # fmt: on

        video_slice = video.flatten()
        video_slice = torch.cat([video_slice[:8], video_slice[-8:]])
        video_slice = [round(x, 5) for x in video_slice.tolist()]
        self.assertTrue(np.allclose(video_slice, expected_slice, atol=1e-3))

    def test_inference_with_multiple_reference_image(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["reference_images"] = [[Image.new("RGB", (16, 16))] * 2]
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = [0.45321, 0.45221, 0.44818, 0.45375, 0.45268, 0.4519, 0.45271, 0.45253, 0.51244, 0.52223, 0.51253, 0.51321, 0.50743, 0.51177, 0.51626, 0.50983]
        # fmt: on

        video_slice = video.flatten()
        video_slice = torch.cat([video_slice[:8], video_slice[-8:]])
        video_slice = [round(x, 5) for x in video_slice.tolist()]
        self.assertTrue(np.allclose(video_slice, expected_slice, atol=1e-3))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("Errors out because passing multiple prompts at once is not yet supported by this pipeline.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip("Batching is not yet supported with this pipeline")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("Batching is not yet supported with this pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()

    @unittest.skip(
        "AutoencoderKLWan encoded latents are always in FP32. This test is not designed to handle mixed dtype inputs"
    )
    def test_float16_inference(self):
        pass

    @unittest.skip(
        "AutoencoderKLWan encoded latents are always in FP32. This test is not designed to handle mixed dtype inputs"
    )
    def test_save_load_float16(self):
        pass
