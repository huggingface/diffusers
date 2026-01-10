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

import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel, WanVideoToVideoPipeline

from ...testing_utils import (
    enable_full_determinism,
)
from ..pipeline_params import TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineTesterMixin,
)


enable_full_determinism()


class WanVideoToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WanVideoToVideoPipeline
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
        scheduler = UniPCMultistepScheduler(flow_shift=3.0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
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

        video = [Image.new("RGB", (16, 16))] * 17
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
        self.assertEqual(generated_video.shape, (17, 3, 16, 16))

        # fmt: off
        expected_slice = torch.tensor([0.4522, 0.4534, 0.4532, 0.4553, 0.4526, 0.4538, 0.4533, 0.4547, 0.513, 0.5176, 0.5286, 0.4958, 0.4955, 0.5381, 0.5154, 0.5195])
        # fmt:on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-3))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip(
        "WanVideoToVideoPipeline has to run in mixed precision. Casting the entire pipeline will result in errors"
    )
    def test_float16_inference(self):
        pass

    @unittest.skip(
        "WanVideoToVideoPipeline has to run in mixed precision. Save/Load the entire pipeline in FP16 will result in errors"
    )
    def test_save_load_float16(self):
        pass
