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
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    ChronoEditPipeline,
    ChronoEditTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)

from ...testing_utils import enable_full_determinism
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class ChronoEditPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ChronoEditPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs", "height", "width"}
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
        # TODO: impl FlowDPMSolverMultistepScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = ChronoEditTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
            image_dim=4,
        )

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=4,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            intermediate_size=16,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        torch.manual_seed(0)
        image_processor = CLIPImageProcessor(crop_size=32, size=32)

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "height": image_height,
            "width": image_width,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "num_frames": 5,
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
        self.assertEqual(generated_video.shape, (5, 3, 16, 16))

        # fmt: off
        expected_slice = torch.tensor([0.4525, 0.4520, 0.4485, 0.4534, 0.4523, 0.4522, 0.4529, 0.4528, 0.5022, 0.5064, 0.5011, 0.5061, 0.5028, 0.4979, 0.5117, 0.5192])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-3))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("TODO: revisit failing as it requires a very high threshold to pass")
    def test_inference_batch_single_identical(self):
        pass

    @unittest.skip(
        "ChronoEditPipeline has to run in mixed precision. Save/Load the entire pipeline in FP16 will result in errors"
    )
    def test_save_load_float16(self):
        pass
