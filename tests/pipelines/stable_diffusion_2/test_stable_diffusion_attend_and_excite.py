# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionAttendAndExcitePipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    skip_mps,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineFromPipeTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


torch.backends.cuda.matmul.allow_tf32 = False


@skip_mps
class StableDiffusionAttendAndExcitePipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    PipelineFromPipeTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionAttendAndExcitePipeline
    test_attention_slicing = False
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS.union({"token_indices"})
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    # Attend and excite requires being able to run a backward pass at
    # inference time. There's no deterministic backward operator for pad

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.use_deterministic_algorithms(False)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        torch.use_deterministic_algorithms(True)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "a cat and a frog",
            "token_indices": [2, 5],
            "generator": generator,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "output_type": "np",
            "max_iter_to_alter": 2,
            "thresholds": {0: 0.7},
        }
        return inputs

    def test_dict_tuple_outputs_equivalent(self):
        expected_slice = None
        if torch_device == "cpu":
            expected_slice = np.array([0.6391, 0.6290, 0.4860, 0.5134, 0.5550, 0.4577, 0.5033, 0.5023, 0.4538])
        super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice, expected_max_difference=3e-3)

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 64, 64, 3))
        expected_slice = np.array(
            [0.63905364, 0.62897307, 0.48599017, 0.5133624, 0.5550048, 0.45769516, 0.50326973, 0.5023139, 0.45384496]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_sequential_cpu_offload_forward_pass(self):
        super().test_sequential_cpu_offload_forward_pass(expected_max_diff=5e-4)

    def test_inference_batch_consistent(self):
        # NOTE: Larger batch sizes cause this test to timeout, only test on smaller batches
        self._test_inference_batch_consistent(batch_sizes=[1, 2])

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=2, expected_max_diff=7e-4)

    def test_pt_np_pil_outputs_equivalent(self):
        super().test_pt_np_pil_outputs_equivalent(expected_max_diff=5e-4)

    def test_save_load_local(self):
        super().test_save_load_local(expected_max_difference=5e-4)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=4e-4)

    def test_karras_schedulers_shape(self):
        super().test_karras_schedulers_shape(num_inference_steps_for_strength_for_iterations=3)

    def test_from_pipe_consistent_forward_pass_cpu_offload(self):
        super().test_from_pipe_consistent_forward_pass_cpu_offload(expected_max_diff=5e-3)

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs(device=torch_device).get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


@require_torch_accelerator
@nightly
class StableDiffusionAttendAndExcitePipelineIntegrationTests(unittest.TestCase):
    # Attend and excite requires being able to run a backward pass at
    # inference time. There's no deterministic backward operator for pad

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.use_deterministic_algorithms(False)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_attend_and_excite_fp16(self):
        generator = torch.manual_seed(51)

        pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", safety_checker=None, torch_dtype=torch.float16
        )
        pipe.to(torch_device)

        prompt = "a painting of an elephant with glasses"
        token_indices = [5, 7]

        image = pipe(
            prompt=prompt,
            token_indices=token_indices,
            guidance_scale=7.5,
            generator=generator,
            num_inference_steps=5,
            max_iter_to_alter=5,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/attend-and-excite/elephant_glasses.npy"
        )
        max_diff = numpy_cosine_similarity_distance(image.flatten(), expected_image.flatten())
        assert max_diff < 5e-1
