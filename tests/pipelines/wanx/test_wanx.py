# Copyright 2024 The HuggingFace Team.
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
import inspect
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLHunyuanVideo, WanxPipeline, WanxTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import (
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
    to_np,
)


enable_full_determinism()


class WanxPipelineFastTests(PipelineTesterMixin, PyramidAttentionBroadcastTesterMixin, unittest.TestCase):
    pipeline_class = WanxPipeline
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
    # TODO: test below
    test_layerwise_casting = False
    test_group_offloading = False

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        # TODO: impl AutoencoderKLWanx
        vae = AutoencoderKLHunyuanVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=(
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            up_block_types=(
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=4,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
            mid_block_add_attention=True,
        )

        torch.manual_seed(0)
        # TODO: impl FlowDPMSolverMultistepScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanxTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads = 12,
            attention_head_dim = 128,
            in_channels = 16,
            out_channels = 16,
            text_dim = text_encoder.config.d_model,
            freq_dim = 256,
            ffn_dim = 8960,
            num_layers = num_layers,
            window_size = (-1, -1),
            cross_attn_norm = True,
            qk_norm = True,
            eps = 1e-6,
            add_img_emb = False,
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
            "negative_prompt": "negative", # TODO
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": 8,
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

        # self.assertEqual(generated_video.shape, (8, 3, 16, 16))
        # expected_video = torch.randn(8, 3, 16, 16)
        # max_diff = np.abs(generated_video - expected_video).max()
        # self.assertLessEqual(max_diff, 1e10)

    # def test_callback_inputs(self):
    #     sig = inspect.signature(self.pipeline_class.__call__)
    #     has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
    #     has_callback_step_end = "callback_on_step_end" in sig.parameters

    #     if not (has_callback_tensor_inputs and has_callback_step_end):
    #         return

    #     components = self.get_dummy_components()
    #     pipe = self.pipeline_class(**components)
    #     pipe = pipe.to(torch_device)
    #     pipe.set_progress_bar_config(disable=None)
    #     self.assertTrue(
    #         hasattr(pipe, "_callback_tensor_inputs"),
    #         f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
    #     )

    #     def callback_inputs_subset(pipe, i, t, callback_kwargs):
    #         # iterate over callback args
    #         for tensor_name, tensor_value in callback_kwargs.items():
    #             # check that we're only passing in allowed tensor inputs
    #             assert tensor_name in pipe._callback_tensor_inputs

    #         return callback_kwargs

    #     def callback_inputs_all(pipe, i, t, callback_kwargs):
    #         for tensor_name in pipe._callback_tensor_inputs:
    #             assert tensor_name in callback_kwargs

    #         # iterate over callback args
    #         for tensor_name, tensor_value in callback_kwargs.items():
    #             # check that we're only passing in allowed tensor inputs
    #             assert tensor_name in pipe._callback_tensor_inputs

    #         return callback_kwargs

    #     inputs = self.get_dummy_inputs(torch_device)

    #     # Test passing in a subset
    #     inputs["callback_on_step_end"] = callback_inputs_subset
    #     inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
    #     output = pipe(**inputs)[0]

    #     # Test passing in a everything
    #     inputs["callback_on_step_end"] = callback_inputs_all
    #     inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
    #     output = pipe(**inputs)[0]

    #     def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
    #         is_last = i == (pipe.num_timesteps - 1)
    #         if is_last:
    #             callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
    #         return callback_kwargs

    #     inputs["callback_on_step_end"] = callback_inputs_change_tensor
    #     inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
    #     output = pipe(**inputs)[0]
    #     assert output.abs().sum() < 1e10


@slow
@require_torch_accelerator
class WanxPipelineIntegrationTests(unittest.TestCase):
    prompt = "A painting of a squirrel eating a burger."

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_Wanx(self):
        pass
        # generator = torch.Generator("cpu").manual_seed(0)

        # pipe = WanxPipeline.from_pretrained("THUDM/Wanx-2b", torch_dtype=torch.float16)
        # pipe.enable_model_cpu_offload(device=torch_device)
        # prompt = self.prompt

        # videos = pipe(
        #     prompt=prompt,
        #     height=480,
        #     width=720,
        #     num_frames=16,
        #     generator=generator,
        #     num_inference_steps=2,
        #     output_type="pt",
        # ).frames

        # video = videos[0]
        # expected_video = torch.randn(1, 16, 480, 720, 3).numpy()

        # max_diff = numpy_cosine_similarity_distance(video, expected_video)
        # assert max_diff < 1e-3, f"Max diff is too high. got {video}"
