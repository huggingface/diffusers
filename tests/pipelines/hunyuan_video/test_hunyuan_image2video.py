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
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    LlamaConfig,
    LlamaTokenizerFast,
    LlavaConfig,
    LlavaForConditionalGeneration,
)
from transformers.models.clip import CLIPVisionConfig

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoTransformer3DModel,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..test_pipelines_common import PipelineTesterMixin, PyramidAttentionBroadcastTesterMixin, to_np


enable_full_determinism()


class HunyuanVideoImageToVideoPipelineFastTests(
    PipelineTesterMixin, PyramidAttentionBroadcastTesterMixin, unittest.TestCase
):
    pipeline_class = HunyuanVideoImageToVideoPipeline
    params = frozenset(
        ["image", "prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_params = frozenset(["prompt", "image"])
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
    supports_dduf = False

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = HunyuanVideoTransformer3DModel(
            in_channels=2 * 4 + 1,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=10,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_refiner_layers=1,
            patch_size=1,
            patch_size_t=1,
            guidance_embeds=False,
            text_embed_dim=16,
            pooled_projection_dim=8,
            rope_axes_dim=(2, 4, 4),
            image_condition_type="latent_concat",
        )

        torch.manual_seed(0)
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
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        text_config = LlamaConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=100,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
        vision_config = CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=37,
            projection_dim=32,
            num_attention_heads=4,
            num_hidden_layers=2,
            image_size=224,
        )
        llava_text_encoder_config = LlavaConfig(vision_config, text_config, pad_token_id=100, image_token_index=101)

        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=8,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = LlavaForConditionalGeneration(llava_text_encoder_config)
        tokenizer = LlamaTokenizerFast.from_pretrained("finetrainers/dummy-hunyaunvideo", subfolder="tokenizer")

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModel(clip_text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        image_processor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
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
            "prompt_template": {
                "template": "{}",
                "crop_start": 0,
                "image_emb_len": 49,
                "image_emb_start": 5,
                "image_emb_end": 54,
                "double_return_token_id": 0,
            },
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": image_height,
            "width": image_width,
            "num_frames": 9,
            "max_sequence_length": 64,
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
        # NOTE: The expected video has 4 lesser frames because they are dropped in the pipeline
        self.assertEqual(generated_video.shape, (5, 3, 16, 16))

        # fmt: off
        expected_slice = torch.tensor([0.444, 0.479, 0.4485, 0.5752, 0.3539, 0.1548, 0.2706, 0.3593, 0.5323, 0.6635, 0.6795, 0.5255, 0.5091, 0.345, 0.4276, 0.4128])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(
            torch.allclose(generated_slice, expected_slice, atol=1e-3),
            "The generated video does not match the expected slice.",
        )

    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs

            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        output = pipe(**inputs)[0]

        # Test passing in a everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        output = pipe(**inputs)[0]

        def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
            is_last = i == (pipe.num_timesteps - 1)
            if is_last:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_inputs_change_tensor
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        output = pipe(**inputs)[0]
        assert output.abs().sum() < 1e10

    def test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing1 = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=2)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing2 = pipe(**inputs)[0]

        if test_max_difference:
            max_diff1 = np.abs(to_np(output_with_slicing1) - to_np(output_without_slicing)).max()
            max_diff2 = np.abs(to_np(output_with_slicing2) - to_np(output_without_slicing)).max()
            self.assertLess(
                max(max_diff1, max_diff2),
                expected_max_diff,
                "Attention slicing should not affect the inference results",
            )

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        # Seems to require higher tolerance than the other tests
        expected_diff_max = 0.6
        generator_device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        # Without tiling
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs(generator_device)
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        self.assertLess(
            (to_np(output_without_tiling) - to_np(output_with_tiling)).max(),
            expected_diff_max,
            "VAE tiling should not affect the inference results",
        )

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

    @unittest.skip(
        "Encode prompt currently does not work in isolation because of requiring image embeddings from image processor. The test does not handle this case, or we need to rewrite encode_prompt."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass
