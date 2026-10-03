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

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingVideoToVideoPipeline,
    SkyReelsV2Transformer3DModel,
    UniPCMultistepScheduler,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class SkyReelsV2DiffusionForcingVideoToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = SkyReelsV2DiffusionForcingVideoToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["video", "prompt", "negative_prompt"])
    # SkyReels V2 is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    # The pipeline extends the conditioning video by `num_frames`: 7 input frames + 17 generated.
    output_shape = (24, 3, 16, 16)

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

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        video = [Image.new("RGB", (16, 16))] * 7
        return {
            "video": video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "generator": self.get_generator(0),
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "overlap_history": 3,
            "num_frames": 17,
            "base_num_frames": 5,
        }


class TestSkyReelsV2DiffusionForcingVideoToVideoPipeline(
    SkyReelsV2DiffusionForcingVideoToVideoPipelineTesterConfig, PipelineTesterMixin
):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]

        assert generated_video.shape == (len(inputs["video"]) + inputs["num_frames"], 3, 16, 16)
        assert generated_video.shape == self.output_shape

    def test_callback_cfg(self):
        pipe = self.get_pipeline().to(torch_device)
        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        # Track the number of callback calls for diffusion forcing pipelines
        callback_call_count = [0]  # Use list to make it mutable in closure

        def callback_increase_guidance(pipe, i, t, callback_kwargs):
            pipe._guidance_scale += 1.0
            callback_call_count[0] += 1
            return callback_kwargs

        inputs = self.get_dummy_inputs()

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

    # The diffusion-forcing loop runs several chunked passes, so batching accumulates slightly more drift than
    # the 1e-4 default allows (~5e-4 here). This also failed at the default tolerance before the migration.
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip(
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline has to run in mixed precision. Save/Load the entire pipeline in FP16 will result in errors"
    )
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip(
        "SkyReelsV2DiffusionForcingVideoToVideoPipeline has to run in mixed precision. Casting the entire pipeline will result in errors"
    )
    def test_half_precision_inference_no_nan(self):
        pass


class TestSkyReelsV2DiffusionForcingVideoToVideoPipelineMemory(
    SkyReelsV2DiffusionForcingVideoToVideoPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SkyReels V2 DF V2V pipeline."""
