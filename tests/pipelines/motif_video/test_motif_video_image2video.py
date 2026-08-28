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

import pytest
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionConfig,
    T5Gemma2Encoder,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, MotifVideoImage2VideoPipeline
from diffusers.guiders import AdaptiveProjectedGuidance
from diffusers.models.transformers.transformer_motif_video import MotifVideoTransformer3DModel

from ...testing_utils import enable_full_determinism
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class MotifVideoImage2VideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MotifVideoImage2VideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "height", "width", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "image"])
    output_shape = (9, 3, 16, 16)
    group_offloading_leaf_level_exclude_modules = ["text_encoder"]
    # MotifVideo is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

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

        # Build a tiny T5Gemma2Encoder to match the pipeline's expected text_encoder type
        text_config = T5Gemma2TextConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=1104,
            max_position_embeddings=128,
            head_dim=16,
            num_key_value_heads=2,
            dropout_rate=0.0,
        )

        vision_config = SiglipVisionConfig(
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            image_size=16,
            patch_size=4,
            num_channels=3,
        )

        encoder_config = T5Gemma2EncoderConfig(
            text_config=text_config,
            vision_config=vision_config,
        )
        text_encoder = T5Gemma2Encoder(encoder_config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        feature_extractor = SiglipImageProcessor(
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            size={"height": 16, "width": 16},
        )

        torch.manual_seed(0)
        transformer = MotifVideoTransformer3DModel(
            in_channels=33,
            out_channels=16,
            num_attention_heads=2,
            attention_head_dim=12,
            num_layers=1,
            num_single_layers=1,
            mlp_ratio=4.0,
            patch_size=1,
            patch_size_t=1,
            qk_norm="rms_norm",
            text_embed_dim=32,
            image_embed_dim=4,
            rope_axes_dim=(4, 4, 4),
        )

        guider = AdaptiveProjectedGuidance()

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "guider": guider,
        }

    def get_dummy_inputs(self):
        return {
            "image": Image.new("RGB", (16, 16)),
            "prompt": "A test video",
            "negative_prompt": "bad quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestMotifVideoImage2VideoPipeline(MotifVideoImage2VideoPipelineTesterConfig, PipelineTesterMixin):
    # The pipeline rejects a batched `image` ("`image` must be a single image, got a list of N images"), so the two
    # tests that batch every input in `batch_input_params` cannot pass.
    SINGLE_IMAGE_XFAIL = pytest.mark.xfail(
        condition=True,
        reason="MotifVideo I2V only supports a single conditioning image.",
        strict=False,
    )

    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    @SINGLE_IMAGE_XFAIL
    def test_inference_batch_consistent(self, batch_sizes=[2], batch_generator=True):
        super().test_inference_batch_consistent(batch_sizes=batch_sizes, batch_generator=batch_generator)

    @SINGLE_IMAGE_XFAIL
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    # The image conditioning goes through the text encoder's vision tower, so a pipeline holding only the text stack
    # cannot reproduce the full pipeline's output: the isolated run differs by ~7e-3.
    @pytest.mark.xfail(
        condition=True,
        reason="MotifVideo I2V conditions on the image through the text encoder's vision tower.",
        strict=False,
    )
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        super().test_encode_prompt_works_in_isolation(
            extra_required_param_value_dict=extra_required_param_value_dict, atol=atol, rtol=rtol
        )


class TestMotifVideoImage2VideoPipelineMemory(MotifVideoImage2VideoPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the MotifVideo I2V pipeline.

    The `text_encoder`'s vision tower cannot be offloaded at leaf level, so it is excluded from the pipeline-level
    group offloading test (see `group_offloading_leaf_level_exclude_modules`); the tests that offload every component
    unconditionally are expected failures below.
    """

    VISION_TOWER_OFFLOAD_XFAIL = pytest.mark.xfail(
        condition=True,
        reason="T5Gemma2Encoder's vision_tower doesn't support block-level or leaf-level offloading.",
        strict=False,
    )

    @VISION_TOWER_OFFLOAD_XFAIL
    def test_group_offloading_inference(self):
        super().test_group_offloading_inference()

    @VISION_TOWER_OFFLOAD_XFAIL
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=1e-4):
        super().test_sequential_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @VISION_TOWER_OFFLOAD_XFAIL
    def test_sequential_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        super().test_sequential_offload_forward_pass_twice(expected_max_diff=expected_max_diff)
