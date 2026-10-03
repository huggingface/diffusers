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
from transformers import (
    AutoTokenizer,
    T5Gemma2Encoder,
    T5Gemma2EncoderConfig,
    T5Gemma2TextConfig,
)

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, MotifVideoPipeline
from diffusers.guiders import AdaptiveProjectedGuidance
from diffusers.models.transformers.transformer_motif_video import MotifVideoTransformer3DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class MotifVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MotifVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (9, 3, 16, 16)
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
        encoder_config = T5Gemma2EncoderConfig(text_config=text_config)
        text_encoder = T5Gemma2Encoder(encoder_config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            rope_axes_dim=(4, 4, 4),
        )

        guider = AdaptiveProjectedGuidance()

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "feature_extractor": None,
            "guider": guider,
        }

    def get_dummy_inputs(self):
        return {
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


class TestMotifVideoPipeline(MotifVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    # T5Gemma2Encoder rebuilds its non-persistent RoPE buffers on load, so the reloaded encoder computes the prompt
    # embeddings from fp16 buffers rather than the fp32-derived ones the in-memory pipeline half()-ed. The drift stays
    # within tolerance on CPU but measures ~0.1 on an accelerator.
    @pytest.mark.xfail(
        condition=torch_device != "cpu",
        reason="fp16 drift from the text encoder's rebuilt RoPE buffers exceeds the tolerance on accelerators.",
        strict=False,
    )
    def test_save_load_float16(self, tmp_path, expected_max_diff=5e-2):
        super().test_save_load_float16(tmp_path, expected_max_diff=expected_max_diff)


class TestMotifVideoPipelineMemory(MotifVideoPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the MotifVideo pipeline."""
