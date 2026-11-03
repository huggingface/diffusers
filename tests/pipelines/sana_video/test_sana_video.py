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

import gc

import pytest
import torch
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizer

from diffusers import AutoencoderKLWan, DPMSolverMultistepScheduler, SanaVideoPipeline, SanaVideoTransformer3DModel

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class SanaVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = SanaVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # Sana Video is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    output_shape = (9, 3, 32, 32)

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
        scheduler = DPMSolverMultistepScheduler()

        torch.manual_seed(0)
        text_encoder_config = Gemma2Config(
            head_dim=16,
            hidden_size=8,
            initializer_range=0.02,
            intermediate_size=64,
            max_position_embeddings=8192,
            model_type="gemma2",
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=2,
            vocab_size=8,
            attn_implementation="eager",
        )
        text_encoder = Gemma2Model(text_encoder_config)
        tokenizer = GemmaTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        torch.manual_seed(0)
        transformer = SanaVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            num_attention_heads=2,
            attention_head_dim=12,
            num_layers=2,
            num_cross_attention_heads=2,
            cross_attention_head_dim=12,
            cross_attention_dim=24,
            caption_channels=8,
            mlp_ratio=2.5,
            dropout=0.0,
            attention_bias=False,
            sample_size=8,
            patch_size=(1, 2, 2),
            norm_elementwise_affine=False,
            norm_eps=1e-6,
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
        return {
            "prompt": "",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "complex_human_instruction": [],
            "use_resolution_binning": False,
        }


class TestSanaVideoPipeline(SanaVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    # TODO(aryan): Create a dummy gemma model with smol vocab size
    @pytest.mark.skip(
        "A very small vocab size is used for fast tests. So, Any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(
        "A very small vocab size is used for fast tests. So, Any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_single_identical(self):
        pass

    def test_save_load_float16(self, tmp_path):
        # Requires higher tolerance as model seems very sensitive to dtype
        super().test_save_load_float16(tmp_path, expected_max_diff=0.2)


class TestSanaVideoPipelineMemory(SanaVideoPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Sana Video pipeline."""


@slow
@require_torch_accelerator
class TestSanaVideoPipelineIntegration:
    prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    @pytest.mark.skip("TODO: test needs to be implemented")
    def test_sana_video_480p(self):
        pass
