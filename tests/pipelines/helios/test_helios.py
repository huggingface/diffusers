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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, HeliosPipeline, HeliosScheduler, HeliosTransformer3DModel

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class HeliosPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HeliosPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (33, 3, 16, 16)
    # Helios is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
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
        scheduler = HeliosScheduler(stage_range=[0, 1], stages=1, use_dynamic_shifting=True)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = HeliosTransformer3DModel(
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
            rope_dim=(4, 4, 4),
            has_multi_term_memory_patch=True,
            guidance_cross_attn=True,
            zero_history_timestep=True,
            is_amplify_history=False,
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
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHeliosPipeline(HeliosPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.4529, 0.4527, 0.4499, 0.4542, 0.4528, 0.4524, 0.4531, 0.4534, 0.5328,
        0.5340, 0.5012, 0.5135, 0.5322, 0.5203, 0.5144, 0.5101])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    @pytest.mark.skip("Helios uses a lot of mixed precision internally, which is not suitable for this test case")
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip("Optional components not applicable for Helios")
    def test_save_load_optional_components(self):
        pass


class TestHeliosPipelineMemory(HeliosPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Helios pipeline."""


class TestHeliosPipelineLoRA(HeliosPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Helios pipeline."""


class TestHeliosPipelineLoRAMemory(HeliosPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the Helios pipeline."""


@slow
@require_torch_accelerator
class TestHeliosPipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    @pytest.mark.skip("TODO: test needs to be implemented")
    def test_helios(self):
        pass
