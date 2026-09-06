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

import pytest
import torch
from transformers import (
    AutoConfig,
    ByT5Tokenizer,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLHunyuanVideo15,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
)
from diffusers.guiders import ClassifierFreeGuidance

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class HunyuanVideo15PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanVideo15Pipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "prompt_embeds_2",
            "prompt_embeds_mask_2",
            "negative_prompt_embeds_2",
            "negative_prompt_embeds_mask_2",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (9, 3, 16, 16)
    # HunyuanVideo 1.5 is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        transformer = HunyuanVideo15Transformer3DModel(
            in_channels=9,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=num_layers,
            num_refiner_layers=1,
            mlp_ratio=2.0,
            patch_size=1,
            patch_size_t=1,
            text_embed_dim=16,
            text_embed_2_dim=32,
            image_embed_dim=12,
            rope_axes_dim=(2, 2, 4),
            target_size=16,
            task_type="t2v",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo15(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(16, 16),
            layers_per_block=1,
            spatial_compression_ratio=4,
            temporal_compression_ratio=2,
            downsample_match_channel=False,
            upsample_match_channel=False,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        torch.manual_seed(0)
        qwen_config = Qwen2_5_VLTextConfig(
            **{
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            }
        )
        text_encoder = Qwen2_5_VLTextModel(qwen_config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_2 = T5EncoderModel(config)
        tokenizer_2 = ByT5Tokenizer()

        guider = ClassifierFreeGuidance(guidance_scale=1.0)

        return {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder.eval(),
            "text_encoder_2": text_encoder_2.eval(),
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "guider": guider,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "monkey",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanVideo15Pipeline(HunyuanVideo15PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.4296, 0.5549, 0.3088, 0.9115, 0.5049, 0.7926, 0.5549, 0.8618, 0.5091, 0.5075, 0.7117, 0.5292, 0.7053, 0.4864, 0.5206, 0.3878])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    @pytest.mark.skip("TODO: Test not supported for now because needs to be adjusted to work with guiders.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @pytest.mark.skip("Needs to be revisited.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("Needs to be revisited.")
    def test_inference_batch_single_identical(self):
        pass


class TestHunyuanVideo15PipelineMemory(HunyuanVideo15PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanVideo 1.5 pipeline."""
