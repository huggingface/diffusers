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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, GlmImagePipeline, GlmImageTransformer2DModel
from diffusers.utils import is_transformers_version

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    require_torch_accelerator,
    require_transformers_version_greater,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


if is_transformers_version(">=", "5.0.0.dev0"):
    from transformers import GlmImageConfig, GlmImageForConditionalGeneration, GlmImageProcessor


enable_full_determinism()


class GlmImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = GlmImagePipeline
    # GLM-Image does not expose `negative_prompt` — guidance is applied against unconditional embeddings.
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        glm_config = GlmImageConfig(
            text_config={
                "vocab_size": 168064,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "vision_vocab_size": 128,
                "rope_parameters": {"mrope_section": (4, 2, 2)},
            },
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "num_heads": 2,
                "image_size": 32,
                "patch_size": 8,
                "intermediate_size": 32,
            },
            vq_config={"embed_dim": 32, "num_embeddings": 128, "latent_channels": 32},
        )

        torch.manual_seed(0)
        vision_language_encoder = GlmImageForConditionalGeneration(glm_config)

        processor = GlmImageProcessor.from_pretrained("zai-org/GLM-Image", subfolder="processor")

        torch.manual_seed(0)
        # For GLM-Image, the relationship between components must satisfy:
        # patch_size × vae_scale_factor = 16 (since AR tokens are upsampled 2× from d32)
        transformer = GlmImageTransformer2DModel(
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            text_embed_dim=text_encoder.config.hidden_size,
            time_embed_dim=16,
            condition_dim=8,
            prior_vq_quantizer_codebook_size=128,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=(4, 8, 16, 16),
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=4,
            sample_size=128,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "tokenizer": tokenizer,
            "processor": processor,
            "text_encoder": text_encoder,
            "vision_language_encoder": vision_language_encoder,
            "vae": vae,
            "transformer": transformer,
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A photo of a cat",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


@require_transformers_version_greater("4.57.4")
@require_torch_accelerator
class TestGlmImagePipeline(GlmImagePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images[0]
        assert image.shape == self.output_shape

        generated_slice = image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])

        # fmt: off
        expected_slice = torch.tensor(
            [
                0.5849247, 0.50278825, 0.45747858, 0.45895284, 0.43804976, 0.47044256, 0.5239665, 0.47904694, 0.3323419, 0.38725388, 0.28505728, 0.3161863, 0.35026982, 0.37546024, 0.4090118, 0.46629113
            ]
        )
        # fmt: on

        assert_tensors_close(generated_slice, expected_slice, atol=1e-4, rtol=1e-4)

    def test_prompt_with_prior_token_ids(self):
        """Test that prompt and prior_token_ids can be provided together.

        When both are given, the AR generation step is skipped (prior_token_ids is used
        directly) and prompt is used to generate prompt_embeds via the glyph encoder.
        """
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()

        # Step 1: Run with prompt only to get prior_token_ids from AR model
        prior_token_ids, _, _ = pipe.generate_prior_tokens(
            prompt=inputs["prompt"],
            height=inputs["height"],
            width=inputs["width"],
            device=torch.device("cpu"),
            generator=self.get_generator(0),
        )

        # Step 2: Run with both prompt and prior_token_ids — should not raise
        images = pipe(**inputs, prior_token_ids=prior_token_ids).images
        assert len(images) == 1
        assert images[0].shape == self.output_shape

    def test_check_inputs_rejects_invalid_combinations(self):
        """Test that check_inputs correctly rejects invalid input combinations."""
        pipe = self.get_pipeline()
        height = width = 32

        # Neither prompt nor prior_token_ids → error
        with pytest.raises(ValueError):
            pipe.check_inputs(
                prompt=None,
                height=height,
                width=width,
                callback_on_step_end_tensor_inputs=None,
                prompt_embeds=torch.randn(1, 16, 32),
            )

        # prior_token_ids alone without prompt or prompt_embeds → error
        with pytest.raises(ValueError):
            pipe.check_inputs(
                prompt=None,
                height=height,
                width=width,
                callback_on_step_end_tensor_inputs=None,
                prior_token_ids=torch.randint(0, 100, (1, 64)),
            )

        # prompt + prompt_embeds together → error
        with pytest.raises(ValueError):
            pipe.check_inputs(
                prompt="A cat",
                height=height,
                width=width,
                callback_on_step_end_tensor_inputs=None,
                prompt_embeds=torch.randn(1, 16, 32),
            )

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-3):
        # GLM-Image runs an autoregressive prior before the diffusion loop; batching it accumulates slightly more
        # numerical drift than the default tolerance allows.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip(
        "`encode_prompt` cannot run in isolation: it goes through `check_inputs`, which requires either `prompt` "
        "or `prior_token_ids`, and the isolation harness supplies neither."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass


@require_transformers_version_greater("4.57.4")
@require_torch_accelerator
class TestGlmImagePipelineMemory(GlmImagePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the GLM-Image pipeline."""
