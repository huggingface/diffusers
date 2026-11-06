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
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers import (
    AutoencoderKLWan,
    Cosmos2_5_TransferPipeline,
    CosmosControlNetModel,
    CosmosTransformer3DModel,
    UniPCMultistepScheduler,
)

from ...testing_utils import enable_full_determinism
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin
from .cosmos_guardrail import DummyCosmosSafetyChecker
from .testing_utils import CosmosSafetyCheckerTesterMixin


enable_full_determinism()


class Cosmos2_5_TransferWrapper(Cosmos2_5_TransferPipeline):
    @staticmethod
    def from_pretrained(*args, **kwargs):
        if "safety_checker" not in kwargs or kwargs["safety_checker"] is None:
            safety_checker = DummyCosmosSafetyChecker()
            device_map = kwargs.get("device_map", "cpu")
            dtype = kwargs.get("dtype")
            if device_map is not None or dtype is not None:
                safety_checker = safety_checker.to(device_map, dtype=dtype)
            kwargs["safety_checker"] = safety_checker
        return Cosmos2_5_TransferPipeline.from_pretrained(*args, **kwargs)


class Cosmos2_5_TransferPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Cosmos2_5_TransferWrapper
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "controls"])
    output_shape = (3, 3, 32, 32)
    # Cosmos2.5 transfer is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        # Transformer with img_context support for Transfer2.5
        transformer = CosmosTransformer3DModel(
            in_channels=16 + 1,
            out_channels=16,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=4,
            max_size=(4, 32, 32),
            patch_size=(1, 2, 2),
            rope_scale=(2.0, 1.0, 1.0),
            concat_padding_mask=True,
            extra_pos_embed_type="learnable",
            controlnet_block_every_n=1,
            img_context_dim_in=32,
            img_context_num_tokens=4,
            img_context_dim_out=32,
        )

        torch.manual_seed(0)
        controlnet = CosmosControlNetModel(
            n_controlnet_blocks=2,
            in_channels=16 + 1 + 1,  # control latent channels + condition_mask + padding_mask
            latent_channels=16 + 1 + 1,  # base latent channels (16) + condition_mask (1) + padding_mask (1) = 18
            model_channels=32,
            num_attention_heads=2,
            attention_head_dim=16,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=4,
            patch_size=(1, 2, 2),
            max_size=(4, 32, 32),
            rope_scale=(2.0, 1.0, 1.0),
            extra_pos_embed_type="learnable",  # Match transformer's config
            img_context_dim_in=32,
            img_context_dim_out=32,
            use_crossattn_projection=False,  # Test doesn't need this projection
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = UniPCMultistepScheduler()

        torch.manual_seed(0)
        config = Qwen2_5_VLConfig(
            text_config={
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
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            hidden_size=16,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "controlnet": controlnet,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": DummyCosmosSafetyChecker(),
        }

    def get_dummy_inputs(self):
        controls_generator = torch.Generator(device="cpu").manual_seed(0)

        return {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "controls": [torch.randn(3, 32, 32, generator=controls_generator) for _ in range(5)],
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            "num_frames": 3,
            "num_frames_per_chunk": 16,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestCosmos2_5_TransferPipeline(
    Cosmos2_5_TransferPipelineTesterConfig, CosmosSafetyCheckerTesterMixin, PipelineTesterMixin
):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape
        assert torch.isfinite(generated_video).all()

    def test_inference_autoregressive_multi_chunk(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 5
        inputs["num_frames_per_chunk"] = 3
        inputs["num_ar_conditional_frames"] = 1

        video = pipe(**inputs).frames
        generated_video = video[0]

        assert generated_video.shape == (5, *self.output_shape[1:])
        assert torch.isfinite(generated_video).all()

    def test_inference_autoregressive_multi_chunk_no_condition_frames(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 5
        inputs["num_frames_per_chunk"] = 3
        inputs["num_ar_conditional_frames"] = 0

        video = pipe(**inputs).frames
        generated_video = video[0]

        assert generated_video.shape == (5, *self.output_shape[1:])
        assert torch.isfinite(generated_video).all()

    def test_num_frames_per_chunk_above_rope_raises(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_frames_per_chunk"] = 17

        with pytest.raises(ValueError, match="too large for RoPE setting"):
            pipe(**inputs)

    def test_inference_with_controls(self):
        """Test inference with control inputs (ControlNet)."""
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["controls"] = [torch.randn(3, 32, 32) for _ in range(5)]  # list of 5 frames (C, H, W)
        inputs["controls_conditioning_scale"] = 1.0
        inputs["num_frames"] = None

        video = pipe(**inputs).frames
        generated_video = video[0]

        assert generated_video.shape == (5, *self.output_shape[1:])
        assert torch.isfinite(generated_video).all()

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-2):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestCosmos2_5_TransferPipelineMemory(Cosmos2_5_TransferPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Cosmos2.5 transfer pipeline."""
