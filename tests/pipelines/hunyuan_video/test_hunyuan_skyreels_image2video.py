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
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanSkyreelsImageToVideoPipeline,
    HunyuanVideoTransformer3DModel,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
)


enable_full_determinism()


class HunyuanSkyreelsImageToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanSkyreelsImageToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "image"])
    output_shape = (9, 3, 16, 16)
    # HunyuanVideo is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = HunyuanVideoTransformer3DModel(
            in_channels=8,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=10,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_refiner_layers=1,
            patch_size=1,
            patch_size_t=1,
            guidance_embeds=True,
            text_embed_dim=16,
            pooled_projection_dim=8,
            rope_axes_dim=(2, 4, 4),
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

        llama_text_encoder_config = LlamaConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )
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
        text_encoder = LlamaModel(llama_text_encoder_config)
        tokenizer = LlamaTokenizer.from_pretrained("finetrainers/dummy-hunyaunvideo", subfolder="tokenizer")

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModel(clip_text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
        }

    def get_dummy_inputs(self):
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        return {
            "image": image,
            "prompt": "dance monkey",
            "prompt_template": {
                "template": "{}",
                "crop_start": 0,
            },
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": image_height,
            "width": image_width,
            # 4 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanSkyreelsImageToVideoPipeline(HunyuanSkyreelsImageToVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.5979, 0.5689, 0.5049, 0.4954, 0.4626, 0.5027, 0.4998, 0.5639, 0.5746, 0.5710, 0.5034, 0.5987, 0.6288, 0.5199, 0.5518, 0.5783])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(
            generated_slice, expected_slice, atol=1e-3, msg="The generated video does not match the expected slice."
        )

    def test_vae_tiling(self, expected_diff_max: float = 0.6):
        # Seems to require higher tolerance than the other tests
        pipe = self.get_pipeline().to(torch_device)

        # Without tiling
        output_without_tiling = self.run_pipe(pipe, height=128, width=128)

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        output_with_tiling = self.run_pipe(pipe, height=128, width=128)

        assert (output_without_tiling - output_with_tiling).abs().max() < expected_diff_max, (
            "VAE tiling should not affect the inference results."
        )

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


class TestHunyuanSkyreelsImageToVideoPipelineMemory(
    HunyuanSkyreelsImageToVideoPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Skyreels I2V pipeline."""


class TestHunyuanSkyreelsImageToVideoPipelinePyramidAttentionBroadcast(
    HunyuanSkyreelsImageToVideoPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin
):
    """Pyramid Attention Broadcast cache tests for the Skyreels I2V pipeline."""


class TestHunyuanSkyreelsImageToVideoPipelineLoRA(HunyuanSkyreelsImageToVideoPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Skyreels I2V pipeline."""


class TestHunyuanSkyreelsImageToVideoPipelineLoRAMemory(
    HunyuanSkyreelsImageToVideoPipelineTesterConfig, LoraMemoryTesterMixin
):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the Skyreels I2V pipeline."""
