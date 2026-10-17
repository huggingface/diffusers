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


class HunyuanVideoImageToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanVideoImageToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "image"])
    # NOTE: The generated video has 4 fewer frames than requested because they are dropped in the pipeline.
    output_shape = (5, 3, 16, 16)
    # HunyuanVideo is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

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
        llava_text_encoder_config = LlavaConfig(
            vision_config=vision_config, text_config=text_config, pad_token_id=100, image_token_index=101
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

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "image_processor": image_processor,
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
                "image_emb_len": 49,
                "image_emb_start": 5,
                "image_emb_end": 54,
            },
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": image_height,
            "width": image_width,
            "num_frames": 9,
            "max_sequence_length": 64,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanVideoImageToVideoPipeline(HunyuanVideoImageToVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.4477, 0.4781, 0.4478, 0.5687, 0.3446, 0.1606, 0.2699, 0.3613, 0.5592, 0.6789, 0.6793, 0.5311, 0.5175, 0.3748, 0.4228, 0.4149])
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

    @pytest.mark.skip(
        "Encode prompt currently does not work in isolation because of requiring image embeddings from image processor. The test does not handle this case, or we need to rewrite encode_prompt."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestHunyuanVideoImageToVideoPipelineMemory(HunyuanVideoImageToVideoPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanVideo I2V pipeline."""


class TestHunyuanVideoImageToVideoPipelinePyramidAttentionBroadcast(
    HunyuanVideoImageToVideoPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin
):
    """Pyramid Attention Broadcast cache tests for the HunyuanVideo I2V pipeline."""


class TestHunyuanVideoImageToVideoPipelineLoRA(HunyuanVideoImageToVideoPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the HunyuanVideo I2V pipeline."""


class TestHunyuanVideoImageToVideoPipelineLoRAMemory(
    HunyuanVideoImageToVideoPipelineTesterConfig, LoraMemoryTesterMixin
):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the HunyuanVideo I2V pipeline."""
