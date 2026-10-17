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


import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXVideoToVideoPipeline, DDIMScheduler

from ...testing_utils import assert_tensors_close
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class CogVideoXVideoToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = CogVideoXVideoToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "video"])
    output_shape = (8, 3, 16, 16)
    # CogVideoX is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = CogVideoXTransformer3DModel(
            # Product of num_attention_heads * attention_head_dim must be divisible by 16 for 3D positional embeddings
            # But, since we are using tiny-random-t5 here, we need the internal dim of CogVideoXTransformer3DModel
            # to be 32. The internal dim is product of num_attention_heads and attention_head_dim
            num_attention_heads=4,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            time_embed_dim=2,
            text_embed_dim=32,  # Must match with tiny-random-t5
            num_layers=1,
            sample_width=2,  # latent width: 2 -> final width: 16
            sample_height=2,  # latent height: 2 -> final height: 16
            sample_frames=9,  # latent frames: (9 - 1) / 4 + 1 = 3 -> final frames: 9
            patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=16,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLCogVideoX(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ),
            up_block_types=(
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=2,
            temporal_compression_ratio=4,
        )

        torch.manual_seed(0)
        scheduler = DDIMScheduler()
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self):
        video_height = 16
        video_width = 16
        video = [Image.new("RGB", (video_width, video_height))] * 8

        inputs = {
            "video": video,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "strength": 0.5,
            "guidance_scale": 6.0,
            # Cannot reduce because convolution kernel becomes bigger than sample
            "height": video_height,
            "width": video_width,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs


class TestCogVideoXVideoToVideoPipeline(CogVideoXVideoToVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.5644, 0.6029, 0.6017, 0.5937, 0.5991, 0.5907, 0.6141, 0.5340, 0.3184, 0.4219, 0.4406, 0.4330, 0.4692, 0.4547, 0.4562, 0.5092])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-3)

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        # Since VideoToVideo uses both encoder and decoder tiling, there seems to be much more numerical
        # difference. We seem to need a higher tolerance here...
        # TODO(aryan): Look into this more deeply
        expected_diff_max = 0.4

        pipe = self.get_pipeline()

        # Without tiling
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_overlap_factor_height=1 / 12,
            tile_overlap_factor_width=1 / 12,
        )
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        assert (output_without_tiling - output_with_tiling).max() < expected_diff_max, (
            "VAE tiling should not affect the inference results"
        )


class TestCogVideoXVideoToVideoPipelineMemory(CogVideoXVideoToVideoPipelineTesterConfig, MemoryTesterMixin):
    pass
