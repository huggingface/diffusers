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

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)

from ...testing_utils import (
    Expectations,
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    require_peft_backend,
    require_torch_accelerator,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    FasterCacheTesterMixin,
    FirstBlockCacheTesterMixin,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
    TaylorSeerCacheTesterMixin,
)


enable_full_determinism()


class HunyuanVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (9, 3, 16, 16)
    # HunyuanVideo is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = HunyuanVideoTransformer3DModel(
            in_channels=4,
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
        return {
            "prompt": "dance monkey",
            "prompt_template": {
                "template": "{}",
                "crop_start": 0,
            },
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": 16,
            "width": 16,
            # 4 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanVideoPipeline(HunyuanVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.3966, 0.4693, 0.3223, 0.4634, 0.3316, 0.3698, 0.3201, 0.3954, 0.4430, 0.3860, 0.3925, 0.3823, 0.3478, 0.3901, 0.3837, 0.3547])
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
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

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


class TestHunyuanVideoPipelineMemory(HunyuanVideoPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanVideo pipeline."""


class TestHunyuanVideoPipelinePyramidAttentionBroadcast(
    HunyuanVideoPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin
):
    """Pyramid Attention Broadcast cache tests for the HunyuanVideo pipeline."""


class TestHunyuanVideoPipelineFasterCache(HunyuanVideoPipelineTesterConfig, FasterCacheTesterMixin):
    """FasterCache tests for the HunyuanVideo pipeline."""

    # HunyuanVideo is guidance-distilled, so the FasterCache tester must skip the low/high-frequency-delta checks.
    FASTER_CACHE_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (-1, 901),
        "unconditional_batch_skip_range": 2,
        "attention_weight_callback": lambda _: 0.5,
        "is_guidance_distilled": True,
    }


class TestHunyuanVideoPipelineFirstBlockCache(HunyuanVideoPipelineTesterConfig, FirstBlockCacheTesterMixin):
    """First Block Cache tests for the HunyuanVideo pipeline."""


class TestHunyuanVideoPipelineTaylorSeerCache(HunyuanVideoPipelineTesterConfig, TaylorSeerCacheTesterMixin):
    """TaylorSeer cache tests for the HunyuanVideo pipeline."""


class TestHunyuanVideoPipelineLoRA(HunyuanVideoPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the HunyuanVideo pipeline."""


class TestHunyuanVideoPipelineLoRAMemory(HunyuanVideoPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the HunyuanVideo pipeline."""


@nightly
@require_torch_accelerator
@require_peft_backend
@require_big_accelerator
class TestHunyuanVideoLoRAIntegration:
    """internal note: The integration slices were obtained on DGX.

    torch: 2.5.1+cu124 with CUDA 12.5. Need the same setup for the
    assertions to pass.
    """

    num_inference_steps = 10
    seed = 0
    repo_id = "hunyuanvideo-community/HunyuanVideo"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    @pytest.fixture
    def pipeline(self):
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            self.repo_id, subfolder="transformer", dtype=torch.bfloat16
        )
        return HunyuanVideoPipeline.from_pretrained(self.repo_id, transformer=transformer, dtype=torch.float16).to(
            torch_device
        )

    def test_original_format_cseti(self, pipeline):
        pipeline.load_lora_weights(
            "Cseti/HunyuanVideo-LoRA-Arcane_Jinx-v1", weight_name="csetiarcane-nfjinx-v1-6000.safetensors"
        )
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
        pipeline.vae.enable_tiling()

        prompt = "CSETIARCANE. A cat walks on the grass, realistic"

        out = pipeline(
            prompt=prompt,
            height=320,
            width=512,
            num_frames=9,
            num_inference_steps=self.num_inference_steps,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).frames[0]
        out = out.flatten()
        out_slice = np.concatenate((out[:8], out[-8:]))

        # fmt: off
        expected_slices = Expectations(
            {
                ("cuda", 7): np.array([0.1013, 0.1924, 0.0078, 0.1021, 0.1929, 0.0078, 0.1023, 0.1919, 0.7402, 0.104, 0.4482, 0.7354, 0.0925, 0.4382, 0.7275, 0.0815]),
                ("xpu", 3): np.array([0.1013, 0.1924, 0.0078, 0.1021, 0.1929, 0.0078, 0.1023, 0.1919, 0.7402, 0.104, 0.4482, 0.7354, 0.0925, 0.4382, 0.7275, 0.0815]),
            }
        )
        # fmt: on
        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3
