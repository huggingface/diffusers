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
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    LlamaConfig,
    LlamaModel,
    LlamaTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoFramepackPipeline,
    HunyuanVideoFramepackTransformer3DModel,
)

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    FasterCacheTesterMixin,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
)


enable_full_determinism()


class HunyuanVideoFramepackPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanVideoFramepackPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["image", "prompt"])
    output_shape = (13, 3, 32, 32)
    # Framepack is a video pipeline (`num_videos_per_prompt`) and takes `image_latents` rather than `latents`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "output_type", "return_dict"]
    )

    # `image_encoder` is a `SiglipVisionModel`, whose attention pooling head
    # (`SiglipMultiheadAttentionPoolingHead`) wraps a `torch.nn.MultiheadAttention`. That hands
    # `self.out_proj.weight` to `torch.nn.functional.multi_head_attention_forward` instead of calling
    # `self.out_proj`, so the leaf-level onload hook on `out_proj` never fires and its weights stay on the offload
    # device. Same root cause as the sequential CPU offloading skips below. Block-level offloading is unaffected
    # (the whole head is onloaded as one unmatched module), and every other component offloads fine at leaf level,
    # so exclude just this one rather than skipping the test.
    group_offloading_leaf_level_exclude_modules = ["image_encoder"]

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        torch.manual_seed(0)
        transformer = HunyuanVideoFramepackTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=10,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_refiner_layers=1,
            patch_size=2,
            patch_size_t=1,
            guidance_embeds=True,
            text_embed_dim=16,
            pooled_projection_dim=8,
            rope_axes_dim=(2, 4, 4),
            image_condition_type=None,
            has_image_proj=True,
            image_proj_dim=32,
            has_clean_x_embedder=True,
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

        feature_extractor = SiglipImageProcessor.from_pretrained(
            "hf-internal-testing/tiny-random-SiglipVisionModel", size={"height": 30, "width": 30}
        )
        image_encoder = SiglipVisionModel.from_pretrained("hf-internal-testing/tiny-random-SiglipVisionModel")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": feature_extractor,
            "image_encoder": image_encoder,
        }

    def get_dummy_inputs(self):
        image_height = 32
        image_width = 32
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
            "num_frames": 9,
            "latent_window_size": 3,
            "max_sequence_length": 256,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanVideoFramepackPipeline(HunyuanVideoFramepackPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.3628, 0.3380, 0.3421, 0.3505, 0.3362, 0.3268, 0.4167, 0.4063, 0.5225, 0.4693, 0.4827, 0.4583, 0.4144, 0.3983, 0.4089, 0.4587])
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

    def test_callback_inputs(self):
        # Framepack does not fit the shared version of this test: with `output_type="latent"` it returns the
        # accumulated history as a one-element list rather than a tensor, and that history keeps the leading
        # image-conditioning latent frame the callback never sees. So zeroing `latents` in the callback zeroes
        # every generated frame but not that first one — assert exactly that instead.
        pipe = self.get_pipeline().to(torch_device)
        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            for tensor_name in callback_kwargs:
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs

            for tensor_name in callback_kwargs:
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        pipe(**inputs)

        # Test passing in everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        pipe(**inputs)

        def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
            is_last = i == (pipe.num_timesteps - 1)
            if is_last:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_inputs_change_tensor
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        latents = pipe(**inputs)[0][0]
        # Frame 0 is the image-conditioning latent; the rest are what the denoising loop produced.
        assert latents[:, :, 1:].abs().sum() == 0

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


class TestHunyuanVideoFramepackPipelineMemory(HunyuanVideoFramepackPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Framepack pipeline."""

    @pytest.mark.skip("The image_encoder uses SiglipVisionModel, which does not support sequential CPU offloading.")
    def test_sequential_cpu_offload_forward_pass(self):
        # https://github.com/huggingface/transformers/blob/21cb353b7b4f77c6f5f5c3341d660f86ff416d04/src/transformers/models/siglip/modeling_siglip.py#L803
        # This is because it instantiates it's attention layer from torch.nn.MultiheadAttention, which calls to
        # `torch.nn.functional.multi_head_attention_forward` with the weights and bias. Since the hook is never
        # triggered with a forward pass call, the weights stay on the CPU. There are more examples where we skip
        # this test because of MHA (example: HunyuanDiT because of AttentionPooling layer).
        pass

    @pytest.mark.skip("The image_encoder uses SiglipVisionModel, which does not support sequential CPU offloading.")
    def test_sequential_offload_forward_pass_twice(self):
        # https://github.com/huggingface/transformers/blob/21cb353b7b4f77c6f5f5c3341d660f86ff416d04/src/transformers/models/siglip/modeling_siglip.py#L803
        # This is because it instantiates it's attention layer from torch.nn.MultiheadAttention, which calls to
        # `torch.nn.functional.multi_head_attention_forward` with the weights and bias. Since the hook is never
        # triggered with a forward pass call, the weights stay on the CPU. There are more examples where we skip
        # this test because of MHA (example: HunyuanDiT because of AttentionPooling layer).
        pass


class TestHunyuanVideoFramepackPipelinePyramidAttentionBroadcast(
    HunyuanVideoFramepackPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin
):
    """Pyramid Attention Broadcast cache tests for the Framepack pipeline."""


class TestHunyuanVideoFramepackPipelineFasterCache(HunyuanVideoFramepackPipelineTesterConfig, FasterCacheTesterMixin):
    """FasterCache tests for the Framepack pipeline."""

    # Framepack is guidance-distilled, so the FasterCache tester must skip the low/high-frequency-delta checks.
    FASTER_CACHE_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (-1, 901),
        "unconditional_batch_skip_range": 2,
        "attention_weight_callback": lambda _: 0.5,
        "is_guidance_distilled": True,
    }


class TestHunyuanVideoFramepackPipelineLoRA(HunyuanVideoFramepackPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Framepack pipeline."""


class TestHunyuanVideoFramepackPipelineLoRAMemory(HunyuanVideoFramepackPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the Framepack pipeline."""
