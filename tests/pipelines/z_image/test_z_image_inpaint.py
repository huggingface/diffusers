# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

import os
import random

import pytest
import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImageInpaintPipeline,
    ZImageTransformer2DModel,
)

from ...testing_utils import assert_tensors_close, floats_tensor, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


# Z-Image requires torch.use_deterministic_algorithms(False) due to complex64 RoPE operations
# Cannot use enable_full_determinism() which sets it to True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False

# Note: Some tests (test_half_precision_inference_no_nan, test_save_load_float16) may fail in full suite
# due to RopeEmbedder cache state pollution between tests. They pass when run individually.
# This is a known test isolation issue, not a functional bug.


def get_dummy_mask(height=32, width=32):
    """Mask marking the center region for inpainting (1 = inpaint, 0 = preserve)."""
    mask_image = torch.zeros((1, 1, height, width), device=torch_device)
    mask_image[:, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0
    return mask_image


class ZImageInpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = ZImageInpaintPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "image",
            "mask_image",
            "strength",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=16,
            dim=32,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=2,
            norm_eps=1e-5,
            qk_norm=True,
            cap_feat_dim=16,
            rope_theta=256.0,
            t_scale=1000.0,
            axes_dims=[8, 4, 4],
            axes_lens=[256, 32, 32],
        )
        # `x_pad_token` and `cap_pad_token` are initialized with `torch.empty`.
        # This can cause NaN data values in our testing environment. Fixating them
        # helps prevent that issue.
        with torch.no_grad():
            transformer.x_pad_token.copy_(torch.ones_like(transformer.x_pad_token.data))
            transformer.cap_pad_token.copy_(torch.ones_like(transformer.cap_pad_token.data))

        torch.manual_seed(0)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64],
            layers_per_block=1,
            latent_channels=16,
            norm_num_groups=32,
            sample_size=32,
            scaling_factor=0.3611,
            shift_factor=0.1159,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=512,
        )
        text_encoder = Qwen3Model(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

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
            "negative_prompt": "bad quality",
            "image": floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device),
            "mask_image": get_dummy_mask(),
            "strength": 1.0,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "cfg_normalization": False,
            "cfg_truncation": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestZImageInpaintPipeline(ZImageInpaintPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.5002, 0.5153, 0.5248, 0.5328, 0.5400, 0.5032, 0.4884, 0.4892, 0.4327, 0.4214, 0.5545, 0.6027, 0.4683, 0.4561, 0.4936, 0.4785])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=5e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-1):
        # Z-Image pads the batch to a common sequence length, so batched and single runs diverge slightly more.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_vae_tiling(self, expected_diff_max: float = 0.7):
        pipe = self.get_pipeline().to(torch_device)

        # Without tiling
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        # Generate a larger image and mask for the input
        inputs["image"] = floats_tensor((1, 3, 128, 128), rng=random.Random(0)).to(torch_device)
        inputs["mask_image"] = get_dummy_mask(128, 128)
        output_without_tiling = pipe(**inputs)[0]

        # With tiling (standard AutoencoderKL doesn't accept parameters)
        pipe.vae.enable_tiling()
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        inputs["image"] = floats_tensor((1, 3, 128, 128), rng=random.Random(0)).to(torch_device)
        inputs["mask_image"] = get_dummy_mask(128, 128)
        output_with_tiling = pipe(**inputs)[0]

        assert (output_without_tiling - output_with_tiling).abs().max() < expected_diff_max, (
            "VAE tiling should not affect the inference results."
        )

    def test_strength_parameter(self):
        """`strength` controls how much of the input image is kept, so it must change the output."""
        pipe = self.get_pipeline().to(torch_device)

        output_low = self.run_pipe(pipe, strength=0.2)
        output_high = self.run_pipe(pipe, strength=0.8)

        assert not torch.allclose(output_low, output_high, atol=1e-3)

    @pytest.mark.parametrize("strength", [-0.1, 1.5])
    def test_invalid_strength(self, strength):
        """`strength` outside of [0, 1] is rejected."""
        pipe = self.get_pipeline().to(torch_device)

        with pytest.raises(ValueError):
            self.run_pipe(pipe, strength=strength)

    def test_mask_inpainting(self):
        """The mask controls which regions get inpainted, so an all-ones and an all-zeros mask must differ."""
        pipe = self.get_pipeline().to(torch_device)

        # Inpaint everything vs. preserve everything.
        output_full = self.run_pipe(pipe, mask_image=torch.ones((1, 1, 32, 32), device=torch_device))
        output_none = self.run_pipe(pipe, mask_image=torch.zeros((1, 1, 32, 32), device=torch_device))

        assert not torch.allclose(output_full, output_none, atol=1e-3)


class TestZImageInpaintPipelineMemory(ZImageInpaintPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Z-Image inpaint pipeline."""

    def test_pipeline_with_accelerator_device_map(self, tmp_path, base_pipe_output, expected_max_difference=1e-3):
        # Z-Image RoPE embeddings (complex64) have slightly higher numerical tolerance and inpainting mask blending
        # adds additional numerical variance.
        super().test_pipeline_with_accelerator_device_map(
            tmp_path, base_pipe_output, expected_max_difference=expected_max_difference
        )
