# Copyright 2025 HuggingFace Inc.
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

from diffusers import WanAnimateTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    GGUFCompileTesterMixin,
    GGUFTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class WanAnimateTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return WanAnimateTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-wan-animate-transformer"

    @property
    def output_shape(self) -> tuple[int, ...]:
        # Output has fewer channels than input (4 vs 12)
        return (4, 21, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (12, 21, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool | float | dict]:
        # Use custom channel sizes since the default Wan Animate channel sizes will cause the motion encoder to
        # contain the vast majority of the parameters in the test model
        channel_sizes = {"4": 16, "8": 16, "16": 16}

        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 12,  # 2 * C + 4 = 2 * 4 + 4 = 12
            "latent_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "image_dim": 16,
            "rope_max_seq_len": 32,
            "motion_encoder_channel_sizes": channel_sizes,  # Start of Wan Animate-specific config
            "motion_encoder_size": 16,  # Ensures that there will be 2 motion encoder resblocks
            "motion_style_dim": 8,
            "motion_dim": 4,
            "motion_encoder_dim": 16,
            "face_encoder_hidden_dim": 16,
            "face_encoder_num_heads": 2,
            "inject_face_latents_blocks": 2,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 4
        num_frames = 20  # To make the shapes work out; for complicated reasons we want 21 to divide num_frames + 1
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        clip_seq_len = 12
        clip_dim = 16

        inference_segment_length = 77  # The inference segment length in the full Wan2.2-Animate-14B model
        face_height = 16  # Should be square and match `motion_encoder_size`
        face_width = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, 2 * num_channels + 4, num_frames + 1, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
            ),
            "encoder_hidden_states_image": randn_tensor(
                (batch_size, clip_seq_len, clip_dim),
                generator=self.generator,
                device=torch_device,
            ),
            "pose_hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "face_pixel_values": randn_tensor(
                (batch_size, 3, inference_segment_length, face_height, face_width),
                generator=self.generator,
                device=torch_device,
            ),
        }


class TestWanAnimateTransformer3D(WanAnimateTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Wan Animate Transformer 3D."""

    def test_output(self):
        # Override test_output because the transformer output is expected to have less channels
        # than the main transformer input.
        expected_output_shape = (1, 4, 21, 16, 16)
        super().test_output(expected_output_shape=expected_output_shape)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol (~1e-2) to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestWanAnimateTransformer3DMemory(WanAnimateTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Wan Animate Transformer 3D."""


class TestWanAnimateTransformer3DTraining(WanAnimateTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Wan Animate Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"WanAnimateTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestWanAnimateTransformer3DAttention(WanAnimateTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Wan Animate Transformer 3D."""


class TestWanAnimateTransformer3DCompile(WanAnimateTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Wan Animate Transformer 3D."""

    def test_torch_compile_recompilation_and_graph_break(self):
        # Skip: F.pad with mode="replicate" in WanAnimateFaceEncoder triggers importlib.import_module
        # internally, which dynamo doesn't support tracing through.
        pytest.skip("F.pad with replicate mode triggers unsupported import in torch.compile")


class TestWanAnimateTransformer3DBitsAndBytes(WanAnimateTransformer3DTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Wan Animate Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.float16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan Animate model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 21, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states_image": randn_tensor(
                (1, 257, 1280), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "pose_hidden_states": randn_tensor(
                (1, 16, 20, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "face_pixel_values": randn_tensor(
                (1, 3, 77, 512, 512), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanAnimateTransformer3DTorchAo(WanAnimateTransformer3DTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Wan Animate Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan Animate model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 21, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states_image": randn_tensor(
                (1, 257, 1280), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "pose_hidden_states": randn_tensor(
                (1, 16, 20, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "face_pixel_values": randn_tensor(
                (1, 3, 77, 512, 512), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanAnimateTransformer3DGGUF(WanAnimateTransformer3DTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Wan Animate Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/blob/main/Wan2.2-Animate-14B-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan Animate model dimensions.

        Wan 2.2 Animate: in_channels=36 (2*16+4), text_dim=4096, image_dim=1280
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 21, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states_image": randn_tensor(
                (1, 257, 1280), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "pose_hidden_states": randn_tensor(
                (1, 16, 20, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "face_pixel_values": randn_tensor(
                (1, 3, 77, 512, 512), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanAnimateTransformer3DGGUFCompile(WanAnimateTransformer3DTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Wan Animate Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/blob/main/Wan2.2-Animate-14B-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan Animate model dimensions.

        Wan 2.2 Animate: in_channels=36 (2*16+4), text_dim=4096, image_dim=1280
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 21, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states_image": randn_tensor(
                (1, 257, 1280), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "pose_hidden_states": randn_tensor(
                (1, 16, 20, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "face_pixel_values": randn_tensor(
                (1, 3, 77, 512, 512), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }
