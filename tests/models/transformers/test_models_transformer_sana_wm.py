# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers import SanaWMTransformer3DModel
from diffusers.models.transformers.transformer_sana_wm import (
    torch_chunk_cam_single_path_delta_rule,
    torch_chunk_sana_gdn,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class SanaWMTransformer3DTesterConfig(BaseModelTesterConfig):
    # Tiny stand-in for the public `Efficient-Large-Model/SANA-WM_bidirectional` release
    # (depth 20 / hidden 2240 / 20 heads). `num_layers=2` together with `softmax_every_n=2`
    # keeps both camera-branch variants covered: block 0 is the GDN one and block 1 is the
    # softmax one the model's per-layer `attn_cls` loop selects.
    num_layers = 2
    in_channels = 4
    caption_channels = 8
    chunk_plucker_channels = 8
    sequence_length = 16

    num_frames = 4
    height = 8
    width = 8

    @property
    def model_class(self):
        return SanaWMTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (self.in_channels, self.num_frames, self.height, self.width)

    @property
    def output_shape(self) -> tuple:
        return (self.in_channels, self.num_frames, self.height, self.width)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "num_layers": self.num_layers,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "patch_size": (1, 1, 1),
            "softmax_every_n": 2,
            "linear_head_dim": 16,
            "t_kernel_size": 3,
            "conv_kernel_size": 4,
            "caption_channels": self.caption_channels,
            "mlp_ratio": 2.0,
            "chunk_plucker_channels": self.chunk_plucker_channels,
            "chunk_plucker_post_attn_blocks": self.num_layers,
        }

    def get_dummy_camera_conditions(self, batch_size: int = 1) -> torch.Tensor:
        """Build the `(B, F, 20)` camera conditioning: a flat 4x4 c2w followed by `[fx, fy, cx, cy]`.

        The trajectory is a pure forward translation with an identity rotation, and the intrinsics are a
        pinhole camera centred on the latent grid, which keeps the UCPE ray maps well conditioned.
        """
        c2w = torch.eye(4).repeat(batch_size, self.num_frames, 1, 1)
        c2w[..., 2, 3] = torch.arange(self.num_frames, dtype=torch.float32) * 0.1
        intrinsics = torch.tensor([float(self.width), float(self.height), self.width / 2, self.height / 2])
        intrinsics = intrinsics.expand(batch_size, self.num_frames, 4)
        return torch.cat([c2w.flatten(start_dim=-2), intrinsics], dim=-1).to(torch_device)

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        shape = (batch_size, self.in_channels, self.num_frames, self.height, self.width)
        plucker_shape = (batch_size, self.chunk_plucker_channels, self.num_frames, self.height, self.width)

        return {
            "hidden_states": randn_tensor(shape, generator=self.generator, device=torch_device),
            "timestep": torch.randint(0, 1000, size=(batch_size, 1, self.num_frames), generator=self.generator).to(
                device=torch_device, dtype=torch.float32
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, 1, self.sequence_length, self.caption_channels),
                generator=self.generator,
                device=torch_device,
            ),
            # SANA-WM's cross-attention needs the text padding mask to build its attention bias.
            "encoder_attention_mask": torch.ones(
                batch_size, self.sequence_length, dtype=torch.long, device=torch_device
            ),
            "camera_conditions": self.get_dummy_camera_conditions(batch_size),
            "chunk_plucker": randn_tensor(plucker_shape, generator=self.generator, device=torch_device),
        }


class TestSanaWMTransformer3D(SanaWMTransformer3DTesterConfig, ModelTesterMixin):
    def test_dtype_cast_keeps_complex_rotary_frequencies(self):
        # The RoPE table is complex. A registered buffer would be cast to the *real* dtype by `model.to(dtype)`,
        # silently dropping the sine half of every rotation, so a same-dtype round trip must be a no-op.
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        with torch.no_grad():
            reference = model(**inputs, return_dict=False)[0]
            model.to(torch.float32)
            assert model.rope.freqs.is_complex()
            output = model(**inputs, return_dict=False)[0]
        assert torch.equal(reference, output)


class TestSanaWMTransformer3DMemory(SanaWMTransformer3DTesterConfig, MemoryTesterMixin):
    pass


class TestSanaWMTransformer3DCompile(SanaWMTransformer3DTesterConfig, TorchCompileTesterMixin):
    def test_torch_compile_repeated_blocks(self):
        # The repeated `SanaVideoMSCamCtrlBlock` runs two attention variants (`softmax_every_n` swaps a
        # softmax block in for the GDN one), so the shared block forward compiles once per variant.
        super().test_torch_compile_repeated_blocks(recompile_limit=2)


class TestSanaWMTransformer3DTraining(SanaWMTransformer3DTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SanaWMTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


def _flip_and_shift(x, shift_val):
    # Reverse the frame axis, drop the last entry and pad the front with `shift_val`.
    x_shifted = torch.flip(x, dims=[2]).narrow(2, 0, x.shape[2] - 1)
    pad_shape = list(x.shape)
    pad_shape[2] = 1
    padding = torch.full(pad_shape, shift_val, device=x.device, dtype=x.dtype)
    return torch.cat([padding, x_shifted], dim=2)


class TestSanaWMBidirectionalScan:
    """The bidirectional scans must match a second scan on flipped and shifted inputs bit for bit."""

    batch_size, num_heads, head_dim, spatial_size = 2, 2, 4, 6
    token_inputs = ("q", "k", "v", "q_rot", "k_rot")

    def get_leaves(self, num_frames, requires_grad=False):
        generator = torch.Generator("cpu").manual_seed(0)
        B, H, D, S, T = self.batch_size, self.num_heads, self.head_dim, self.spatial_size, num_frames
        leaves = {name: torch.randn(B, T * S, H, D, generator=generator) for name in self.token_inputs}
        leaves["beta"] = torch.rand(B, H, T, S, generator=generator)
        leaves["decay"] = torch.rand(B, H, T, generator=generator)
        return {name: x.to(torch_device).requires_grad_(requires_grad) for name, x in leaves.items()}

    def get_inputs(self, leaves):
        # Same layout as the attention modules pass in: `(B, N, H, D)` tokens permuted to `(B, H, D, N)`.
        return {name: x.permute(0, 2, 3, 1) if name in self.token_inputs else x for name, x in leaves.items()}

    def to_frames(self, x, num_frames):
        B, H, D, N = x.shape
        return x.view(B, H, D, num_frames, N // num_frames).permute(0, 1, 3, 2, 4)

    def from_frames(self, x):
        B, H, T, D, S = x.shape
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, T * S)

    def flip_back(self, x, num_frames):
        B, H, D, N = x.shape
        return torch.flip(x.view(B, H, D, num_frames, N // num_frames), dims=[3]).reshape(B, H, D, N)

    def reference_gdn(self, q, k, v, q_rot, k_rot, beta, decay, chunk_size):
        T = beta.shape[2]
        num_fwd, den_fwd = torch_chunk_sana_gdn(
            q, k, v, q_rot, k_rot, beta, decay, chunk_size=chunk_size, return_components=True
        )
        num_bwd, den_bwd = torch_chunk_sana_gdn(
            self.from_frames(torch.flip(self.to_frames(q, T), dims=[2])),
            self.from_frames(_flip_and_shift(self.to_frames(k, T), 0.0)),
            self.from_frames(_flip_and_shift(self.to_frames(v, T), 0.0)),
            self.from_frames(torch.flip(self.to_frames(q_rot, T), dims=[2])),
            self.from_frames(_flip_and_shift(self.to_frames(k_rot, T), 0.0)),
            _flip_and_shift(beta, 0.0),
            _flip_and_shift(decay, 1.0),
            chunk_size=chunk_size,
            return_components=True,
        )
        return num_fwd + self.flip_back(num_bwd, T), den_fwd + self.flip_back(den_bwd, T)

    def reference_cam(self, q_rot, k_rot, v, beta, decay, chunk_size):
        T = beta.shape[2]
        out_fwd = torch_chunk_cam_single_path_delta_rule(q_rot, k_rot, v, beta, decay, chunk_size=chunk_size)
        out_bwd = torch_chunk_cam_single_path_delta_rule(
            self.from_frames(torch.flip(self.to_frames(q_rot, T), dims=[2])),
            self.from_frames(_flip_and_shift(self.to_frames(k_rot, T), 0.0)),
            self.from_frames(_flip_and_shift(self.to_frames(v, T), 0.0)),
            _flip_and_shift(beta, 0.0),
            _flip_and_shift(decay, 1.0),
            chunk_size=chunk_size,
        )
        return out_fwd + self.flip_back(out_bwd, T)

    @pytest.mark.parametrize("num_frames, chunk_size", [(1, 21), (2, 21), (5, 21), (5, 2), (7, 3)])
    def test_gdn_matches_flipped_scan(self, num_frames, chunk_size):
        inputs = self.get_inputs(self.get_leaves(num_frames))
        with torch.no_grad():
            expected = self.reference_gdn(**inputs, chunk_size=chunk_size)
            output = torch_chunk_sana_gdn(**inputs, chunk_size=chunk_size, return_components=True, bidirectional=True)
        assert torch.equal(output[0], expected[0])
        assert torch.equal(output[1], expected[1])

    @pytest.mark.parametrize("num_frames, chunk_size", [(1, 21), (2, 21), (5, 21), (5, 2), (7, 3)])
    def test_cam_matches_flipped_scan(self, num_frames, chunk_size):
        leaves = self.get_leaves(num_frames)
        del leaves["q"], leaves["k"]
        inputs = self.get_inputs(leaves)
        with torch.no_grad():
            expected = self.reference_cam(**inputs, chunk_size=chunk_size)
            output = torch_chunk_cam_single_path_delta_rule(**inputs, chunk_size=chunk_size, bidirectional=True)
        assert torch.equal(output, expected)

    def test_gdn_gradients_match_flipped_scan(self):
        leaves = self.get_leaves(num_frames=5, requires_grad=True)
        inputs = self.get_inputs(leaves)
        expected = self.reference_gdn(**inputs, chunk_size=2)
        expected_grads = torch.autograd.grad(sum(x.square().sum() for x in expected), list(leaves.values()))
        output = torch_chunk_sana_gdn(**inputs, chunk_size=2, return_components=True, bidirectional=True)
        grads = torch.autograd.grad(sum(x.square().sum() for x in output), list(leaves.values()))
        assert torch.equal(output[0], expected[0])
        assert torch.equal(output[1], expected[1])
        # The gradients of both scans sum the same terms in a different order.
        for grad, expected_grad in zip(grads, expected_grads):
            assert (grad - expected_grad).abs().max() <= 1e-5 * expected_grad.abs().max()

    def test_cam_gradients_match_flipped_scan(self):
        leaves = self.get_leaves(num_frames=5, requires_grad=True)
        del leaves["q"], leaves["k"]
        inputs = self.get_inputs(leaves)
        expected = self.reference_cam(**inputs, chunk_size=2)
        expected_grads = torch.autograd.grad(expected.square().sum(), list(leaves.values()))
        output = torch_chunk_cam_single_path_delta_rule(**inputs, chunk_size=2, bidirectional=True)
        grads = torch.autograd.grad(output.square().sum(), list(leaves.values()))
        assert torch.equal(output, expected)
        # The gradients of both scans sum the same terms in a different order.
        for grad, expected_grad in zip(grads, expected_grads):
            assert (grad - expected_grad).abs().max() <= 1e-5 * expected_grad.abs().max()
