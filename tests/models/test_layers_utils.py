# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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


import unittest

import numpy as np
import torch
from torch import nn

from diffusers.models.attention import GEGLU, AdaLayerNorm, ApproximateGELU
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.utils import torch_device


class EmbeddingsTests(unittest.TestCase):
    def test_timestep_embeddings(self):
        embedding_dim = 256
        timesteps = torch.arange(16)

        t1 = get_timestep_embedding(timesteps, embedding_dim)

        # first vector should always be composed only of 0's and 1's
        assert (t1[0, : embedding_dim // 2] - 0).abs().sum() < 1e-5
        assert (t1[0, embedding_dim // 2 :] - 1).abs().sum() < 1e-5

        # last element of each vector should be one
        assert (t1[:, -1] - 1).abs().sum() < 1e-5

        # For large embeddings (e.g. 128) the frequency of every vector is higher
        # than the previous one which means that the gradients of later vectors are
        # ALWAYS higher than the previous ones
        grad_mean = np.abs(np.gradient(t1, axis=-1)).mean(axis=1)

        prev_grad = 0.0
        for grad in grad_mean:
            assert grad > prev_grad
            prev_grad = grad

    def test_timestep_defaults(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim)
        t2 = get_timestep_embedding(
            timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, max_period=10_000
        )

        assert torch.allclose(t1.cpu(), t2.cpu(), 1e-3)

    def test_timestep_flip_sin_cos(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
        t1 = torch.cat([t1[:, embedding_dim // 2 :], t1[:, : embedding_dim // 2]], dim=-1)

        t2 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)

        assert torch.allclose(t1.cpu(), t2.cpu(), 1e-3)

    def test_timestep_downscale_freq_shift(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0)
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1)

        # get cosine half (vectors that are wrapped into cosine)
        cosine_half = (t1 - t2)[:, embedding_dim // 2 :]

        # cosine needs to be negative
        assert (np.abs((cosine_half <= 0).numpy()) - 1).sum() < 1e-5

    def test_sinoid_embeddings_hardcoded(self):
        embedding_dim = 64
        timesteps = torch.arange(128)

        # standard unet, score_vde
        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1, flip_sin_to_cos=False)
        # glide, ldm
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0, flip_sin_to_cos=True)
        # grad-tts
        t3 = get_timestep_embedding(timesteps, embedding_dim, scale=1000)

        assert torch.allclose(
            t1[23:26, 47:50].flatten().cpu(),
            torch.tensor([0.9646, 0.9804, 0.9892, 0.9615, 0.9787, 0.9882, 0.9582, 0.9769, 0.9872]),
            1e-3,
        )
        assert torch.allclose(
            t2[23:26, 47:50].flatten().cpu(),
            torch.tensor([0.3019, 0.2280, 0.1716, 0.3146, 0.2377, 0.1790, 0.3272, 0.2474, 0.1864]),
            1e-3,
        )
        assert torch.allclose(
            t3[23:26, 47:50].flatten().cpu(),
            torch.tensor([-0.9801, -0.9464, -0.9349, -0.3952, 0.8887, -0.9709, 0.5299, -0.2853, -0.9927]),
            1e-3,
        )


class Upsample2DBlockTests(unittest.TestCase):
    def test_upsample_default(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=False)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.2173, -1.2079, -1.2079, 0.2952, 1.1254, 1.1254, 0.2952, 1.1254, 1.1254])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_conv(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=True)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([0.7145, 1.3773, 0.3492, 0.8448, 1.0839, -0.3341, 0.5956, 0.1250, -0.4841])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_conv_out_dim(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=True, out_channels=64)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 64, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([0.2703, 0.1656, -0.2538, -0.0553, -0.2984, 0.1044, 0.1155, 0.2579, 0.7755])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_transpose(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=False, use_conv_transpose=True)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.3028, -0.1582, 0.0071, 0.0350, -0.4799, -0.1139, 0.1056, -0.1153, -0.1046])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class Downsample2DBlockTests(unittest.TestCase):
    def test_downsample_default(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=False)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.0513, -0.3889, 0.0640, 0.0836, -0.5460, -0.0341, -0.0169, -0.6967, 0.1179])
        max_diff = (output_slice.flatten() - expected_slice).abs().sum().item()
        assert max_diff <= 1e-3
        # assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-1)

    def test_downsample_with_conv(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=True)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [0.9267, 0.5878, 0.3337, 1.2321, -0.1191, -0.3984, -0.7532, -0.0715, -0.3913],
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_downsample_with_conv_pad1(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=True, padding=1)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([0.9267, 0.5878, 0.3337, 1.2321, -0.1191, -0.3984, -0.7532, -0.0715, -0.3913])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_downsample_with_conv_out_dim(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=True, out_channels=16)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 16, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.6586, 0.5985, 0.0721, 0.1256, -0.1492, 0.4436, -0.2544, 0.5021, 1.1522])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class ResnetBlock2DTests(unittest.TestCase):
    def test_resnet_default(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        temb = torch.randn(1, 128).to(torch_device)
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128).to(torch_device)
        with torch.no_grad():
            output_tensor = resnet_block(sample, temb)

        assert output_tensor.shape == (1, 32, 64, 64)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [-1.9010, -0.2974, -0.8245, -1.3533, 0.8742, -0.9645, -2.0584, 1.3387, -0.4746], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_restnet_with_use_in_shortcut(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        temb = torch.randn(1, 128).to(torch_device)
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, use_in_shortcut=True).to(torch_device)
        with torch.no_grad():
            output_tensor = resnet_block(sample, temb)

        assert output_tensor.shape == (1, 32, 64, 64)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [0.2226, -1.0791, -0.1629, 0.3659, -0.2889, -1.2376, 0.0582, 0.9206, 0.0044], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_resnet_up(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        temb = torch.randn(1, 128).to(torch_device)
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, up=True).to(torch_device)
        with torch.no_grad():
            output_tensor = resnet_block(sample, temb)

        assert output_tensor.shape == (1, 32, 128, 128)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [1.2130, -0.8753, -0.9027, 1.5783, -0.5362, -0.5001, 1.0726, -0.7732, -0.4182], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_resnet_down(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        temb = torch.randn(1, 128).to(torch_device)
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, down=True).to(torch_device)
        with torch.no_grad():
            output_tensor = resnet_block(sample, temb)

        assert output_tensor.shape == (1, 32, 32, 32)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [-0.3002, -0.7135, 0.1359, 0.0561, -0.7935, 0.0113, -0.1766, -0.6714, -0.0436], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_restnet_with_kernel_fir(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        temb = torch.randn(1, 128).to(torch_device)
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, kernel="fir", down=True).to(torch_device)
        with torch.no_grad():
            output_tensor = resnet_block(sample, temb)

        assert output_tensor.shape == (1, 32, 32, 32)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [-0.0934, -0.5729, 0.0909, -0.2710, -0.5044, 0.0243, -0.0665, -0.5267, -0.3136], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_restnet_with_kernel_sde_vp(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        temb = torch.randn(1, 128).to(torch_device)
        resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, kernel="sde_vp", down=True).to(torch_device)
        with torch.no_grad():
            output_tensor = resnet_block(sample, temb)

        assert output_tensor.shape == (1, 32, 32, 32)
        output_slice = output_tensor[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [-0.3002, -0.7135, 0.1359, 0.0561, -0.7935, 0.0113, -0.1766, -0.6714, -0.0436], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class Transformer2DModelTests(unittest.TestCase):
    def test_spatial_transformer_default(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        spatial_transformer_block = Transformer2DModel(
            in_channels=32,
            num_attention_heads=1,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=None,
        ).to(torch_device)
        with torch.no_grad():
            attention_scores = spatial_transformer_block(sample).sample

        assert attention_scores.shape == (1, 32, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-1.9455, -0.0066, -1.3933, -1.5878, 0.5325, -0.6486, -1.8648, 0.7515, -0.9689], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_cross_attention_dim(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 64, 64, 64).to(torch_device)
        spatial_transformer_block = Transformer2DModel(
            in_channels=64,
            num_attention_heads=2,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=64,
        ).to(torch_device)
        with torch.no_grad():
            context = torch.randn(1, 4, 64).to(torch_device)
            attention_scores = spatial_transformer_block(sample, context).sample

        assert attention_scores.shape == (1, 64, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]
        expected_slice = torch.tensor(
            [0.0143, -0.6909, -2.1547, -1.8893, 1.4097, 0.1359, -0.2521, -1.3359, 0.2598], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_timestep(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        num_embeds_ada_norm = 5

        sample = torch.randn(1, 64, 64, 64).to(torch_device)
        spatial_transformer_block = Transformer2DModel(
            in_channels=64,
            num_attention_heads=2,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=64,
            num_embeds_ada_norm=num_embeds_ada_norm,
        ).to(torch_device)
        with torch.no_grad():
            timestep_1 = torch.tensor(1, dtype=torch.long).to(torch_device)
            timestep_2 = torch.tensor(2, dtype=torch.long).to(torch_device)
            attention_scores_1 = spatial_transformer_block(sample, timestep=timestep_1).sample
            attention_scores_2 = spatial_transformer_block(sample, timestep=timestep_2).sample

        assert attention_scores_1.shape == (1, 64, 64, 64)
        assert attention_scores_2.shape == (1, 64, 64, 64)

        output_slice_1 = attention_scores_1[0, -1, -3:, -3:]
        output_slice_2 = attention_scores_2[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-0.3923, -1.0923, -1.7144, -1.5570, 1.4154, 0.1738, -0.1157, -1.2998, -0.1703], device=torch_device
        )
        expected_slice_2 = torch.tensor(
            [-0.4311, -1.1376, -1.7732, -1.5997, 1.3450, 0.0964, -0.1569, -1.3590, -0.2348], device=torch_device
        )

        assert torch.allclose(output_slice_1.flatten(), expected_slice, atol=1e-3)
        assert torch.allclose(output_slice_2.flatten(), expected_slice_2, atol=1e-3)

    def test_spatial_transformer_dropout(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        spatial_transformer_block = (
            Transformer2DModel(
                in_channels=32,
                num_attention_heads=2,
                attention_head_dim=16,
                dropout=0.3,
                cross_attention_dim=None,
            )
            .to(torch_device)
            .eval()
        )
        with torch.no_grad():
            attention_scores = spatial_transformer_block(sample).sample

        assert attention_scores.shape == (1, 32, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-1.9380, -0.0083, -1.3771, -1.5819, 0.5209, -0.6441, -1.8545, 0.7563, -0.9615], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    @unittest.skipIf(torch_device == "mps", "MPS does not support float64")
    def test_spatial_transformer_discrete(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        num_embed = 5

        sample = torch.randint(0, num_embed, (1, 32)).to(torch_device)
        spatial_transformer_block = (
            Transformer2DModel(
                num_attention_heads=1,
                attention_head_dim=32,
                num_vector_embeds=num_embed,
                sample_size=16,
            )
            .to(torch_device)
            .eval()
        )

        with torch.no_grad():
            attention_scores = spatial_transformer_block(sample).sample

        assert attention_scores.shape == (1, num_embed - 1, 32)

        output_slice = attention_scores[0, -2:, -3:]

        expected_slice = torch.tensor([-1.7648, -1.0241, -2.0985, -1.8035, -1.6404, -1.2098], device=torch_device)
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_default_norm_layers(self):
        spatial_transformer_block = Transformer2DModel(num_attention_heads=1, attention_head_dim=32, in_channels=32)

        assert spatial_transformer_block.transformer_blocks[0].norm1.__class__ == nn.LayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm3.__class__ == nn.LayerNorm

    def test_spatial_transformer_ada_norm_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            in_channels=32,
            num_embeds_ada_norm=5,
        )

        assert spatial_transformer_block.transformer_blocks[0].norm1.__class__ == AdaLayerNorm
        assert spatial_transformer_block.transformer_blocks[0].norm3.__class__ == nn.LayerNorm

    def test_spatial_transformer_default_ff_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            in_channels=32,
        )

        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].__class__ == GEGLU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[1].__class__ == nn.Dropout
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].__class__ == nn.Linear

        dim = 32
        inner_dim = 128

        # First dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.in_features == dim
        # NOTE: inner_dim * 2 because GEGLU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.out_features == inner_dim * 2

        # Second dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].in_features == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].out_features == dim

    def test_spatial_transformer_geglu_approx_ff_layers(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1,
            attention_head_dim=32,
            in_channels=32,
            activation_fn="geglu-approximate",
        )

        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].__class__ == ApproximateGELU
        assert spatial_transformer_block.transformer_blocks[0].ff.net[1].__class__ == nn.Dropout
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].__class__ == nn.Linear

        dim = 32
        inner_dim = 128

        # First dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.in_features == dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[0].proj.out_features == inner_dim

        # Second dimension change
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].in_features == inner_dim
        assert spatial_transformer_block.transformer_blocks[0].ff.net[2].out_features == dim

    def test_spatial_transformer_attention_bias(self):
        spatial_transformer_block = Transformer2DModel(
            num_attention_heads=1, attention_head_dim=32, in_channels=32, attention_bias=True
        )

        assert spatial_transformer_block.transformer_blocks[0].attn1.to_q.bias is not None
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_k.bias is not None
        assert spatial_transformer_block.transformer_blocks[0].attn1.to_v.bias is not None
