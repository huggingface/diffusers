# Copyright 2025 SandAI and The HuggingFace Team. All rights reserved.
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

from diffusers import AutoencoderKLMagi
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, require_torch_multi_accelerator, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class AutoencoderKLMagiTesterConfig(BaseModelTesterConfig):
    main_input_name = "sample"

    @property
    def model_class(self):
        return AutoencoderKLMagi

    @property
    def pretrained_model_name_or_path(self):
        return None

    @property
    def pretrained_model_kwargs(self):
        return {}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "latent_channels": 4,
            "embed_dim": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "mlp_ratio": 2,
            "patch_size": 2,
            "patch_length": 4,
            "sample_size": 8,
            "sample_frames": 8,
        }

    def get_dummy_inputs(self):
        return {"sample": randn_tensor((2, 3, 8, 8, 8), generator=self.generator, device=torch_device)}

    @property
    def input_shape(self):
        return (3, 8, 8, 8)

    @property
    def output_shape(self):
        return (3, 8, 8, 8)


class TestAutoencoderKLMagiModel(AutoencoderKLMagiTesterConfig, ModelTesterMixin):
    @require_torch_multi_accelerator
    @torch.no_grad()
    def test_model_parallelism(self, base_model_output, tmp_path):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval()
        model.save_pretrained(tmp_path)
        model = self.model_class.from_pretrained(tmp_path, device_map={"encoder": 0, "decoder": 1})
        output = model(**self.get_dummy_inputs()).sample
        assert next(model.encoder.parameters()).device == torch.device("cuda:0")
        assert next(model.decoder.parameters()).device == torch.device("cuda:1")
        torch.testing.assert_close(output.cpu(), base_model_output.cpu(), atol=1e-5, rtol=0)

    @torch.no_grad()
    def test_single_frame_encode_and_decode(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        image = self.get_dummy_inputs()["sample"][:, :, :1]
        posterior = model.encode(image).latent_dist
        repeated = model.encode(image.repeat(1, 1, 4, 1, 1)).latent_dist
        torch.testing.assert_close(posterior.parameters, repeated.parameters)
        latent = posterior.mode()
        assert latent.shape == (2, 4, 1, 4, 4)
        decoded = model.decode(latent).sample
        assert decoded.shape == image.shape
        torch.testing.assert_close(model(image).sample, decoded)
        torch.testing.assert_close(decoded, model.decoder(latent)[:, :, :1])

    @torch.no_grad()
    def test_encoder_preserves_channel_last_storage(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        moments = model.encoder(self.get_dummy_inputs()["sample"])
        assert moments.stride(1) == 1
        assert not moments.is_contiguous()
        model.enable_tiling()
        posterior = model.encode(self.get_dummy_inputs()["sample"][:1]).latent_dist
        assert posterior.mean.is_contiguous(memory_format=torch.channels_last_3d)
        torch.testing.assert_close(posterior.mean, posterior.parameters.chunk(2, dim=1)[0], atol=0, rtol=0)

    @torch.no_grad()
    def test_four_frame_forward_preserves_frames(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        video = self.get_dummy_inputs()["sample"][:, :, :4]
        assert model(video).sample.shape == video.shape

    @torch.no_grad()
    def test_position_interpolation(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = randn_tensor((1, 3, 12, 12, 8), generator=self.generator, device=torch_device)
        posterior = model.encode(sample).latent_dist
        assert posterior.mean.shape == (1, 4, 3, 6, 4)
        assert model.decode(posterior.mode()).sample.shape == sample.shape

    @pytest.mark.parametrize("shape", [(1, 3, 6, 8, 8), (1, 3, 8, 7, 8)])
    def test_invalid_video_shape(self, shape):
        model = self.model_class(**self.get_init_dict()).to(torch_device)
        with pytest.raises(ValueError, match="divisible"):
            model.encode(torch.zeros(shape, device=torch_device))

    @torch.no_grad()
    def test_posterior_sampling_generator(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = self.get_dummy_inputs()["sample"]
        first = model(sample, sample_posterior=True, generator=self.generator).sample
        second = model(sample, sample_posterior=True, generator=self.generator).sample
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    @pytest.mark.parametrize("frames", [1, 4, 8, 12, 16, 17])
    @torch.no_grad()
    def test_temporal_tiles_and_frame_count(self, frames):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = randn_tensor((1, 3, frames, 8, 8), generator=self.generator, device=torch_device)
        expected_latents = torch.cat([model.encode(tile).latent_dist.mode() for tile in sample.split(8, dim=2)], dim=2)
        expected_video = torch.cat([model.decode(tile).sample for tile in expected_latents.split(2, dim=2)], dim=2)
        model.enable_tiling(tile_sample_min_length=8, allow_spatial_tiling=False)
        actual_latents = model.encode(sample).latent_dist.mode()
        torch.testing.assert_close(actual_latents, expected_latents)
        torch.testing.assert_close(model.decode(actual_latents).sample, expected_video)
        assert model(sample).sample.shape == sample.shape
        assert model.decode(actual_latents, num_frames=frames).sample.shape == sample.shape

    @torch.no_grad()
    def test_spatial_tiling_and_slicing(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        sample = randn_tensor((2, 3, 16, 12, 12), generator=self.generator, device=torch_device)
        model.enable_tiling(tile_sample_min_length=8, temporal_tile_overlap_factor=0.5)
        expected_latents = model.encode(sample).latent_dist.mode()
        expected_video = model.decode(expected_latents, num_frames=16).sample
        assert expected_latents.shape == (2, 4, 4, 6, 6)
        assert expected_video.shape == sample.shape
        assert torch.isfinite(expected_video).all()
        model.enable_slicing()
        torch.testing.assert_close(model.encode(sample).latent_dist.mode(), expected_latents, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            model.decode(expected_latents, num_frames=16).sample, expected_video, atol=1e-5, rtol=1e-5
        )
        model.disable_slicing()
        model.disable_tiling()
        untiled = model.encode(sample).latent_dist.mode()
        assert not torch.allclose(untiled, expected_latents)
        assert model(sample).sample.shape == sample.shape

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_overlap_blending(self, dim):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = 4
        before = torch.full(shape, 10.0, device=torch_device)
        after = torch.full(shape, 30.0, device=torch_device)
        output = self.model_class._blend(before, after, 4, dim)
        torch.testing.assert_close(output.flatten(), torch.tensor([10.0, 15.0, 20.0, 25.0], device=torch_device))

    @pytest.mark.parametrize("dim", [2, 3, 4])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_decoder_blending(self, dim, dtype):
        before = randn_tensor((1, 3, 8, 8, 8), generator=self.generator, device=torch_device).to(dtype)
        after = randn_tensor((1, 3, 8, 8, 8), generator=self.generator, device=torch_device).to(dtype)
        expected = after.clone()
        for index in range(4):
            previous = before.select(dim, 4 + index).float()
            current = after.select(dim, index).float()
            expected.select(dim, index).copy_(previous * (1 - index / 4) + current * (index / 4))
        actual = self.model_class._blend(before, after, 4, dim, upcast=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tile_sample_min_length": 0},
            {"tile_sample_min_length": 6},
            {"tile_sample_min_height": 7},
            {"temporal_tile_overlap_factor": 1.0},
            {"spatial_tile_overlap_factor": -0.1},
            {"temporal_tile_overlap_factor": 0.3},
        ],
    )
    def test_invalid_tiling_settings(self, kwargs):
        model = self.model_class(**self.get_init_dict())
        with pytest.raises(ValueError):
            model.enable_tiling(**kwargs)
        assert not model.use_tiling

    def test_partial_temporal_patch_is_rejected(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device)
        model.enable_tiling(tile_sample_min_length=8, temporal_tile_overlap_factor=0.5, allow_spatial_tiling=False)
        with pytest.raises(ValueError, match="complete patches"):
            model.encode(torch.zeros((1, 3, 9, 8, 8), device=torch_device))

    @pytest.mark.parametrize("num_frames", [0, 4, 9])
    def test_invalid_decode_frame_count(self, num_frames):
        model = self.model_class(**self.get_init_dict()).to(torch_device)
        latent = torch.zeros((1, 4, 2, 4, 4), device=torch_device)
        with pytest.raises(ValueError, match="num_frames"):
            model.decode(latent, num_frames=num_frames)

    @torch.no_grad()
    def test_posterior_sampling_dtype_and_seed(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device, dtype=torch.bfloat16).eval()
        posterior = model.encode(self.get_dummy_inputs()["sample"].bfloat16()).latent_dist
        first = posterior.sample(generator=torch.Generator("cpu").manual_seed(12))
        second = posterior.sample(generator=torch.Generator("cpu").manual_seed(12))
        different = posterior.sample(generator=torch.Generator("cpu").manual_seed(13))
        assert first.dtype == torch.bfloat16 and first.device == posterior.mean.device
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert not torch.equal(first, different)


class TestAutoencoderKLMagiMemory(AutoencoderKLMagiTesterConfig, MemoryTesterMixin):
    @torch.no_grad()
    def test_tiled_group_offload(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        model.enable_tiling(temporal_tile_overlap_factor=0.5)
        sample = self.get_dummy_inputs()["sample"]
        expected = model(sample).sample
        model.enable_group_offload(
            onload_device=torch_device, offload_device="cpu", offload_type="block_level", num_blocks_per_group=1
        )
        torch.testing.assert_close(model(sample).sample, expected, atol=1e-5, rtol=0)


class TestAutoencoderKLMagiTorchCompile(AutoencoderKLMagiTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height=4, width=4):
        return {"sample": randn_tensor((2, 3, 8, height, width), generator=self.generator, device=torch_device)}


class TestAutoencoderKLMagiTraining(AutoencoderKLMagiTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"MagiVAEEncoder", "MagiVAEDecoder"})

    def test_tiled_backward(self):
        model = self.model_class(**self.get_init_dict()).train()
        model.enable_tiling(temporal_tile_overlap_factor=0.5)
        model(self.get_dummy_inputs()["sample"].cpu()).sample.square().mean().backward()
        assert model.encoder.patch_embed.proj.weight.grad.isfinite().all()
        assert model.decoder.last_layer.weight.grad.isfinite().all()


class TestAutoencoderKLMagiAttention(AutoencoderKLMagiTesterConfig, AttentionTesterMixin):
    pass
