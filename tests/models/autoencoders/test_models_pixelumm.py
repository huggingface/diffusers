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

from diffusers import PixelUMMDecoder
from diffusers.models.autoencoders.autoencoder_pixelumm import local_attention_layout
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    enable_full_determinism,
    is_torch_compile,
    require_torch_gpu,
    require_torch_multi_gpu,
    torch_device,
)
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class PixelUMMDecoderTesterConfig(BaseModelTesterConfig):
    main_input_name = "sample"
    # The tiny model is dominated by the fixed RGB projections; reserve room for their offload staging.
    model_split_percents = [0.95, 0.97]

    @property
    def model_class(self):
        return PixelUMMDecoder

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "hidden_size": 16,
            # Keep the blocks large enough for balanced placement to split the fixed-size RGB projections.
            "intermediate_size": 512,
            "num_layers": 2,
            "num_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 8,
            "local_attention_tile_pixels": 0,
            "temporal_window_latents": 3,
            "temporal_overlap_latents": 1,
        }

    def get_dummy_inputs(self, height=1, width=1):
        return {
            "latent": randn_tensor((1, 32, 1, height, width), generator=self.generator, device=torch_device),
            "sample": randn_tensor((1, 3, 4, height * 32, width * 32), generator=self.generator, device=torch_device),
        }

    @property
    def input_shape(self):
        return (3, 4, 32, 32)

    @property
    def output_shape(self):
        return (3, 4, 32, 32)


class TestPixelUMMDecoderModel(PixelUMMDecoderTesterConfig, ModelTesterMixin):
    @torch.no_grad()
    def test_decode_generator_and_crop(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        latent = self.get_dummy_inputs()["latent"]
        first = model.decode(latent, generator=torch.Generator().manual_seed(7)).sample
        second = model.decode(latent, generator=torch.Generator().manual_seed(7), pixel_frames=3, return_dict=False)[0]
        torch.testing.assert_close(first[:, :, :3], second, atol=0, rtol=0)
        other = model.decode(latent, generator=torch.Generator().manual_seed(8)).sample
        assert not torch.equal(first, other)
        large_seed = model.decode(latent, generator=torch.Generator().manual_seed(2**64 - 1)).sample
        assert large_seed.isfinite().all()
        with pytest.raises(ValueError, match="pixel_frames"):
            model.decode(latent, pixel_frames=5)

    @torch.no_grad()
    def test_decode_without_temporal_or_spatial_overlap(self):
        config = self.get_init_dict()
        config.update(
            temporal_window_latents=2,
            temporal_overlap_latents=0,
            decoder_tile_pixels=32,
            decoder_tile_overlap_pixels=0,
        )
        model = self.model_class(**config).to(torch_device).eval()
        latent = randn_tensor((1, 32, 4, 2, 2), generator=self.generator, device=torch_device)
        first = model.decode(latent).sample
        assert first.shape == (1, 3, 16, 64, 64)
        assert first.isfinite().all()
        latent[:, :, 2:, 1:, 1:] += 5
        second = model.decode(latent).sample
        torch.testing.assert_close(first[:, :, :8], second[:, :, :8], atol=0, rtol=0)
        torch.testing.assert_close(first[..., :32, :32], second[..., :32, :32], atol=0, rtol=0)
        assert not torch.equal(first[:, :, 8:, 32:, 32:], second[:, :, 8:, 32:, 32:])

    @torch.no_grad()
    def test_streaming_matches_whole_clip_with_segment_padding(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        latent = randn_tensor((1, 32, 7, 1, 1), generator=self.generator, device=torch_device)
        complete = model.decode(latent).sample
        session = model.create_streaming_decode_session(to_cpu=False)
        first = session.push_preview(latent[:, :, :4], pixel_frames=15)
        second = session.push_preview(latent[:, :, 4:], pixel_frames=10)
        preview = torch.cat((first.video[:, :, : second.pixel_start], second.video), dim=2)
        expected = torch.cat((complete[:, :, :15], complete[:, :, 16:26]), dim=2)
        torch.testing.assert_close(preview, expected, atol=1e-5, rtol=1e-5)
        session.finish()
        torch.testing.assert_close(session.full_video(), expected, atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_streaming_tail_correction_and_rollback(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        latent = randn_tensor((1, 32, 8, 1, 1), generator=self.generator, device=torch_device)
        session = model.create_streaming_decode_session(to_cpu=False)
        session.final_segment = False
        first = session.push_preview(latent[:, :, :4])
        session.push_preview(latent[:, :, 4:6])
        session.rollback_last_push()
        session.final_segment = True
        update = session.push_preview(latent[:, :, 4:])
        preview = torch.cat((first.video[:, :, : update.pixel_start], update.video), dim=2)
        expected = model.decode(latent).sample
        torch.testing.assert_close(preview, expected, atol=1e-5, rtol=1e-5)
        session.finish()
        torch.testing.assert_close(session.full_video(), expected, atol=1e-5, rtol=1e-5)
        with pytest.raises(RuntimeError, match="after finish"):
            session.push(latent)

    @torch.no_grad()
    def test_local_attention_excludes_distant_spatial_tiles(self):
        config = self.get_init_dict()
        config["local_attention_tile_pixels"] = 16
        config["num_layers"] = 1
        config["head_dim"] = 16
        model = self.model_class(**config).to(torch_device).eval()
        inputs = self.get_dummy_inputs(height=4, width=4)
        first = model(**inputs).sample
        inputs["sample"] = inputs["sample"].clone()
        inputs["sample"][..., -16:, -16:] += 5
        second = model(**inputs).sample
        torch.testing.assert_close(first[..., :16, :16], second[..., :16, :16], atol=0, rtol=0)
        assert not torch.equal(first[..., -16:, -16:], second[..., -16:, -16:])

    @require_torch_gpu
    @pytest.mark.parametrize("grid,tile", [((3, 8, 8), 2), ((2, 9, 5), 2), ((2, 2, 9), 2), ((1, 9, 2), 2)])
    def test_local_attention_mask_matches_tile_neighborhoods(self, grid, tile):
        _, _, (neighborhoods, block_mask) = local_attention_layout(grid, tile, torch_device)
        count = grid[0] * grid[1] * grid[2]
        indices = torch.arange(count, device=torch_device)
        actual = block_mask.mask_mod(None, None, indices[:, None], indices[None, :])
        expected = torch.zeros((count, count), dtype=torch.bool, device=torch_device)
        for (start, end), neighbors in neighborhoods:
            for first, last in neighbors:
                expected[start:end, first:last] = True
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.no_grad()
    def test_multistep_decode(self):
        config = self.get_init_dict()
        config["num_steps"] = 2
        model = self.model_class(**config).to(torch_device).eval()
        output = model.decode(self.get_dummy_inputs()["latent"]).sample
        assert output.shape == (1, *self.output_shape)
        assert output.isfinite().all()
        with pytest.raises(ValueError, match="num_steps=1"):
            model.create_streaming_decode_session()


class TestPixelUMMDecoderMemory(PixelUMMDecoderTesterConfig, MemoryTesterMixin):
    @require_torch_multi_gpu
    @pytest.mark.parametrize("local_attention_tile_pixels,num_steps", [(0, 1), (0, 2), (16, 1)])
    @torch.no_grad()
    def test_device_map_embeddings_and_local_layout(self, local_attention_tile_pixels, num_steps, tmp_path):
        torch.manual_seed(0)
        config = self.get_init_dict()
        config.update(
            hidden_size=32, head_dim=16, local_attention_tile_pixels=local_attention_tile_pixels, num_steps=num_steps
        )
        model = self.model_class(**config).to("cuda:0").eval()
        inputs = self.get_dummy_inputs(height=2, width=2)
        expected = model(**inputs).sample
        expected_decoded = model.decode(inputs["latent"]).sample
        model.cpu().save_pretrained(tmp_path)
        device_map = {
            "decoder.embed": "cuda:1",
            "decoder.cond_proj": "cuda:0",
            "decoder.layers.0": "cuda:0",
            "decoder.layers.1": "cuda:1",
            "decoder.final_norm": "cuda:0",
            "decoder.head": "cuda:1",
        }
        loaded = self.model_class.from_pretrained(tmp_path, device_map=device_map)
        assert loaded.decoder.embed.weight.device == torch.device("cuda:1")
        assert loaded.decoder.cond_proj.weight.device == torch.device("cuda:0")
        torch.testing.assert_close(loaded(**inputs).sample, expected, atol=1e-5, rtol=0)
        actual_decoded = loaded.decode(inputs["latent"]).sample
        torch.testing.assert_close(actual_decoded.cpu(), expected_decoded.cpu(), atol=1e-5, rtol=0)


class TestPixelUMMDecoderTorchCompile(PixelUMMDecoderTesterConfig, TorchCompileTesterMixin):
    @is_torch_compile
    @require_torch_gpu
    @torch.no_grad()
    def test_compile_local_attention(self):
        config = self.get_init_dict()
        config.update(local_attention_tile_pixels=16, num_layers=1, head_dim=16)
        model = self.model_class(**config).to(torch_device).eval()
        inputs = self.get_dummy_inputs(height=2, width=2)
        expected = model(**inputs).sample
        compiled = torch.compile(model, fullgraph=True)
        actual = compiled(**inputs).sample
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestPixelUMMDecoderTraining(PixelUMMDecoderTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"PixelUMMDecoderModel"})


class TestPixelUMMDecoderAttention(PixelUMMDecoderTesterConfig, AttentionTesterMixin):
    pass
