# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers import MagiImageToVideoBlocks, MagiModularPipeline, MagiVideoToVideoBlocks, ModularPipeline
from diffusers.modular_pipelines.magi.decoders import MagiPrefixVaeDecoderStep
from diffusers.modular_pipelines.magi.encoders import MagiImageVaeEncoderStep, MagiVideoVaeEncoderStep

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
)
from .test_modular_pipeline_magi import tiny_magi_path  # noqa: F401
from .testing_utils import MagiGuiderTesterMixin


@pytest.fixture(scope="module")
def tiny_prefix_paths(request, tmp_path_factory):
    base_path = request.getfixturevalue("tiny_magi_path")
    paths = {}
    for name, blocks in [("image", MagiImageToVideoBlocks), ("video", MagiVideoToVideoBlocks)]:
        pipe = blocks().init_pipeline(base_path)
        pipe.load_components()
        path = str(tmp_path_factory.mktemp(f"tiny-magi-{name}"))
        pipe.save_pretrained(path, overwrite_modular_index=True)
        paths[name] = path
    return paths


class MagiImagePipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = MagiModularPipeline
    pipeline_blocks_class = MagiImageToVideoBlocks
    params = frozenset(["prompt", "image", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    output_name = "videos"
    workflow = "image"

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def model_path(cls, tiny_prefix_paths):
        cls.pretrained_model_name_or_path = tiny_prefix_paths[cls.workflow]

    def get_dummy_inputs(self, seed=0):
        pixels = torch.arange(3 * 8 * 8).reshape(1, 3, 8, 8).to(torch.uint8)
        return {
            "prompt": "a cat runs",
            "image": pixels,
            "height": 8,
            "width": 8,
            "num_frames": 16,
            "chunk_width": 2,
            "window_size": 2,
            "num_inference_steps": 4,
            "max_sequence_length": 8,
            "clean_caption": False,
            "output_type": "pt",
            "generator": self.get_generator(seed),
        }


class TestMagiImagePipelineFast(MagiImagePipelineTesterConfig, ModularPipelineTesterMixin):
    def test_image_prefix_and_roundtrip(self, tmp_path):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        result = pipe(**inputs, output=["videos", "latents", "conditioning_latents", "completed_chunks"])
        assert result["videos"].shape == (1, 24, 3, 8, 8)
        assert result["conditioning_latents"].shape == (1, 4, 1, 4, 4)
        assert result["completed_chunks"] == [0, 1, 2]
        assert not torch.equal(result["latents"][:, :, :1], result["conditioning_latents"])
        pipe.save_pretrained(str(tmp_path), overwrite_modular_index=True)
        restored = ModularPipeline.from_pretrained(str(tmp_path))
        restored.load_components()
        actual = restored(**self.get_dummy_inputs(), output="videos")
        torch.testing.assert_close(actual, result["videos"], atol=0, rtol=0)

    def test_prefix_batch_expansion(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        pixels = torch.cat([inputs["image"], 255 - inputs["image"]])
        original = pixels.clone()
        inputs.update(prompt=["a cat", "a cat runs"], image=pixels, num_images_per_prompt=2)
        result = pipe(**inputs, output=["conditioning_latents", "videos"])
        prefix = result["conditioning_latents"]
        assert prefix.shape[0] == 4 and result["videos"].shape[0] == 4
        torch.testing.assert_close(prefix[0], prefix[1], atol=0, rtol=0)
        torch.testing.assert_close(prefix[2], prefix[3], atol=0, rtol=0)
        assert not torch.equal(prefix[0], prefix[2])
        torch.testing.assert_close(pixels, original, atol=0, rtol=0)

    def test_prefix_reinjected_for_every_branch(self):
        pipe = self.get_pipeline()
        calls = []

        def observe(module, args, kwargs):
            calls.append(kwargs["hidden_states"].detach().cpu().clone())

        handle = pipe.transformer.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            result = pipe(**self.get_dummy_inputs(), output=["conditioning_latents", "latents"])
        finally:
            handle.remove()
        prefix = result["conditioning_latents"].cpu()
        for states in calls[:12]:
            torch.testing.assert_close(states[:1, :, :1], prefix, atol=0, rtol=0)
        assert not torch.equal(result["latents"][:, :, :1].cpu(), prefix)

    @pytest.mark.parametrize("bad", [torch.zeros(1, 3, 8, 8), torch.zeros(3, 8, 8, dtype=torch.uint8)])
    def test_invalid_image(self, bad):
        with pytest.raises(ValueError):
            self.run_pipe(self.get_pipeline(), image=bad)

    @torch.no_grad()
    def test_encoder_matches_posterior_mode(self):
        pipe = MagiImageVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components()
        image = self.get_dummy_inputs()["image"]
        actual = pipe(image=image, output="conditioning_latents")
        expected = pipe.vae.encode(image.unsqueeze(2).float() / 127.5 - 1).latent_dist.mode() * 0.18215
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestMagiImagePipelineLoading(MagiImagePipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestMagiImagePipelineMemory(MagiImagePipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class MagiVideoPipelineTesterConfig(MagiImagePipelineTesterConfig):
    pipeline_blocks_class = MagiVideoToVideoBlocks
    params = frozenset(["prompt", "video", "height", "width", "num_frames"])
    workflow = "video"

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        image = inputs.pop("image")
        inputs["video"] = image.unsqueeze(2).repeat(1, 1, 12, 1, 1)
        return inputs


class TestMagiVideoPipelineFast(MagiVideoPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.parametrize("prefix_frames, expected_frames", [(8, 16), (12, 17)])
    def test_prefix_duration_and_trim(self, prefix_frames, expected_frames):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["video"] = inputs["video"][:, :, :prefix_frames]
        result = pipe(**inputs, output=["latents", "videos", "conditioning_latents", "prompt_embeds"])
        prefix = result["conditioning_latents"]
        full_length = prefix.shape[2] // 2 * 2
        torch.testing.assert_close(result["latents"][:, :, :full_length], prefix[:, :, :full_length], atol=0, rtol=0)
        assert result["videos"].shape == (1, expected_frames, 3, 8, 8)
        generated = result["latents"].shape[2] // 2 - prefix.shape[2] // 2
        expected_duration = pipe.text_conditioning.special_embedding.weight[generated]
        torch.testing.assert_close(result["prompt_embeds"][0, prefix.shape[2] // 2, 0], expected_duration)
        decoder = MagiPrefixVaeDecoderStep().init_pipeline(self.pretrained_model_name_or_path)
        decoder.load_components()
        suffix = decoder(
            latents=result["latents"],
            conditioning_latents=prefix,
            chunk_width=2,
            output_type="latent",
            output="videos",
        )
        torch.testing.assert_close(suffix, result["latents"][:, :, prefix.shape[2] :], atol=0, rtol=0)
        chunks = []
        with torch.no_grad():
            for start in range(0, result["latents"].shape[2], 2):
                if start + 2 <= prefix.shape[2]:
                    continue
                chunk = result["latents"][:, :, max(start, prefix.shape[2]) : start + 2]
                chunks.append(pipe.vae.decode(chunk / 0.18215).sample)
        expected = pipe.video_processor.postprocess_video(torch.cat(chunks, dim=2), output_type="pt")
        torch.testing.assert_close(result["videos"], expected, atol=0, rtol=0)

    def test_tiled_single_position_suffix(self):
        decoder = MagiPrefixVaeDecoderStep().init_pipeline(self.pretrained_model_name_or_path)
        decoder.load_components()
        decoder.vae.enable_tiling(tile_sample_min_length=12)
        latents = torch.randn(1, 4, 18, 4, 4)
        video = decoder(
            latents=latents, conditioning_latents=latents[:, :, :8], chunk_width=6, output_type="pt", output="videos"
        )
        assert video.shape == (1, 37, 3, 8, 8)

    def test_partial_prefix_reinjected(self):
        pipe = self.get_pipeline()
        calls = []

        def observe(module, args, kwargs):
            calls.append(kwargs["hidden_states"].detach().cpu().clone())

        handle = pipe.transformer.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            result = pipe(**self.get_dummy_inputs(), output=["conditioning_latents", "latents"])
        finally:
            handle.remove()
        prefix = result["conditioning_latents"].cpu()
        torch.testing.assert_close(calls[0], prefix[:, :, :2], atol=0, rtol=0)
        for states in calls[1:13]:
            torch.testing.assert_close(states[:1, :, :1], prefix[:, :, 2:3], atol=0, rtol=0)
        assert not torch.equal(result["latents"][:, :, 2:3].cpu(), prefix[:, :, 2:3])

    @torch.no_grad()
    def test_encoder_reusable(self):
        pipe = MagiVideoVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components()
        video = self.get_dummy_inputs()["video"]
        actual = pipe(video=video, output="conditioning_latents")
        expected = pipe.vae.encode(video.float() / 127.5 - 1).latent_dist.mode() * 0.18215
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestMagiVideoPipelineLoading(MagiVideoPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestMagiVideoPipelineMemory(MagiVideoPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestMagiImagePipelineGuider(MagiImagePipelineTesterConfig, MagiGuiderTesterMixin):
    pass


class TestMagiVideoPipelineGuider(MagiVideoPipelineTesterConfig, MagiGuiderTesterMixin):
    pass
