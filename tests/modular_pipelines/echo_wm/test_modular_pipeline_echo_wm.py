# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np
import PIL.Image
import pytest
import torch

from diffusers import (
    EchoWMBlocks,
    EchoWMFlashBlocks,
    EchoWMFlashModularPipeline,
    EchoWMModularPipeline,
    FlowMatchEulerDiscreteScheduler,
    LTX2Guidance,
    ModularPipeline,
)
from diffusers.modular_pipelines.echo_wm import apply_action_overlay
from diffusers.modular_pipelines.echo_wm.action import action_to_camera_trajectory
from diffusers.modular_pipelines.echo_wm.decoders import EchoWMVaeDecoderStep
from diffusers.modular_pipelines.echo_wm.denoise import (
    _clear_echo_wm_audio_caches,
    _flash_layout,
    echo_wm_flash_sigmas,
)
from diffusers.modular_pipelines.echo_wm.encoders import _preprocess_echo_wm_images

from ...testing_utils import torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


TINY_ECHO_WM_BASE_REPO = "Echo-Team/tiny-echo-wm-base-diffusers"
TINY_ECHO_WM_FLASH_REPO = "Echo-Team/tiny-echo-wm-flash-diffusers"


class EchoWMModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = EchoWMModularPipeline
    pipeline_blocks_class = EchoWMBlocks
    pretrained_model_name_or_path = TINY_ECHO_WM_BASE_REPO
    params = frozenset(["prompt", "image", "action", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt", "image"])
    optional_params = frozenset(["num_inference_steps", "latents", "output_type"])
    output_name = "videos"
    expected_workflow_blocks = {}
    num_frames = 9

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot",
            "negative_prompt": "",
            "image": PIL.Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)),
            "image_crf": 0,
            "action": f"none-{self.num_frames - 1}",
            "num_frames": self.num_frames,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "num_inference_steps": 2,
            "generator": self.get_generator(seed),
            "output_type": "pt",
        }


class EchoWMModularPipelineFastTesterMixin(ModularPipelineTesterMixin):
    def test_inference_batch_single_identical(self):
        # Match the tolerance used by the LTX2 modular pipeline this implementation builds on.
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=9e-2)


class TestEchoWMModularPipelineFast(EchoWMModularPipelineTesterConfig, EchoWMModularPipelineFastTesterMixin):
    def test_audio_output(self):
        pipe = self.get_pipeline().to(torch_device)
        output = pipe(**self.get_dummy_inputs(), output=["videos", "audio"])

        assert output["videos"].shape == (1, self.num_frames, 3, 32, 32)
        assert output["audio"].shape[:2] == (1, 2)
        assert torch.isfinite(output["audio"]).all()

    def test_blockset_is_modular_only(self):
        blocks = EchoWMBlocks()
        assert EchoWMModularPipeline.default_blocks_name == "EchoWMBlocks"
        assert list(blocks.sub_blocks) == ["text", "camera", "image_encoder", "denoise", "decode"]
        assert isinstance(blocks.sub_blocks["decode"].sub_blocks["video_decode"], EchoWMVaeDecoderStep)
        assert isinstance(blocks.init_pipeline(), EchoWMModularPipeline)

    def test_default_sampling_components(self, tmp_path):
        pipe = EchoWMBlocks().init_pipeline()

        assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler)
        assert not pipe.scheduler.config.use_dynamic_shifting
        assert pipe.scheduler.config.shift == pytest.approx(np.exp(2.05))
        assert pipe.scheduler.config.shift_terminal == 0.1
        assert isinstance(pipe.guider, LTX2Guidance)
        assert pipe.guider.config.guidance_scale == 4.0
        assert pipe.guider.config.spatio_temporal_guidance_blocks == [29]
        assert isinstance(pipe.audio_guider, LTX2Guidance)
        assert pipe.audio_guider.config.guidance_scale == 2.0

        pipe.save_pretrained(str(tmp_path))
        with open(tmp_path / "modular_model_index.json") as file:
            index = json.load(file)
        assert "scheduler" not in index
        assert "guider" not in index
        assert "audio_guider" not in index

        reloaded = ModularPipeline.from_pretrained(tmp_path)
        assert isinstance(reloaded.scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(reloaded.guider, LTX2Guidance)
        assert isinstance(reloaded.audio_guider, LTX2Guidance)

    def test_action_to_camera_trajectory(self):
        poses, intrinsics = action_to_camera_trajectory("w-2,j-2", 5, 1280, 704)
        assert poses.shape == (5, 4, 4)
        assert intrinsics.shape == (3, 3)
        assert torch.allclose(poses[0], torch.eye(4))
        assert not torch.allclose(poses[-1], poses[0])
        assert intrinsics[0, 2] == 640
        assert intrinsics[1, 2] == 352

    @pytest.mark.parametrize("action", ["", "x-2", "w-0", "w"])
    def test_invalid_action_raises(self, action):
        with pytest.raises(ValueError):
            action_to_camera_trajectory(action, 5, 1280, 704)

    def test_apply_action_overlay(self):
        frames = [PIL.Image.new("RGB", (320, 176), "navy") for _ in range(2)]
        output = apply_action_overlay(frames, "w-1,l-1")

        assert len(output) == 2
        assert all(frame.mode == "RGB" and frame.size == (320, 176) for frame in output)
        assert np.any(np.asarray(output[0]) != np.asarray(frames[0]))
        assert np.any(np.asarray(output[0]) != np.asarray(output[1]))
        assert frames[0].getpixel((0, 0)) == (0, 0, 128)

    def test_apply_action_overlay_rejects_non_pil_output(self):
        with pytest.raises(TypeError, match="output_type='pil'"):
            apply_action_overlay(torch.zeros(2, 3, 16, 16), "none-2")

    def test_image_preprocessing_uses_torch_resize(self):
        image = PIL.Image.fromarray(np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3))
        actual = _preprocess_echo_wm_images(image, height=4, width=4, crf=0, device=torch.device("cpu"))

        source = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        expected = torch.nn.functional.interpolate(source, size=(4, 6), mode="bilinear", align_corners=False)
        expected = expected[:, :, :, 1:5] / 127.5 - 1.0

        torch.testing.assert_close(actual, expected)


class TestEchoWMModularPipelineLoading(EchoWMModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestEchoWMModularPipelineWorkflow(EchoWMModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestEchoWMModularPipelineMemory(EchoWMModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class EchoWMFlashModularPipelineTesterConfig(EchoWMModularPipelineTesterConfig):
    pipeline_class = EchoWMFlashModularPipeline
    pipeline_blocks_class = EchoWMFlashBlocks
    pretrained_model_name_or_path = TINY_ECHO_WM_FLASH_REPO
    optional_params = frozenset(["output_type"])
    not_params = frozenset(["negative_prompt", "num_inference_steps", "latents"])
    num_frames = 25

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        inputs.pop("negative_prompt")
        inputs.pop("num_inference_steps")
        return inputs


class TestEchoWMFlashModularPipelineFast(EchoWMFlashModularPipelineTesterConfig, EchoWMModularPipelineFastTesterMixin):
    def test_audio_output(self):
        pipe = self.get_pipeline().to(torch_device)
        output = pipe(**self.get_dummy_inputs(), output=["videos", "audio"])

        assert output["videos"].shape == (1, self.num_frames, 3, 32, 32)
        assert output["audio"].shape[:2] == (1, 2)
        assert torch.isfinite(output["audio"]).all()

    def test_blockset_is_modular_only(self):
        blocks = EchoWMFlashBlocks()
        assert EchoWMFlashModularPipeline.default_blocks_name == "EchoWMFlashBlocks"
        assert list(blocks.sub_blocks) == ["text", "camera", "image_encoder", "denoise", "decode"]
        assert blocks.sub_blocks["denoise"].__class__.__name__ == "EchoWMFlashDenoiseStep"
        assert list(blocks.sub_blocks["denoise"].sub_blocks) == ["chunk_denoiser"]
        assert isinstance(blocks.sub_blocks["decode"].sub_blocks["video_decode"], EchoWMVaeDecoderStep)
        assert "negative_prompt" not in blocks.input_names
        assert isinstance(blocks.init_pipeline(), EchoWMFlashModularPipeline)

    def test_default_scheduler(self):
        pipe = EchoWMFlashBlocks().init_pipeline()

        assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler)
        assert not pipe.scheduler.config.use_dynamic_shifting
        assert pipe.scheduler.config.shift == pytest.approx(np.exp(2.05))
        assert pipe.scheduler.config.shift_terminal == 0.1

    def test_layout_and_schedule(self):
        video_blocks, audio_blocks, audio_frames = _flash_layout(31)
        assert video_blocks[:2] == [(0, 1), (1, 4)]
        assert video_blocks[-1] == (28, 31)
        assert audio_blocks[:2] == [(0, 2), (2, 27)]
        assert audio_frames == 252
        sigmas = echo_wm_flash_sigmas([1000, 750, 500, 250])
        assert sigmas[0] == 1.0
        assert all(current > following for current, following in zip(sigmas, sigmas[1:]))

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match=r"1 \+ 3"):
            _flash_layout(30)

    def test_video_sink_warmup_clears_audio_caches(self):
        caches = [
            {
                "video_self": {"key": torch.ones(1), "value": torch.ones(1), "positions": torch.tensor([0])},
                "audio_self": {"key": torch.ones(1), "value": torch.ones(1), "positions": torch.tensor([0])},
                "audio_text": {"key": torch.ones(1), "value": torch.ones(1)},
            }
        ]

        _clear_echo_wm_audio_caches(caches)

        assert caches[0]["video_self"]["key"] is not None
        assert all(caches[0]["audio_self"][name] is None for name in ("key", "value", "positions"))
        assert all(caches[0]["audio_text"][name] is None for name in ("key", "value"))


class TestEchoWMFlashModularPipelineLoading(EchoWMFlashModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestEchoWMFlashModularPipelineWorkflow(EchoWMFlashModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestEchoWMFlashModularPipelineMemory(EchoWMFlashModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass


class TestEchoWMVideoDecode:
    @pytest.fixture
    def pipe(self):
        pipe = EchoWMVaeDecoderStep().init_pipeline("hf-internal-testing/tiny-ltx2-modular-pipe")
        pipe.load_components(dtype=torch.float32)
        return pipe.to(torch_device)

    @staticmethod
    def get_inputs(pipe, num_frames=73, height=8, width=12):
        latent_frames = (num_frames - 1) // pipe.vae.temporal_compression_ratio + 1
        latent_height = height // pipe.vae.spatial_compression_ratio
        latent_width = width // pipe.vae.spatial_compression_ratio
        return {
            "latents": torch.randn(
                1,
                latent_frames * latent_height * latent_width,
                pipe.vae.config.latent_channels,
                generator=torch.Generator("cpu").manual_seed(0),
            ).to(torch_device),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "dtype": torch.float32,
            "output_type": "pt",
        }

    @staticmethod
    def tiling_settings(vae):
        return {
            name: value
            for name, value in vars(vae).items()
            if name.startswith("tile_sample_") or name in ("use_tiling", "use_framewise_decoding")
        }

    @pytest.mark.parametrize(
        "num_frames,height,width,tiling_kwargs,max_tile_shape",
        [
            (73, 8, 12, {}, (33, 4, 6)),
            (1, 8, 520, {}, (1, 4, 256)),
            (
                25,
                16,
                24,
                {
                    "vae_tile_size": 16,
                    "vae_tile_overlap": 4,
                    "vae_temporal_tile_size": 8,
                    "vae_temporal_tile_overlap": 4,
                },
                (5, 5, 8),
            ),
        ],
    )
    def test_decode_tiles_bound_decoder_inputs(self, pipe, num_frames, height, width, tiling_kwargs, max_tile_shape):
        inputs = self.get_inputs(pipe, num_frames, height, width)
        previous_settings = self.tiling_settings(pipe.vae)
        tile_shapes = []
        handle = pipe.vae.decoder.register_forward_pre_hook(
            lambda module, args: tile_shapes.append(tuple(args[0].shape[-3:]))
        )
        try:
            video = pipe(**inputs, **tiling_kwargs, output="videos")
        finally:
            handle.remove()

        assert video.shape == (1, num_frames, 3, height, width)
        assert torch.isfinite(video).all()
        assert len(tile_shapes) > 1
        assert all(all(size <= limit for size, limit in zip(shape, max_tile_shape)) for shape in tile_shapes)
        assert self.tiling_settings(pipe.vae) == previous_settings

    def test_disable_tiling(self, pipe):
        inputs = self.get_inputs(pipe)
        tile_shapes = []
        handle = pipe.vae.decoder.register_forward_pre_hook(
            lambda module, args: tile_shapes.append(tuple(args[0].shape[-3:]))
        )
        try:
            video = pipe(**inputs, vae_tiling=False, output="videos")
        finally:
            handle.remove()
        assert tile_shapes == [(37, 4, 6)]
        assert video.shape == (1, 73, 3, 8, 12)
        assert torch.isfinite(video).all()

    def test_restore_settings_after_decode_error(self, pipe):
        # A one-latent-pixel spatial axis cannot be reflection-padded by the real decoder.
        inputs = self.get_inputs(pipe, height=2)
        previous_settings = self.tiling_settings(pipe.vae)
        with pytest.raises(RuntimeError):
            pipe(**inputs, output="videos")
        assert self.tiling_settings(pipe.vae) == previous_settings

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vae_tile_size": 0},
            {"vae_tile_overlap": 512},
            {"vae_temporal_tile_size": 3},
            {"vae_temporal_tile_overlap": -2},
        ],
    )
    def test_invalid_tiling_parameters(self, pipe, kwargs):
        with pytest.raises(ValueError, match="VAE .* tile size"):
            pipe(**self.get_inputs(pipe), **kwargs, output="videos")
