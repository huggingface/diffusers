# coding=utf-8
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
from PIL import Image

from diffusers import ModularPipeline, UniPCMultistepScheduler
from diffusers.modular_pipelines import (
    Cosmos3OmniBlocks,
    Cosmos3OmniModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.cosmos.before_denoise import (
    Cosmos3ActionDenoiseInputStep,
    Cosmos3ActionPackSequenceStep,
    Cosmos3SetTimestepsStep,
    Cosmos3SoundDenoiseInputStep,
    Cosmos3VisionDenoiseInputStep,
    Cosmos3VisionPackSequenceStep,
)
from diffusers.modular_pipelines.cosmos.encoders import Cosmos3TextEncoderStep

from ...testing_utils import torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


TEXT_VISION_WORKFLOW = [
    ("text_encoder", "Cosmos3TextEncoderStep"),
    ("denoise.prepare_text_segments", "Cosmos3PrepareTextSegmentsStep"),
    ("denoise.prepare_vision_latents", "Cosmos3VisionPrepareLatentsStep"),
    ("denoise.pack_vision_sequence", "Cosmos3VisionPackSequenceStep"),
    ("denoise.prepare_vision_denoiser_inputs", "Cosmos3VisionDenoiseInputStep"),
    ("denoise.set_timesteps", "Cosmos3SetTimestepsStep"),
    ("denoise.denoise", "Cosmos3VisionDenoiseStep"),
    ("decode.video", "Cosmos3VideoDecodeStep"),
    ("after_decode", "Cosmos3ActionOutputStep"),
]

IMAGE_VISION_WORKFLOW = [
    ("text_encoder", "Cosmos3TextEncoderStep"),
    ("vae_encoder", "Cosmos3ImageVaeEncoderStep"),
    *TEXT_VISION_WORKFLOW[1:],
]

VIDEO_VISION_WORKFLOW = [
    ("text_encoder", "Cosmos3TextEncoderStep"),
    ("vae_encoder", "Cosmos3VideoVaeEncoderStep"),
    *TEXT_VISION_WORKFLOW[1:],
]

TEXT_VISION_SOUND_WORKFLOW = [
    *TEXT_VISION_WORKFLOW[:6],
    ("denoise.prepare_sound_latents", "Cosmos3SoundPrepareLatentsStep"),
    ("denoise.pack_sound_sequence", "Cosmos3SoundPackSequenceStep"),
    ("denoise.prepare_sound_denoiser_inputs", "Cosmos3SoundDenoiseInputStep"),
    ("denoise.denoise", "Cosmos3VisionSoundDenoiseStep"),
    ("decode.video", "Cosmos3VideoDecodeStep"),
    ("decode.sound", "Cosmos3SoundDecodeStep"),
    ("after_decode", "Cosmos3ActionOutputStep"),
]

IMAGE_VISION_SOUND_WORKFLOW = [
    ("text_encoder", "Cosmos3TextEncoderStep"),
    ("vae_encoder", "Cosmos3ImageVaeEncoderStep"),
    *TEXT_VISION_SOUND_WORKFLOW[1:],
]

VIDEO_VISION_SOUND_WORKFLOW = [
    ("text_encoder", "Cosmos3TextEncoderStep"),
    ("vae_encoder", "Cosmos3VideoVaeEncoderStep"),
    *TEXT_VISION_SOUND_WORKFLOW[1:],
]

ACTION_WORKFLOW = [
    ("text_encoder", "Cosmos3ActionTextStep"),
    ("vae_encoder", "Cosmos3ActionVisionVaeEncoderStep"),
    ("denoise.prepare_text_segments", "Cosmos3PrepareTextSegmentsStep"),
    ("denoise.prepare_vision_latents", "Cosmos3VisionPrepareLatentsStep"),
    ("denoise.pack_vision_sequence", "Cosmos3VisionPackSequenceStep"),
    ("denoise.prepare_vision_denoiser_inputs", "Cosmos3VisionDenoiseInputStep"),
    ("denoise.set_timesteps", "Cosmos3SetTimestepsStep"),
    ("denoise.prepare_action_latents", "Cosmos3ActionPrepareLatentsStep"),
    ("denoise.pack_action_sequence", "Cosmos3ActionPackSequenceStep"),
    ("denoise.prepare_action_denoiser_inputs", "Cosmos3ActionDenoiseInputStep"),
    ("denoise.denoise", "Cosmos3VisionActionDenoiseStep"),
    ("decode.video", "Cosmos3VideoDecodeStep"),
    ("after_decode", "Cosmos3ActionOutputStep"),
]

COSMOS3_OMNI_WORKFLOWS = {
    "text2image": TEXT_VISION_WORKFLOW,
    "text2video": TEXT_VISION_WORKFLOW,
    "image2video": IMAGE_VISION_WORKFLOW,
    "video2video": VIDEO_VISION_WORKFLOW,
    "text2video_with_sound": TEXT_VISION_SOUND_WORKFLOW,
    "image2video_with_sound": IMAGE_VISION_SOUND_WORKFLOW,
    "video2video_with_sound": VIDEO_VISION_SOUND_WORKFLOW,
    "action_policy": ACTION_WORKFLOW,
    "action_forward_dynamics": ACTION_WORKFLOW,
    "action_inverse_dynamics": ACTION_WORKFLOW,
}


class TestCosmos3OmniModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Cosmos3OmniModularPipeline
    pipeline_blocks_class = Cosmos3OmniBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-cosmos3-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames", "guidance_scale"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "output_type"])
    output_name = "videos"
    expected_workflow_blocks = COSMOS3_OMNI_WORKFLOWS

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipe = super().get_pipeline(components_manager, torch_dtype)
        pipe.disable_safety_checker()
        return pipe

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "A small robot moves across a table.",
            "negative_prompt": "",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "guidance_scale": 2.0,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "output_type": "latent",
        }

    @pytest.mark.skip(reason="Cosmos3 does not support batched prompts.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Cosmos3 does not support batched prompts.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="Cosmos3 does not support multiple videos per prompt.")
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip(reason="Cosmos3 checkpoints support bfloat16, not float16, inference.")
    def test_float16_inference(self):
        pass

    def test_save_from_pretrained(self, tmp_path):
        base_pipe = self.get_pipeline().to(torch_device)
        base_pipe.save_pretrained(str(tmp_path))

        loaded_pipe = ModularPipeline.from_pretrained(str(tmp_path))
        loaded_pipe.load_components(torch_dtype=torch.float32)
        loaded_pipe.disable_safety_checker()
        loaded_pipe.to(torch_device)

        base_output = base_pipe(**self.get_dummy_inputs(), output=self.output_name)
        loaded_output = loaded_pipe(**self.get_dummy_inputs(), output=self.output_name)

        assert torch.abs(base_output - loaded_output).max() < 1e-3

    def test_vae_encoder_is_standalone_and_validates_conditioning_inputs(self):
        pipe = self.get_pipeline()
        vae_encoder = pipe.blocks.sub_blocks["vae_encoder"]

        assert vae_encoder.select_block(action=None, image=None, video=None) is None
        assert vae_encoder.select_block(action=None, image=object(), video=None) == "image_conditioning"
        assert vae_encoder.select_block(action=None, image=None, video=object()) == "video_conditioning"
        assert vae_encoder.select_block(action=object(), image=None, video=None) == "action_conditioning"

        vae_pipe = vae_encoder.init_pipeline(self.pretrained_model_name_or_path)
        vae_pipe.load_components(torch_dtype=torch.float32)
        outputs = vae_pipe(
            image=Image.new("RGB", (32, 32)),
            num_frames=5,
            height=32,
            width=32,
            output=["x0_tokens_vision", "vision_condition_frames"],
        )

        assert outputs["x0_tokens_vision"] is not None
        assert outputs["vision_condition_frames"] == [0]

        action_vae_encoder = vae_encoder.sub_blocks["action_conditioning"]
        assert [input_param.name for input_param in action_vae_encoder.inputs] == ["action"]
        assert [output_param.name for output_param in action_vae_encoder.intermediate_outputs] == [
            "x0_tokens_vision",
            "vision_condition_frames",
            "action_condition_frame_indexes",
        ]

        with pytest.raises(ValueError, match="not top-level image/video"):
            pipe.blocks.get_execution_blocks(action=object(), image=object())
        with pytest.raises(ValueError, match="not top-level image/video"):
            pipe.blocks.get_execution_blocks(action=object(), video=object())
        with pytest.raises(ValueError, match="either image or video"):
            pipe.blocks.get_execution_blocks(image=object(), video=object())

        inputs = self.get_dummy_inputs()
        inputs.update(image=Image.new("RGB", (32, 32)), num_frames=1)
        with pytest.raises(ValueError, match="image-to-image generation is not supported"):
            pipe(**inputs, output=self.output_name)

    @pytest.mark.parametrize("prompt_name", ["prompt", "negative_prompt"])
    def test_rejects_batched_prompts(self, prompt_name):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs[prompt_name] = ["first prompt", "second prompt"]

        with pytest.raises(ValueError, match="batched prompts are not supported"):
            pipe(**inputs, output=self.output_name)

    def test_pack_steps_do_not_require_vae(self):
        cond_text_segment = {"vision_start_temporal_offset": 0, "und_len": 2}
        uncond_text_segment = {"vision_start_temporal_offset": 0, "und_len": 1}

        vision_pipe = Cosmos3VisionPackSequenceStep().init_pipeline(self.pretrained_model_name_or_path)
        vision_pipe.load_components(torch_dtype=torch.float32)
        vision_segments = vision_pipe(
            cond_text_segment=cond_text_segment,
            uncond_text_segment=uncond_text_segment,
            latents=torch.zeros(1, vision_pipe.transformer.config.latent_channel, 2, 2, 2),
            fps_vision=5.0,
            vision_condition_indexes_for_pack=[],
            output=["cond_vision_segment", "uncond_vision_segment"],
        )

        assert set(vision_pipe.components) == {"transformer"}
        assert vision_segments["cond_vision_segment"]["vision_mrope_ids"].shape == (3, 2)
        assert vision_segments["uncond_vision_segment"]["vision_mrope_ids"].shape == (3, 2)

        action_pipe = Cosmos3ActionPackSequenceStep().init_pipeline(self.pretrained_model_name_or_path)
        action_pipe.load_components(torch_dtype=torch.float32)
        action_segments = action_pipe(
            cond_text_segment=cond_text_segment,
            uncond_text_segment=uncond_text_segment,
            cond_sequence_length=2,
            uncond_sequence_length=1,
            action_latents=torch.zeros(4, action_pipe.transformer.config.action_dim),
            action_condition_frame_indexes=[],
            fps_vision=5.0,
            output=["cond_action_segment", "uncond_action_segment"],
        )

        assert set(action_pipe.components) == {"transformer"}
        assert action_segments["cond_action_segment"]["action_mrope_ids"].shape == (3, 4)
        assert action_segments["uncond_action_segment"]["action_mrope_ids"].shape == (3, 4)

    def test_denoise_input_steps_assemble_modality_segments(self):
        cond_text_segment = {"text_mrope_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]), "und_len": 2}
        uncond_text_segment = {"text_mrope_ids": torch.tensor([[7], [8], [9]]), "und_len": 1}
        cond_vision_segment = {"vision_mrope_ids": torch.tensor([[10], [11], [12]]), "num_vision_tokens": 1}
        uncond_vision_segment = {
            "vision_mrope_ids": torch.tensor([[13, 14], [15, 16], [17, 18]]),
            "num_vision_tokens": 2,
        }
        cond_sound_segment = {
            "sound_mrope_ids": torch.tensor([[19, 20], [21, 22], [23, 24]]),
            "sound_len": 2,
        }
        uncond_sound_segment = {"sound_mrope_ids": torch.tensor([[25], [26], [27]]), "sound_len": 1}
        cond_action_segment = {"action_mrope_ids": torch.tensor([[28], [29], [30]]), "action_len": 1}
        uncond_action_segment = {
            "action_mrope_ids": torch.tensor([[31, 32], [33, 34], [35, 36]]),
            "action_len": 2,
        }

        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {
                "vision": Cosmos3VisionDenoiseInputStep(),
                "sound": Cosmos3SoundDenoiseInputStep(),
                "action": Cosmos3ActionDenoiseInputStep(),
            }
        )
        pipe = blocks.init_pipeline()
        state = pipe(
            cond_text_segment=cond_text_segment,
            uncond_text_segment=uncond_text_segment,
            cond_vision_segment=cond_vision_segment,
            uncond_vision_segment=uncond_vision_segment,
            cond_sound_segment=cond_sound_segment,
            uncond_sound_segment=uncond_sound_segment,
            cond_action_segment=cond_action_segment,
            uncond_action_segment=uncond_action_segment,
        )

        torch.testing.assert_close(
            state.get("cond_position_ids"),
            torch.cat(
                [
                    cond_text_segment["text_mrope_ids"],
                    cond_vision_segment["vision_mrope_ids"],
                    cond_sound_segment["sound_mrope_ids"],
                    cond_action_segment["action_mrope_ids"],
                ],
                dim=1,
            ),
        )
        torch.testing.assert_close(
            state.get("uncond_position_ids"),
            torch.cat(
                [
                    uncond_text_segment["text_mrope_ids"],
                    uncond_vision_segment["vision_mrope_ids"],
                    uncond_sound_segment["sound_mrope_ids"],
                    uncond_action_segment["action_mrope_ids"],
                ],
                dim=1,
            ),
        )
        assert state.get("cond_sequence_length") == 6
        assert state.get("uncond_sequence_length") == 6
        assert state.get("cond_packed_static") is None
        assert state.get("uncond_packed_static") is None

    def test_text_step_uses_pipeline_system_prompt_and_safety_configs(self):
        text_pipe = Cosmos3TextEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        text_pipe.load_components()
        text_pipe.disable_safety_checker()
        assert text_pipe.config.default_use_system_prompt
        assert not text_pipe.requires_safety_checker

        inputs = {
            "prompt": "A small robot moves across a table.",
            "negative_prompt": "",
            "num_frames": 5,
            "height": 32,
            "width": 32,
        }
        default_with_system_prompt = text_pipe(**inputs, output="cond_input_ids")
        explicit_with_system_prompt = text_pipe(**inputs, use_system_prompt=True, output="cond_input_ids")
        explicit_without_system_prompt = text_pipe(**inputs, use_system_prompt=False, output="cond_input_ids")

        text_pipe.update_components(default_use_system_prompt=False)
        assert not text_pipe.config.default_use_system_prompt

        default_without_system_prompt = text_pipe(**inputs, output="cond_input_ids")
        updated_with_system_prompt = text_pipe(**inputs, use_system_prompt=True, output="cond_input_ids")
        updated_without_system_prompt = text_pipe(**inputs, use_system_prompt=False, output="cond_input_ids")

        assert default_with_system_prompt == explicit_with_system_prompt == updated_with_system_prompt
        assert explicit_without_system_prompt == default_without_system_prompt == updated_without_system_prompt
        assert len(default_with_system_prompt) > len(default_without_system_prompt)

    def test_set_timesteps_native_flow_schedule(self):
        timesteps_pipe = Cosmos3SetTimestepsStep().init_pipeline(self.pretrained_model_name_or_path)
        timesteps_pipe.load_components()
        assert not timesteps_pipe.config.use_native_flow_schedule

        default_timesteps = timesteps_pipe(num_inference_steps=4, output="timesteps")

        timesteps_pipe.update_components(
            scheduler=UniPCMultistepScheduler(num_train_timesteps=100, use_flow_sigmas=True),
            use_native_flow_schedule=True,
        )
        native_timesteps = timesteps_pipe(num_inference_steps=4, output="timesteps")

        expected_sigmas = torch.tensor([0.99, 0.7425, 0.495, 0.2475])
        torch.testing.assert_close(timesteps_pipe.scheduler.sigmas[:-1], expected_sigmas)
        assert native_timesteps.tolist() == [99, 74, 49, 24]
        assert not torch.equal(native_timesteps, default_timesteps)
