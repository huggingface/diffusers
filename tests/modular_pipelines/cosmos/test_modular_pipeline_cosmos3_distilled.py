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

from types import SimpleNamespace

import pytest
import torch

from diffusers import ModularPipeline
from diffusers.modular_pipelines import Cosmos3DistilledBlocks, Cosmos3DistilledModularPipeline
from diffusers.modular_pipelines.cosmos.before_denoise import Cosmos3DistilledSetTimestepsStep
from diffusers.modular_pipelines.cosmos.encoders import Cosmos3DistilledTextEncoderStep
from diffusers.modular_pipelines.modular_pipeline import PipelineState

from ...testing_utils import torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


# TODO: move this fixture to `hf-internal-testing/tiny-cosmos3-distilled-modular-pipe` and update the
# repo name here. Hosted on a personal account for now so the PR can be tested.
TINY_DISTILLED_REPO = "yzhautouskay/tiny-cosmos3-distilled-modular-pipe"


# text2image / text2video: no visual conditioning, so the auto VAE encoder is skipped.
TEXT_DISTILLED_WORKFLOW = [
    ("text_encoder", "Cosmos3DistilledTextEncoderStep"),
    ("denoise.prepare_text_segments", "Cosmos3PrepareTextSegmentsStep"),
    ("denoise.prepare_vision_latents", "Cosmos3VisionPrepareLatentsStep"),
    ("denoise.pack_vision_sequence", "Cosmos3VisionPackSequenceStep"),
    ("denoise.prepare_vision_denoiser_inputs", "Cosmos3VisionDenoiseInputStep"),
    ("denoise.set_timesteps", "Cosmos3DistilledSetTimestepsStep"),
    ("denoise.denoise", "Cosmos3DistilledVisionDenoiseStep"),
    ("decode", "Cosmos3VideoDecodeStep"),
]

IMAGE_DISTILLED_WORKFLOW = [
    ("text_encoder", "Cosmos3DistilledTextEncoderStep"),
    ("vae_encoder", "Cosmos3ImageVaeEncoderStep"),
    *TEXT_DISTILLED_WORKFLOW[1:],
]

VIDEO_DISTILLED_WORKFLOW = [
    ("text_encoder", "Cosmos3DistilledTextEncoderStep"),
    ("vae_encoder", "Cosmos3VideoVaeEncoderStep"),
    *TEXT_DISTILLED_WORKFLOW[1:],
]

COSMOS3_DISTILLED_WORKFLOWS = {
    "text2image": TEXT_DISTILLED_WORKFLOW,
    "text2video": TEXT_DISTILLED_WORKFLOW,
    "image2video": IMAGE_DISTILLED_WORKFLOW,
    "video2video": VIDEO_DISTILLED_WORKFLOW,
}


class TestCosmos3DistilledModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Cosmos3DistilledModularPipeline
    pipeline_blocks_class = Cosmos3DistilledBlocks
    pretrained_model_name_or_path = TINY_DISTILLED_REPO

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "output_type"])
    output_name = "videos"
    expected_workflow_blocks = COSMOS3_DISTILLED_WORKFLOWS

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipe = super().get_pipeline(components_manager, torch_dtype)
        pipe.disable_safety_checker()
        return pipe

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "A small robot moves across a table.",
            "generator": self.get_generator(seed),
            "num_inference_steps": 4,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "output_type": "latent",
        }

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


def _fake_distilled_components(sigmas=(1.0, 0.9375, 0.8333333333333334, 0.625)):
    config = SimpleNamespace(distilled_sigmas=list(sigmas))
    return SimpleNamespace(_execution_device="cpu", config=config)


def test_cosmos3_distilled_vae_encoder_select_block():
    vae_encoder = Cosmos3DistilledBlocks().sub_blocks["vae_encoder"]
    assert vae_encoder.select_block(image=None, video=None) is None
    assert vae_encoder.select_block(image=object(), video=None) == "image_conditioning"
    assert vae_encoder.select_block(image=None, video=object()) == "video_conditioning"
    with pytest.raises(ValueError, match="either image or video"):
        vae_encoder.select_block(image=object(), video=object())


def test_cosmos3_distilled_set_timesteps_declares_distilled_configs():
    configs = {spec.name: spec.default for spec in Cosmos3DistilledSetTimestepsStep().expected_configs}
    assert configs == {"is_distilled": True, "distilled_sigmas": None}


def test_cosmos3_distilled_text_encoder_omits_negative_prompt():
    input_names = {inp.name for inp in Cosmos3DistilledTextEncoderStep().inputs}
    assert "prompt" in input_names
    assert "negative_prompt" not in input_names


def test_cosmos3_distilled_text_encoder_requires_str_prompt():
    Cosmos3DistilledTextEncoderStep._check_inputs(SimpleNamespace(prompt="a robot"))
    with pytest.raises(ValueError, match="`prompt` must be a str"):
        Cosmos3DistilledTextEncoderStep._check_inputs(SimpleNamespace(prompt=["a robot", "another"]))


def test_cosmos3_distilled_set_timesteps_rejects_step_count_override():
    step = Cosmos3DistilledSetTimestepsStep()
    state = PipelineState()
    state.set("num_inference_steps", 10)
    with pytest.raises(ValueError, match="must be 4 or left unset"):
        step(_fake_distilled_components(), state)


def test_cosmos3_distilled_set_timesteps_rejects_guidance_override():
    step = Cosmos3DistilledSetTimestepsStep()
    state = PipelineState()
    state.set("guidance_scale", 3.0)
    with pytest.raises(ValueError, match="`guidance_scale` must be 1.0"):
        step(_fake_distilled_components(), state)


def test_cosmos3_distilled_set_timesteps_requires_distilled_sigmas():
    step = Cosmos3DistilledSetTimestepsStep()
    components = SimpleNamespace(_execution_device="cpu", config=SimpleNamespace(distilled_sigmas=None))
    with pytest.raises(ValueError, match="distilled_sigmas"):
        step(components, PipelineState())
