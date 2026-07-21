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

from diffusers import ModularPipeline
from diffusers.modular_pipelines import Cosmos3DistilledBlocks, Cosmos3DistilledModularPipeline

from ...testing_utils import torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


TINY_DISTILLED_REPO = "hf-internal-testing/tiny-cosmos3-distilled-modular-pipe"


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

    def test_declares_distilled_configs(self):
        pipe = self.pipeline_class()
        assert pipe.config.is_distilled is True
        assert pipe.config.distilled_sigmas is None

    def test_vae_encoder_rejects_image_and_video_together(self):
        vae_encoder = Cosmos3DistilledBlocks().sub_blocks["vae_encoder"]
        vae_pipe = vae_encoder.init_pipeline(self.pretrained_model_name_or_path)
        vae_pipe.load_components(torch_dtype=torch.float32)

        image = Image.new("RGB", (32, 32))
        with pytest.raises(ValueError, match="either image or video"):
            vae_pipe(image=image, video=[image], num_frames=5, height=32, width=32)

    def test_rejects_batched_prompts(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = ["a robot", "another"]

        with pytest.raises(ValueError, match="batched prompts are not supported"):
            pipe(**inputs, output=self.output_name)

    def test_rejects_num_inference_steps_override(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 10

        with pytest.raises(ValueError, match="must be 4 or left unset"):
            pipe(**inputs, output=self.output_name)

    def test_rejects_guidance_scale_override(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["guidance_scale"] = 3.0

        with pytest.raises(ValueError, match="`guidance_scale` must be 1.0"):
            pipe(**inputs, output=self.output_name)
