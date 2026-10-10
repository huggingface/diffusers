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

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    TripoSplatAutoBlocks,
    TripoSplatImageProcessor,
    TripoSplatPipeline,
)
from diffusers.utils import export_to_gaussian_ply, export_to_splat

from ...testing_utils import assert_tensors_close
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin
from .testing_utils import get_triposplat_dummy_components


class TripoSplatPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = TripoSplatPipeline
    required_input_params_in_call_signature = frozenset(["image", "guidance_scale", "num_gaussians"])
    batch_input_params = frozenset(["image"])
    output_shape = (32768, 14)

    def get_dummy_components(self):
        return get_triposplat_dummy_components()

    def get_dummy_inputs(self):
        return {
            "image": Image.new("RGB", (32, 32), (128, 64, 192)),
            "is_preprocessed": True,
            "num_inference_steps": 2,
            "num_gaussians": 32768,
            "guidance_scale": 3.0,
            "generator": self.get_generator(0),
            "output_type": "pt",
        }


class TestTripoSplatPipeline(TripoSplatPipelineTesterConfig, PipelineTesterMixin):
    def test_gaussian_parameters_and_exports(self, tmp_path):
        pipe = self.get_pipeline()
        output = pipe(**self.get_dummy_inputs())
        gaussians = output.gaussians[0]
        assert gaussians.shape == self.output_shape
        assert gaussians.isfinite().all()
        assert (gaussians[:, 6:9] > 0).all()
        assert ((gaussians[:, 13] >= 0) & (gaussians[:, 13] <= 1)).all()
        export_to_gaussian_ply(gaussians, tmp_path / "object.ply")
        export_to_splat(gaussians, tmp_path / "object.splat")
        assert (tmp_path / "object.ply").read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
        assert (tmp_path / "object.splat").stat().st_size == 32 * gaussians.shape[0]

    def test_rgba_and_optional_background_remover(self):
        processor = TripoSplatImageProcessor(canvas_size=32)
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        image.paste((255, 128, 64, 255), (8, 8, 24, 24))
        prepared = processor.prepare_foreground(image)
        assert prepared[0].mode == "RGB"
        assert prepared[0].size == (32, 32)
        with pytest.raises(ValueError, match="require background_remover"):
            self.get_pipeline().prepare_image(Image.new("RGB", (32, 32)))
        with pytest.raises(ValueError, match="empty"):
            processor.prepare_foreground(Image.new("RGBA", (32, 32)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_flow_match_schedule_and_float32_latents(self, dtype):
        pipe = self.get_pipeline()
        pipe.transformer.to(dtype=dtype)
        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        inputs["num_inference_steps"] = 20

        def callback(pipeline, index, timestep, tensors):
            assert tensors["latents"].dtype == torch.float32
            assert tensors["camera_latents"].dtype == torch.float32
            return tensors

        inputs["callback_on_step_end"] = callback
        inputs["callback_on_step_end_tensor_inputs"] = ["latents", "camera_latents"]
        output = pipe(**inputs)
        assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler)
        sigmas = np.linspace(1.0, 0.0, 21)
        sigmas = 3.0 * sigmas / (1.0 + 2.0 * sigmas)
        np.testing.assert_allclose(pipe.scheduler.sigmas.cpu().numpy(), sigmas, rtol=1e-6, atol=1e-7)
        assert output.latents.shape == (1, pipe.q_token_length, pipe.latent_channels)
        assert output.camera_latents.shape == (1, 1, pipe.camera_channels)
        assert output.latents.dtype == torch.float32
        assert output.camera_latents.dtype == torch.float32
        assert output.latents.isfinite().all()
        assert output.camera_latents.isfinite().all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_encode_image_restores_shared_rope_buffer(self, dtype):
        pipe = self.get_pipeline()
        pipe.image_encoder.to(dtype=dtype)
        original = pipe.image_encoder.rope_embeddings.inv_freq.float()
        pipe.image_encoder.rope_embeddings.inv_freq = original
        images = [Image.new("RGB", (32, 32), (128, 64, 192))]
        actual = pipe.encode_image(images, pipe._execution_device)
        assert actual.isfinite().all()
        assert actual.requires_grad
        assert pipe.image_encoder.rope_embeddings.inv_freq is original
        assert original.dtype == torch.float32

    def test_multiple_densities_denoise_once(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        steps = []

        def callback(pipeline, index, timestep, tensors):
            steps.append(index)
            return tensors

        inputs.update(num_gaussians=[32768, 65536], callback_on_step_end=callback)
        output = pipe(**inputs)
        assert len(steps) == inputs["num_inference_steps"]
        assert [item.shape[1] for item in output.gaussians] == [32768, 65536]

    @pytest.mark.parametrize("batch_size,num_images_per_prompt", [(1, 1), (2, 2)])
    def test_standard_modular_latent_parity(self, batch_size, num_images_per_prompt):
        components = self.get_dummy_components()
        standard = TripoSplatPipeline(**components)
        blocks = TripoSplatAutoBlocks()
        blocks.sub_blocks.pop("decode")
        modular = blocks.init_pipeline()
        modular.update_components(
            **{key: value for key, value in components.items() if key != "canvas_size"},
            image_processor=TripoSplatImageProcessor(canvas_size=32),
        )

        def make_inputs():
            inputs = self.get_dummy_inputs()
            inputs["num_images_per_prompt"] = num_images_per_prompt
            if batch_size > 1:
                inputs["image"] = [inputs["image"]] * batch_size
                inputs["generator"] = [self.get_generator(seed) for seed in range(batch_size * num_images_per_prompt)]
            return inputs

        inputs = make_inputs()
        inputs["output_type"] = "latent"
        expected = standard(**inputs)
        inputs = make_inputs()
        for key in ("guidance_scale", "num_gaussians", "decoder_generator", "output_type"):
            inputs.pop(key, None)
        actual = modular(**inputs)
        assert_tensors_close(actual.get("latents"), expected.latents, atol=1e-5, rtol=1e-5)
        assert_tensors_close(actual.get("camera_latents"), expected.camera_latents, atol=1e-5, rtol=1e-5)


class TestTripoSplatPipelineMemory(TripoSplatPipelineTesterConfig, MemoryTesterMixin):
    pass
