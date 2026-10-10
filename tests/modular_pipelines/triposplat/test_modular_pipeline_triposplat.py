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
from PIL import Image

from diffusers import (
    ComponentsManager,
    ModularPipeline,
    TripoSplatAutoBlocks,
    TripoSplatClassifierFreeGuidance,
    TripoSplatImageProcessor,
    TripoSplatModularPipeline,
    TripoSplatPipeline,
)

from ...pipelines.triposplat.testing_utils import get_triposplat_dummy_components
from ...testing_utils import require_accelerator, torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularGuiderTesterMixin,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
)


MODULAR_PROCESSOR_CONFIG_XFAIL = pytest.mark.xfail(
    strict=True, reason="ModularPipeline does not serialize updated from_config processor components."
)


class TripoSplatModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = TripoSplatModularPipeline
    pipeline_blocks_class = TripoSplatAutoBlocks
    params = frozenset(["image", "num_gaussians", "decoder_generator", "is_preprocessed"])
    batch_params = frozenset(["image"])
    output_name = "gaussians"
    expected_workflow_blocks = {
        None: [
            ("preprocess", "TripoSplatImagePreprocessStep"),
            ("image_encoder", "TripoSplatImageEncoderStep"),
            ("vae_encoder", "TripoSplatVaeEncoderStep"),
            ("denoise", "TripoSplatCoreDenoiseStep"),
            ("decode", "TripoSplatGaussianDecodeStep"),
        ]
    }

    @pytest.fixture(scope="class", autouse=True)
    def tiny_checkpoint(self, tmp_path_factory, request):
        path = tmp_path_factory.mktemp("triposplat")
        standard = TripoSplatPipeline(**get_triposplat_dummy_components())
        standard.save_pretrained(path)
        modular = TripoSplatAutoBlocks().init_pipeline(str(path))
        modular.update_components(**standard.components, image_processor=TripoSplatImageProcessor(canvas_size=32))
        modular.save_pretrained(str(path), overwrite_modular_index=True)
        request.cls._checkpoint_path = str(path)

    @property
    def pretrained_model_name_or_path(self):
        return self._checkpoint_path

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        pipe = super().get_pipeline(components_manager=components_manager, dtype=dtype)
        pipe.update_components(image_processor=TripoSplatImageProcessor(canvas_size=32))
        return pipe

    def get_dummy_inputs(self, seed=0):
        return {
            "image": Image.new("RGB", (32, 32), (128, 64, 192)),
            "is_preprocessed": True,
            "num_inference_steps": 2,
            "num_gaussians": 32768,
            "output_type": "pt",
            "generator": self.get_generator(seed),
        }


class TestTripoSplatModularPipeline(TripoSplatModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_default_blocks(self):
        blocks = self.pipeline_blocks_class()
        actual = [(name, type(block).__name__) for name, block in blocks.sub_blocks.items()]
        assert actual == self.expected_workflow_blocks[None]

    def test_float32_latents_with_half_precision_transformer(self):
        components = get_triposplat_dummy_components()
        components["transformer"].to(dtype=torch.float16)
        pipe = TripoSplatAutoBlocks().sub_blocks["denoise"].init_pipeline()
        pipe.update_components(transformer=components["transformer"], scheduler=components["scheduler"])
        pipe.to(torch_device)
        output = pipe(
            encoder_hidden_states=torch.randn(1, 9, 32, generator=self.get_generator(0), device=torch_device),
            image_latents=torch.randn(1, 9, 8, generator=self.get_generator(1), device=torch_device),
            generator=self.get_generator(2),
            num_inference_steps=2,
        )
        assert output.get("latents").dtype == torch.float32
        assert output.get("camera_latents").dtype == torch.float32
        assert output.get("latents").isfinite().all()
        assert output.get("camera_latents").isfinite().all()

    def test_standalone_decode(self):
        blocks = TripoSplatAutoBlocks()
        pipe = blocks.sub_blocks["decode"].init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components()
        latents = torch.randn(1, 16, 16, generator=self.get_generator(0))
        output = pipe(latents=latents, num_gaussians=32768, decoder_generator=self.get_generator(1))
        assert output.get("gaussians").shape == (1, 32768, 14)

    @pytest.mark.parametrize(
        "block_name,output_name", [("image_encoder", "encoder_hidden_states"), ("vae_encoder", "image_latents")]
    )
    def test_standalone_encoder(self, block_name, output_name):
        pipe = self.pipeline_blocks_class().sub_blocks[block_name].init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        pipe.update_components(image_processor=TripoSplatImageProcessor(canvas_size=32))
        pipe.to(torch_device)
        images = [self.get_dummy_inputs()["image"]]
        actual = pipe(preprocessed_images=images, generator=self.get_generator(0), output=output_name)
        standard = TripoSplatPipeline(**get_triposplat_dummy_components()).to(torch_device)
        if block_name == "image_encoder":
            expected = standard.encode_image(images, torch_device)
        else:
            expected = standard.encode_vae_image(images, torch_device, self.get_generator(0))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_standalone_preprocess(self):
        pipe = self.pipeline_blocks_class().sub_blocks["preprocess"].init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components()
        pipe.update_components(image_processor=TripoSplatImageProcessor(canvas_size=32))
        image = Image.new("RGBA", (16, 32), (0, 0, 0, 0))
        image.paste((128, 64, 192, 255), (4, 8, 12, 24))
        actual = pipe(image=image, erode_radius=0, output="preprocessed_images")
        expected = TripoSplatImageProcessor(canvas_size=32).prepare_foreground(image, erode_radius=0)
        assert len(actual) == len(expected) == 1
        assert actual[0].tobytes() == expected[0].tobytes()

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_standalone_prepare_latents(self, batch_size):
        block = self.pipeline_blocks_class().sub_blocks["denoise"].sub_blocks["prepare"]
        pipe = block.init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        pipe.to(torch_device)
        conditioning = torch.zeros(batch_size, 9, 32, device=torch_device)
        actual = pipe(
            encoder_hidden_states=conditioning, generator=[self.get_generator(index) for index in range(batch_size)]
        )
        standard = TripoSplatPipeline(**get_triposplat_dummy_components()).to(torch_device)
        expected_latents, expected_camera = standard.prepare_latents(
            batch_size,
            torch.float32,
            torch_device,
            generator=[self.get_generator(index) for index in range(batch_size)],
        )
        torch.testing.assert_close(actual.get("latents"), expected_latents, rtol=0, atol=0)
        torch.testing.assert_close(actual.get("camera_latents"), expected_camera, rtol=0, atol=0)
        reused = pipe(
            encoder_hidden_states=conditioning,
            latents=actual.get("latents"),
            camera_latents=actual.get("camera_latents"),
        )
        torch.testing.assert_close(reused.get("latents"), expected_latents, rtol=0, atol=0)
        torch.testing.assert_close(reused.get("camera_latents"), expected_camera, rtol=0, atol=0)


class TestTripoSplatModularLoading(TripoSplatModularPipelineTesterConfig, ModularLoadingTesterMixin):
    @MODULAR_PROCESSOR_CONFIG_XFAIL
    def test_save_from_pretrained(self, tmp_path, base_pipe_output):
        pipe = self.get_pipeline()
        pipe.save_pretrained(str(tmp_path), overwrite_modular_index=True)
        reloaded = ModularPipeline.from_pretrained(str(tmp_path))
        reloaded.load_components(dtype=torch.float32)
        assert reloaded.image_processor.config == pipe.image_processor.config
        reloaded.to(torch_device)
        actual = reloaded(**self.get_dummy_inputs(), output=self.output_name)
        torch.testing.assert_close(actual, base_pipe_output, atol=1e-5, rtol=1e-5)

    def test_optional_component_is_not_loaded(self):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        pipe.load_components()
        assert pipe.background_remover is None

    def test_base_pipeline_dispatch(self, base_pipe_output):
        pipe = ModularPipeline.from_pretrained(self.pretrained_model_name_or_path)
        assert isinstance(pipe, self.pipeline_class)
        pipe.load_components(dtype=torch.float32)
        pipe.update_components(image_processor=TripoSplatImageProcessor(canvas_size=32))
        pipe.to(torch_device)
        actual = pipe(**self.get_dummy_inputs(), output=self.output_name)
        torch.testing.assert_close(actual, base_pipe_output, atol=1e-5, rtol=1e-5)


class TestTripoSplatModularMemory(TripoSplatModularPipelineTesterConfig, ModularMemoryTesterMixin):
    @require_accelerator
    def test_components_auto_cpu_offload_inference_consistent(self, base_pipe_output):
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=torch_device)
        pipe = self.get_pipeline(components_manager=manager)
        actual = pipe(**self.get_dummy_inputs(), output=self.output_name)
        torch.testing.assert_close(actual, base_pipe_output, atol=1e-5, rtol=1e-5)


class TestTripoSplatModularGuider(TripoSplatModularPipelineTesterConfig, ModularGuiderTesterMixin):
    @pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 3.0])
    def test_reference_guidance_arithmetic(self, scale):
        guider = TripoSplatClassifierFreeGuidance(guidance_scale=scale)
        guider.set_state(step=0, num_inference_steps=20, timestep=torch.tensor(1000.0))
        conditional = torch.randn(2, 11, generator=torch.Generator().manual_seed(0)).half()
        unconditional = torch.randn(2, 11, generator=torch.Generator().manual_seed(1)).half()
        actual = guider.forward(conditional, unconditional).pred
        expected = scale * conditional - (scale - 1) * unconditional if scale > 1 else conditional
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
