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


import PIL
import torch

from diffusers import ZImageControlNetModel
from diffusers.modular_pipelines import ZImageAutoBlocks, ZImageModularPipeline
from diffusers.modular_pipelines.z_image.before_denoise import (
    ZImageControlNetBeforeDenoiserStep,
)

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


ZIMAGE_WORKFLOWS = {
    "text2image": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("denoise.input", "ZImageTextInputStep"),
        ("denoise.prepare_latents", "ZImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise.denoise", "ZImageDenoiseStep"),
        ("decode", "ZImageVaeDecoderStep"),
    ],
    "image2image": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("vae_encoder", "ZImageVaeImageEncoderStep"),
        ("denoise.input", "ZImageTextInputStep"),
        ("denoise.additional_inputs", "ZImageAdditionalInputsStep"),
        ("denoise.prepare_latents", "ZImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise.set_timesteps_with_strength", "ZImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_latents_with_image", "ZImagePrepareLatentswithImageStep"),
        ("denoise.denoise", "ZImageDenoiseStep"),
        ("decode", "ZImageVaeDecoderStep"),
    ],
    "inpainting": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("vae_encoder", "ZImageInpaintVaeImageEncoderStep"),
        ("denoise.input", "ZImageTextInputStep"),
        ("denoise.inpaint_input", "ZImageInpaintInputStep"),
        ("denoise.prepare_latents", "ZImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise.set_timesteps_with_strength", "ZImageSetTimestepsWithStrengthStep"),
        ("denoise.prepare_inpaint_latents", "ZImagePrepareInpaintLatentsStep"),
        ("denoise.denoise", "ZImageInpaintDenoiseStep"),
        ("decode.decode", "ZImageVaeDecoderStep"),
        ("decode.mask_overlay", "ZImageInpaintOverlayMaskStep"),
    ],
    "controlnet_inpainting": [
        ("text_encoder", "ZImageTextEncoderStep"),
        ("vae_encoder", "ZImageInpaintVaeImageEncoderStep"),
        ("controlnet_vae_encoder", "ZImageControlNetInpaintVaeEncoderStep"),
        ("denoise.input", "ZImageTextInputStep"),
        ("denoise.controlnet_input", "ZImageControlNetInpaintInputStep"),
        ("denoise.prepare_latents", "ZImagePrepareLatentsStep"),
        ("denoise.set_timesteps", "ZImageSetTimestepsStep"),
        ("denoise.controlnet_before_denoiser", "ZImageControlNetBeforeDenoiserStep"),
        ("denoise.denoise", "ZImageControlNetDenoiseStep"),
        ("decode.decode", "ZImageVaeDecoderStep"),
        ("decode.mask_overlay", "ZImageInpaintOverlayMaskStep"),
    ],
}


class ZImageModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = ZImageModularPipeline
    pipeline_blocks_class = ZImageAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-zimage-modular-pipe"
    params = frozenset(["prompt", "height", "width"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = ZIMAGE_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs


class TestZImageModularPipelineFast(ZImageModularPipelineTesterConfig, ModularPipelineTesterMixin):
    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-3)

    def test_inpaint_inference(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update(
            image=PIL.Image.new("RGB", (32, 32), 0),
            mask_image=PIL.Image.new("L", (32, 32), 255),
            strength=1.0,
        )
        output = pipe(**inputs, output="images")
        assert output.shape == (1, 3, 32, 32)

    def test_inpaint_padding_mask_crop(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update(
            image=PIL.Image.new("RGB", (64, 32), (255, 0, 0)),
            mask_image=PIL.Image.new("L", (64, 32), 0),
            height=32,
            width=32,
            padding_mask_crop=0,
            strength=1.0,
            output_type="pil",
        )
        inputs["mask_image"].paste(255, (16, 0, 48, 32))
        output = pipe(**inputs, output="images")
        assert output[0].size == (64, 32)
        assert output[0].getpixel((0, 16)) == (255, 0, 0)

    def test_controlnet_guidance_window(self):
        pipe = ZImageControlNetBeforeDenoiserStep().init_pipeline()
        pipe.load_components()
        output = pipe(
            timesteps=torch.tensor([3.0, 2.0, 1.0, 0.0]),
            control_guidance_start=0.25,
            control_guidance_end=0.75,
            output="controlnet_keep",
        )
        assert output == [0.0, 1.0, 1.0, 0.0]

    def test_controlnet_inpaint_inference(self):
        pipe = self.get_pipeline()
        transformer_config = pipe.transformer.config
        pipe.update_components(
            controlnet=ZImageControlNetModel(
                control_layers_places=[0],
                control_refiner_layers_places=[],
                control_in_dim=33,
                all_patch_size=transformer_config.all_patch_size,
                all_f_patch_size=transformer_config.all_f_patch_size,
                dim=transformer_config.dim,
                n_refiner_layers=transformer_config.n_refiner_layers,
                n_heads=transformer_config.n_heads,
                n_kv_heads=transformer_config.n_kv_heads,
                norm_eps=transformer_config.norm_eps,
                qk_norm=transformer_config.qk_norm,
            )
        )
        assert pipe.controlnet.t_embedder is pipe.transformer.t_embedder
        inputs = self.get_dummy_inputs()
        inputs.update(
            image=PIL.Image.new("RGB", (64, 32), (255, 0, 0)),
            mask_image=PIL.Image.new("L", (64, 32), 0),
            control_image=PIL.Image.new("RGB", (64, 32), 0),
            height=32,
            width=32,
            padding_mask_crop=0,
            control_guidance_start=0.25,
            control_guidance_end=0.75,
            output_type="pil",
        )
        inputs["mask_image"].paste(255, (16, 0, 48, 32))
        output = pipe(**inputs, output="images")
        assert output[0].size == (64, 32)
        assert output[0].getpixel((0, 16)) == (255, 0, 0)


class TestZImageModularPipelineLoading(ZImageModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestZImageModularPipelineWorkflow(ZImageModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestZImageModularPipelineMemory(ZImageModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
