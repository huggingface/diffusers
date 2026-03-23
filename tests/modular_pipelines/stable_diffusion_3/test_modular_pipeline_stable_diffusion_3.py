# coding=utf-8
# Copyright 2025 HuggingFace Inc.
import random
import numpy as np
import PIL
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ModularPipeline
from diffusers.modular_pipelines.stable_diffusion_3 import SD3AutoBlocks, SD3ModularPipeline

from ...testing_utils import floats_tensor, torch_device
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


SD3_TEXT2IMAGE_WORKFLOWS = {
    "text2image":[
        ("text_encoder", "SD3TextEncoderStep"),
        ("denoise.input", "SD3TextInputStep"),
        ("denoise.before_denoise.prepare_latents", "SD3PrepareLatentsStep"),
        ("denoise.before_denoise.set_timesteps", "SD3SetTimestepsStep"),
        ("denoise.denoise", "SD3DenoiseStep"),
        ("decode", "SD3DecodeStep"),
    ]
}

class TestSD3ModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = SD3ModularPipeline
    pipeline_blocks_class = SD3AutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-sd3-pipe"

    params = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_params = frozenset(["prompt"])
    expected_workflow_blocks = SD3_TEXT2IMAGE_WORKFLOWS

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "output_type": "pt",
        }

    def test_float16_inference(self):
        super().test_float16_inference(9e-2)


SD3_IMAGE2IMAGE_WORKFLOWS = {
    "image2image":[
        ("text_encoder", "SD3TextEncoderStep"),
        ("vae_encoder.preprocess", "SD3ProcessImagesInputStep"),
        ("vae_encoder.encode", "SD3VaeEncoderStep"),
        ("denoise.input.text_inputs", "SD3TextInputStep"),
        ("denoise.input.additional_inputs", "SD3AdditionalInputsStep"),
        ("denoise.before_denoise.prepare_latents", "SD3PrepareLatentsStep"),
        ("denoise.before_denoise.set_timesteps", "SD3Img2ImgSetTimestepsStep"),
        ("denoise.before_denoise.prepare_img2img_latents", "SD3Img2ImgPrepareLatentsStep"),
        ("denoise.denoise", "SD3DenoiseStep"),
        ("decode", "SD3DecodeStep"),
    ]
}

class TestSD3Img2ImgModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = SD3ModularPipeline
    pipeline_blocks_class = SD3AutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-sd3-pipe"

    params = frozenset(["prompt", "height", "width", "guidance_scale", "image"])
    batch_params = frozenset(["prompt", "image"])
    expected_workflow_blocks = SD3_IMAGE2IMAGE_WORKFLOWS

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = super().get_pipeline(components_manager, torch_dtype)
        pipeline.image_processor = VaeImageProcessor(vae_scale_factor=8)
        return pipeline

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 4,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "output_type": "pt",
        }
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = PIL.Image.fromarray(np.uint8(image)).convert("RGB")
        inputs["image"] = init_image
        inputs["strength"] = 0.5
        return inputs

    def test_save_from_pretrained(self, tmp_path):
        pipes =[]
        base_pipe = self.get_pipeline().to(torch_device)
        pipes.append(base_pipe)

        base_pipe.save_pretrained(str(tmp_path))
        pipe = ModularPipeline.from_pretrained(tmp_path).to(torch_device)
        pipe.load_components(torch_dtype=torch.float32)
        pipe.to(torch_device)
        pipe.image_processor = VaeImageProcessor(vae_scale_factor=8)
        pipes.append(pipe)

        image_slices =[]
        for pipe in pipes:
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs, output="images")
            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert torch.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_float16_inference(self):
        super().test_float16_inference(8e-2)