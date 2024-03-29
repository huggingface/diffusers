import gc
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLControlNetPipelineSingleFileSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_file_format_inference_is_same(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
        single_file_url = (
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
        )
        pipe_single_file = StableDiffusionXLControlNetPipeline.from_single_file(
            single_file_url, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe_single_file.unet.set_default_attn_processor()
        pipe_single_file.enable_model_cpu_offload()
        pipe_single_file.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )
        single_file_images = pipe_single_file(
            prompt, image=image, generator=generator, output_type="np", num_inference_steps=2
        ).images

        generator = torch.Generator(device="cpu").manual_seed(0)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload()
        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=2).images

        assert images[0].shape == (512, 512, 3)
        assert single_file_images[0].shape == (512, 512, 3)

        max_diff = numpy_cosine_similarity_distance(images[0].flatten(), single_file_images[0].flatten())
        assert max_diff < 5e-2

    def test_single_file_component_configs(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        single_file_url = (
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
        )
        single_file_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            single_file_url, controlnet=controlnet, torch_dtype=torch.float16
        )

        for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder.config.to_dict()[param_name] == param_value

        for param_name, param_value in single_file_pipe.text_encoder_2.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder_2.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = [
            "torch_dtype",
            "_name_or_path",
            "architectures",
            "_use_default_values",
            "_diffusers_version",
        ]
        for param_name, param_value in single_file_pipe.unet.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.unet.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

        for param_name, param_value in single_file_pipe.vae.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.vae.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

