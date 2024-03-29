import gc
import unittest

import torch

from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)
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
class StableDiffusionXLAdapterPipelineSingleFileSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_file_format_inference_is_same(self):
        ckpt_path = (
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
        )
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16)
        prompt = "toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png"
        )
        pipe_single_file = StableDiffusionXLAdapterPipeline.from_single_file(
            ckpt_path,
            adapter=adapter,
            torch_dtype=torch.float16,
        )
        pipe_single_file.enable_model_cpu_offload()
        pipe_single_file.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        images_single_file = pipe_single_file(
            prompt, image=image, generator=generator, output_type="np", num_inference_steps=3
        ).images

        generator = torch.Generator(device="cpu").manual_seed(0)
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            adapter=adapter,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images_single_file[0].shape == (768, 512, 3)
        assert images[0].shape == (768, 512, 3)

        max_diff = numpy_cosine_similarity_distance(images[0].flatten(), images_single_file[0].flatten())
        assert max_diff < 5e-3

    def test_single_file_component_configs(self):
        ckpt_path = (
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
        )
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16)

        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16
        )
        single_file_pipe = StableDiffusionXLAdapterPipeline.from_single_file(
            ckpt_path,
            adapter=adapter,
            torch_dtype=torch.float16,
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

