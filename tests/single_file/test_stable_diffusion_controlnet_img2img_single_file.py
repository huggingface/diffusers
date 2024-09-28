import gc
import tempfile
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.loaders.single_file_utils import _extract_repo_id_and_weights_name
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from .single_file_testing_utils import (
    SDSingleFileTesterMixin,
    download_diffusers_config,
    download_original_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionControlNetPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionControlNetPipeline
    ckpt_path = (
        "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
    )
    original_config = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        control_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))
        prompt = "bird"

        inputs = {
            "prompt": prompt,
            "image": init_image,
            "control_image": control_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload()

        pipe_sf = self.pipeline_class.from_single_file(
            self.ckpt_path,
            controlnet=controlnet,
        )
        pipe_sf.unet.set_default_attn_processor()
        pipe_sf.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        output = pipe(**inputs).images[0]

        inputs = self.get_inputs(torch_device)
        output_sf = pipe_sf(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(output_sf.flatten(), output.flatten())
        assert max_diff < 1e-3

    def test_single_file_components(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, variant="fp16", safety_checker=None, controlnet=controlnet
        )
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path,
            safety_checker=None,
            controlnet=controlnet,
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weights_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weights_name, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path, controlnet=controlnet, safety_checker=None, local_files_only=True
            )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_original_config(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", variant="fp16")
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, safety_checker=None, original_config=self.original_config
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_original_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            controlnet=controlnet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weights_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weights_name, tmpdir)

            local_original_config = download_original_config(self.original_config, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                original_config=local_original_config,
                controlnet=controlnet,
                safety_checker=None,
                local_files_only=True,
            )
        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", variant="fp16")
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, safety_checker=None, original_config=self.original_config
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            controlnet=controlnet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weights_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weights_name, tmpdir)

            local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                config=local_diffusers_config,
                safety_checker=None,
                controlnet=controlnet,
                local_files_only=True,
            )
        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_setting_pipeline_dtype_to_fp16(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, variant="fp16"
        )
        single_file_pipe = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        super().test_single_file_setting_pipeline_dtype_to_fp16(single_file_pipe)
