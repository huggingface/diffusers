import gc
import tempfile
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
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
    SDXLSingleFileTesterMixin,
    download_diffusers_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLControlNetPipelineSingleFileSlowTests(unittest.TestCase, SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLControlNetPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

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
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )
        inputs = {
            "prompt": "Stormtrooper's lecture",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }

        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe_single_file.unet.set_default_attn_processor()
        pipe_single_file.enable_model_cpu_offload()
        pipe_single_file.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        single_file_images = pipe_single_file(**inputs).images[0]

        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        images = pipe(**inputs).images[0]

        assert images.shape == (512, 512, 3)
        assert single_file_images.shape == (512, 512, 3)

        max_diff = numpy_cosine_similarity_distance(images[0].flatten(), single_file_images[0].flatten())
        assert max_diff < 5e-2

    def test_single_file_components(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path, controlnet=controlnet)
        super().test_single_file_components(pipe, pipe_single_file)

    def test_single_file_components_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)

            single_file_pipe = self.pipeline_class.from_single_file(
                local_ckpt_path, controlnet=controlnet, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_original_config(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path,
            original_config=self.original_config,
            controlnet=controlnet,
        )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_original_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                safety_checker=None,
                controlnet=controlnet,
                local_files_only=True,
            )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, config=self.repo_id
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            controlnet=controlnet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)
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
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
        )
        single_file_pipe = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        super().test_single_file_setting_pipeline_dtype_to_fp16(single_file_pipe)
