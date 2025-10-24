import gc
import tempfile

import pytest
import torch

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.loaders.single_file_utils import _extract_repo_id_and_weights_name
from diffusers.utils import load_image

from ..testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
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
@require_torch_accelerator
class TestStableDiffusionControlNetInpaintPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionControlNetInpaintPipeline
    ckpt_path = "https://huggingface.co/botp/stable-diffusion-v1-5-inpainting/blob/main/sd-v1-5-inpainting.ckpt"
    original_config = "https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inpainting-inference.yaml"
    repo_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self):
        control_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))
        image = load_image(
            "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
        ).resize((512, 512))
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        ).resize((512, 512))

        inputs = {
            "prompt": "bird",
            "image": image,
            "control_image": control_image,
            "mask_image": mask_image,
            "generator": torch.Generator(device="cpu").manual_seed(0),
            "num_inference_steps": 3,
            "output_type": "np",
        }

        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet, safety_checker=None)
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload(device=torch_device)

        pipe_sf = self.pipeline_class.from_single_file(self.ckpt_path, controlnet=controlnet, safety_checker=None)
        pipe_sf.unet.set_default_attn_processor()
        pipe_sf.enable_model_cpu_offload(device=torch_device)

        inputs = self.get_inputs()
        output = pipe(**inputs).images[0]

        inputs = self.get_inputs()
        output_sf = pipe_sf(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(output_sf.flatten(), output.flatten())
        assert max_diff < 2e-3

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
        pipe = self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None, controlnet=controlnet)

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path, controlnet=controlnet, safety_checker=None, local_files_only=True
            )

        super()._compare_component_configs(pipe, pipe_single_file)

    @pytest.mark.skip(reason="runwayml original config repo does not exist")
    def test_single_file_components_with_original_config(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", variant="fp16")
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, original_config=self.original_config
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    @pytest.mark.skip(reason="runwayml original config repo does not exist")
    def test_single_file_components_with_original_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            controlnet=controlnet,
            safety_checker=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)
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
            self.ckpt_path,
            controlnet=controlnet,
            config=self.repo_id,
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            controlnet=controlnet,
            safety_checker=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)
            local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                config=local_diffusers_config,
                controlnet=controlnet,
                safety_checker=None,
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
