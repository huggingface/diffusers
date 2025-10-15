import gc
import tempfile

import torch

from diffusers import EulerDiscreteScheduler, StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline
from diffusers.loaders.single_file_utils import _extract_repo_id_and_weights_name
from diffusers.utils import load_image

from ..testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    require_torch_accelerator,
    slow,
    torch_device,
)
from .single_file_testing_utils import (
    SDSingleFileTesterMixin,
    download_original_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_accelerator
class TestStableDiffusionPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionPipeline
    ckpt_path = (
        "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
    )
    original_config = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)

    def test_single_file_legacy_scheduler_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(repo_id, weight_name, tmpdir)
            local_original_config = download_original_config(self.original_config, tmpdir)

            pipe = self.pipeline_class.from_single_file(
                local_ckpt_path,
                original_config=local_original_config,
                cache_dir=tmpdir,
                local_files_only=True,
                scheduler_type="euler",
            )

        # Default is PNDM for this checkpoint
        assert isinstance(pipe.scheduler, EulerDiscreteScheduler)

    def test_single_file_legacy_scaling_factor(self):
        new_scaling_factor = 10.0
        init_pipe = self.pipeline_class.from_single_file(self.ckpt_path)
        pipe = self.pipeline_class.from_single_file(self.ckpt_path, scaling_factor=new_scaling_factor)

        assert init_pipe.vae.config.scaling_factor != new_scaling_factor
        assert pipe.vae.config.scaling_factor == new_scaling_factor


@slow
class TestStableDiffusion21PipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
    repo_id = "stabilityai/stable-diffusion-2-1"

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)


@nightly
@slow
@require_torch_accelerator
class TestStableDiffusionInstructPix2PixPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInstructPix2PixPipeline
    ckpt_path = "https://huggingface.co/timbrooks/instruct-pix2pix/blob/main/instruct-pix2pix-00-22000.safetensors"
    original_config = (
        "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/refs/heads/main/configs/generate.yaml"
    )
    repo_id = "timbrooks/instruct-pix2pix"
    single_file_kwargs = {"extract_ema": True}

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_pix2pix/example.jpg"
        )
        inputs = {
            "prompt": "turn him into a cyborg",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.0,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
