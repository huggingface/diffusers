import gc

import numpy as np
import pytest
import torch
from torch.backends.cuda import sdp_kernel

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    Expectations,
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    require_torch_2,
    require_torch_accelerator,
    torch_device,
)
from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class ConsistencyModelPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = ConsistencyModelPipeline
    required_input_params_in_call_signature = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_input_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS
    output_shape = (3, 32, 32)
    # Unconditional generation: the pipeline takes a `batch_size` instead of `num_images_per_prompt`, and still
    # exposes the legacy `callback` / `callback_steps` arguments.
    optional_input_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    @property
    def dummy_uncond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "diffusers/consistency-models-test",
            subfolder="test_unet",
        )
        return unet

    @property
    def dummy_cond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "diffusers/consistency-models-test",
            subfolder="test_unet_class_cond",
        )
        return unet

    def get_dummy_components(self, class_cond=False):
        if class_cond:
            unet = self.dummy_cond_unet
        else:
            unet = self.dummy_uncond_unet

        # Default to CM multistep sampler
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )

        return {
            "unet": unet,
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self):
        return {
            "batch_size": 1,
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "generator": self.get_generator(0),
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestConsistencyModelPipeline(ConsistencyModelPipelineTesterConfig, PipelineTesterMixin):
    def test_consistency_model_pipeline_multistep(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])
        # fmt: on

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_consistency_model_pipeline_multistep_class_cond(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline(**self.get_dummy_components(class_cond=True))

        inputs = self.get_dummy_inputs()
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])
        # fmt: on

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_consistency_model_pipeline_onestep(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])
        # fmt: on

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_consistency_model_pipeline_onestep_class_cond(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline(**self.get_dummy_components(class_cond=True))

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])
        # fmt: on

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)


class TestConsistencyModelPipelineMemory(ConsistencyModelPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the consistency model pipeline."""


@nightly
@require_torch_accelerator
class TestConsistencyModelPipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, seed=0, get_fixed_latents=False, device="cpu", dtype=torch.float32, shape=(1, 3, 64, 64)):
        generator = torch.manual_seed(seed)

        inputs = {
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "class_labels": 0,
            "generator": generator,
            "output_type": "np",
        }

        if get_fixed_latents:
            latents = self.get_fixed_latents(seed=seed, device=device, dtype=dtype, shape=shape)
            inputs["latents"] = latents

        return inputs

    def get_fixed_latents(self, seed=0, device="cpu", dtype=torch.float32, shape=(1, 3, 64, 64)):
        if isinstance(device, str):
            device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def test_consistency_model_cd_multistep(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.0146, 0.0158, 0.0092, 0.0086, 0.0000, 0.0000, 0.0000, 0.0000, 0.0058])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_cd_onestep(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.0059, 0.0003, 0.0000, 0.0023, 0.0052, 0.0007, 0.0165, 0.0081, 0.0095])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @require_torch_2
    def test_consistency_model_cd_multistep_flash_attn(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device, torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(get_fixed_latents=True, device=torch_device)
        # Ensure usage of flash attention in torch 2.0
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            image = pipe(**inputs).images

        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slices = Expectations(
            {
                ("xpu", 3): np.array([0.0816, 0.0518, 0.0445, 0.0594, 0.0739, 0.0534, 0.0805, 0.0457, 0.0765]),
                ("cuda", 7): np.array([0.1845, 0.1371, 0.1211, 0.2035, 0.1954, 0.1323, 0.1773, 0.1593, 0.1314]),
                ("cuda", 8): np.array([0.0816, 0.0518, 0.0445, 0.0594, 0.0739, 0.0534, 0.0805, 0.0457, 0.0765]),
            }
        )
        expected_slice = expected_slices.get_expectation()

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @require_torch_2
    def test_consistency_model_cd_onestep_flash_attn(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device, torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(get_fixed_latents=True, device=torch_device)
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        # Ensure usage of flash attention in torch 2.0
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.1623, 0.2009, 0.2387, 0.1731, 0.1168, 0.1202, 0.2031, 0.1327, 0.2447])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
