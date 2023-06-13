# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import gc
import random
import traceback
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from diffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow, torch_device
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
)

from ...models.test_models_unet_2d_condition import create_lora_layers
from ..pipeline_params import TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS, TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


# Will be run via run_test_in_subprocess
def _test_inpaint_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        torch_device = inputs.pop("torch_device")
        seed = inputs.pop("seed")
        inputs["generator"] = torch.Generator(device=torch_device).manual_seed(seed)

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0425, 0.0273, 0.0344, 0.1694, 0.1727, 0.1812, 0.3256, 0.3311, 0.3272])

        assert np.abs(expected_slice - image_slice).max() < 3e-3
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class StableDiffusionInpaintPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4723, 0.5731, 0.3939, 0.5441, 0.5922, 0.4392, 0.5059, 0.4651, 0.4474])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_inpaint_image_tensor(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        out_pil = output.images

        inputs = self.get_dummy_inputs(device)
        inputs["image"] = torch.tensor(np.array(inputs["image"]) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0)
        inputs["mask_image"] = torch.tensor(np.array(inputs["mask_image"]) / 255).permute(2, 0, 1)[:1].unsqueeze(0)
        output = sd_pipe(**inputs)
        out_tensor = output.images

        assert out_pil.shape == (1, 64, 64, 3)
        assert np.abs(out_pil.flatten() - out_tensor.flatten()).max() < 5e-2

    def test_stable_diffusion_inpaint_lora(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward 1
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        # set lora layers
        lora_attn_procs = create_lora_layers(sd_pipe.unet)
        sd_pipe.unet.set_attn_processor(lora_attn_procs)
        sd_pipe = sd_pipe.to(torch_device)

        # forward 2
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.0})
        image = output.images
        image_slice_1 = image[0, -3:, -3:, -1]

        # forward 3
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.5})
        image = output.images
        image_slice_2 = image[0, -3:, -3:, -1]

        assert np.abs(image_slice - image_slice_1).max() < 1e-2
        assert np.abs(image_slice - image_slice_2).max() > 1e-2

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_stable_diffusion_inpaint_strength_zero_test(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        # check that the pipeline raises value error when num_inference_steps is < 1
        inputs["strength"] = 0.01
        with self.assertRaises(ValueError):
            sd_pipe(**inputs).images


class StableDiffusionSimpleInpaintPipelineFastTests(StableDiffusionInpaintPipelineFastTests):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def test_stable_diffusion_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4925, 0.4967, 0.4100, 0.5234, 0.5322, 0.4532, 0.5805, 0.5877, 0.4151])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @unittest.skip("skipped here because area stays unchanged due to mask")
    def test_stable_diffusion_inpaint_lora(self):
        ...


@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0427, 0.0460, 0.0483, 0.0460, 0.0584, 0.0521, 0.1549, 0.1695, 0.1794])

        assert np.abs(expected_slice - image_slice).max() < 6e-4

    def test_stable_diffusion_inpaint_fp16(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.1350, 0.1123, 0.1350, 0.1641, 0.1328, 0.1230, 0.1289, 0.1531, 0.1687])

        assert np.abs(expected_slice - image_slice).max() < 5e-2

    def test_stable_diffusion_inpaint_pndm(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0425, 0.0273, 0.0344, 0.1694, 0.1727, 0.1812, 0.3256, 0.3311, 0.3272])

        assert np.abs(expected_slice - image_slice).max() < 5e-3

    def test_stable_diffusion_inpaint_k_lms(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9314, 0.7575, 0.9432, 0.8885, 0.9028, 0.7298, 0.9811, 0.9667, 0.7633])

        assert np.abs(expected_slice - image_slice).max() < 6e-3

    def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.2 GB is allocated
        assert mem_bytes < 2.2 * 10**9

    @require_torch_2
    def test_inpaint_compile(self):
        seed = 0
        inputs = self.get_inputs(torch_device, seed=seed)
        # Can't pickle a Generator object
        del inputs["generator"]
        inputs["torch_device"] = torch_device
        inputs["seed"] = seed
        run_test_in_subprocess(test_case=self, target_func=_test_inpaint_compile, inputs=inputs)

    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        # change input image to a random size (one that would cause a tensor mismatch error)
        inputs["image"] = inputs["image"].resize((127, 127))
        inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
        inputs["height"] = 128
        inputs["width"] = 128
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, inputs["height"], inputs["width"], 3)

    def test_stable_diffusion_inpaint_strength_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        # change input strength
        inputs["strength"] = 0.75
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, 253:256, 253:256, -1].flatten()
        expected_slice = np.array([0.0021, 0.2350, 0.3712, 0.0575, 0.2485, 0.3451, 0.1857, 0.3156, 0.3943])
        assert np.abs(expected_slice - image_slice).max() < 3e-3

    def test_stable_diffusion_simple_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images

        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.5157, 0.6858, 0.6873, 0.4619, 0.6416, 0.6898, 0.3702, 0.5960, 0.6935])

        assert np.abs(expected_slice - image_slice).max() < 6e-4


@nightly
@require_torch_gpu
class StableDiffusionInpaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_inpaint_ddim(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_dpm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/stable_diffusion_inpaint_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3


class StableDiffusionInpaintingPrepareMaskAndMaskedImageTests(unittest.TestCase):
    def test_pil_inputs(self):
        height, width = 32, 32
        im = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        im = Image.fromarray(im)
        mask = np.random.randint(0, 255, (height, width), dtype=np.uint8) > 127.5
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        t_mask, t_masked, t_image = prepare_mask_and_masked_image(im, mask, height, width, return_image=True)

        self.assertTrue(isinstance(t_mask, torch.Tensor))
        self.assertTrue(isinstance(t_masked, torch.Tensor))
        self.assertTrue(isinstance(t_image, torch.Tensor))

        self.assertEqual(t_mask.ndim, 4)
        self.assertEqual(t_masked.ndim, 4)
        self.assertEqual(t_image.ndim, 4)

        self.assertEqual(t_mask.shape, (1, 1, height, width))
        self.assertEqual(t_masked.shape, (1, 3, height, width))
        self.assertEqual(t_image.shape, (1, 3, height, width))

        self.assertTrue(t_mask.dtype == torch.float32)
        self.assertTrue(t_masked.dtype == torch.float32)
        self.assertTrue(t_image.dtype == torch.float32)

        self.assertTrue(t_mask.min() >= 0.0)
        self.assertTrue(t_mask.max() <= 1.0)
        self.assertTrue(t_masked.min() >= -1.0)
        self.assertTrue(t_masked.min() <= 1.0)
        self.assertTrue(t_image.min() >= -1.0)
        self.assertTrue(t_image.min() >= -1.0)

        self.assertTrue(t_mask.sum() > 0.0)

    def test_np_inputs(self):
        height, width = 32, 32

        im_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        im_pil = Image.fromarray(im_np)
        mask_np = (
            np.random.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )
            > 127.5
        )
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        t_mask_pil, t_masked_pil, t_image_pil = prepare_mask_and_masked_image(
            im_pil, mask_pil, height, width, return_image=True
        )

        self.assertTrue((t_mask_np == t_mask_pil).all())
        self.assertTrue((t_masked_np == t_masked_pil).all())
        self.assertTrue((t_image_np == t_image_pil).all())

    def test_torch_3D_2D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_3D_3D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_4D_2D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_4D_3D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_4D_4D_inputs(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    1,
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0][0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_batch_4D_3D(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                2,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    2,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = torch.cat([n[0] for n in nps])
        t_masked_np = torch.cat([n[1] for n in nps])
        t_image_np = torch.cat([n[2] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_torch_batch_4D_4D(self):
        height, width = 32, 32

        im_tensor = torch.randint(
            0,
            255,
            (
                2,
                3,
                height,
                width,
            ),
            dtype=torch.uint8,
        )
        mask_tensor = (
            torch.randint(
                0,
                255,
                (
                    2,
                    1,
                    height,
                    width,
                ),
                dtype=torch.uint8,
            )
            > 127.5
        )

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy()[0] for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = torch.cat([n[0] for n in nps])
        t_masked_np = torch.cat([n[1] for n in nps])
        t_image_np = torch.cat([n[2] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_shape_mismatch(self):
        height, width = 32, 32

        # test height and width
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.randn(
                    3,
                    height,
                    width,
                ),
                torch.randn(64, 64),
                height,
                width,
                return_image=True,
            )
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.randn(
                    2,
                    3,
                    height,
                    width,
                ),
                torch.randn(4, 64, 64),
                height,
                width,
                return_image=True,
            )
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.randn(
                    2,
                    3,
                    height,
                    width,
                ),
                torch.randn(4, 1, 64, 64),
                height,
                width,
                return_image=True,
            )

    def test_type_mismatch(self):
        height, width = 32, 32

        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ),
                torch.rand(
                    3,
                    height,
                    width,
                ).numpy(),
                height,
                width,
                return_image=True,
            )
        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ).numpy(),
                torch.rand(
                    3,
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )

    def test_channels_first(self):
        height, width = 32, 32

        # test channels first for 3D tensors
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                torch.rand(height, width, 3),
                torch.rand(
                    3,
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )

    def test_tensor_range(self):
        height, width = 32, 32

        # test im <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.ones(
                    3,
                    height,
                    width,
                )
                * 2,
                torch.rand(
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )
        # test im >= -1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.ones(
                    3,
                    height,
                    width,
                )
                * (-2),
                torch.rand(
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )
        # test mask <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ),
                torch.ones(
                    height,
                    width,
                )
                * 2,
                height,
                width,
                return_image=True,
            )
        # test mask >= 0
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                torch.rand(
                    3,
                    height,
                    width,
                ),
                torch.ones(
                    height,
                    width,
                )
                * -1,
                height,
                width,
                return_image=True,
            )
