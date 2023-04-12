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
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    DPTConfig,
    DPTFeatureExtractor,
    DPTForDepthEstimation,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionDepth2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import (
    floats_tensor,
    is_accelerate_available,
    is_accelerate_version,
    load_image,
    load_numpy,
    nightly,
    slow,
    torch_device,
)
from diffusers.utils.testing_utils import require_torch_gpu, skip_mps

from ...pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


@skip_mps
class StableDiffusionDepth2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionDepth2ImgPipeline
    test_save_load_optional_components = False
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=5,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=(2, 4),
            use_linear_projection=True,
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

        backbone_config = {
            "global_padding": "same",
            "layer_type": "bottleneck",
            "depths": [3, 4, 9],
            "out_features": ["stage1", "stage2", "stage3"],
            "embedding_dynamic_padding": True,
            "hidden_sizes": [96, 192, 384, 768],
            "num_groups": 2,
        }
        depth_estimator_config = DPTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=4,
            backbone_out_indices=(0, 1, 2, 3),
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            initializer_range=0.02,
            is_hybrid=True,
            backbone_config=backbone_config,
            backbone_featmap_shape=[1, 384, 24, 24],
        )
        depth_estimator = DPTForDepthEstimation(depth_estimator_config)
        feature_extractor = DPTFeatureExtractor.from_pretrained(
            "hf-internal-testing/tiny-random-DPTForDepthEstimation"
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "depth_estimator": depth_estimator,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_save_load_local(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 1e-4)

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_save_load_float16(self):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 2e-2, "The output of the fp16 pipeline changed after saving and loading.")

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_float16_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.half()
        pipe_fp16 = self.pipeline_class(**components)
        pipe_fp16.to(torch_device)
        pipe_fp16.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(torch_device))[0]
        output_fp16 = pipe_fp16(**self.get_dummy_inputs(torch_device))[0]

        max_diff = np.abs(output - output_fp16).max()
        self.assertLess(max_diff, 1.3e-2, "The outputs of the fp16 and fp32 pipelines are too different.")

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )
    def test_cpu_offload_forward_pass(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(output_with_offload - output_without_offload).max()
        self.assertLess(max_diff, 1e-4, "CPU offloading should not affect the inference results")

    def test_dict_tuple_outputs_equivalent(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(torch_device))[0]
        output_tuple = pipe(**self.get_dummy_inputs(torch_device), return_dict=False)[0]

        max_diff = np.abs(output - output_tuple).max()
        self.assertLess(max_diff, 1e-4)

    def test_progress_bar(self):
        super().test_progress_bar()

    def test_stable_diffusion_depth2img_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        if torch_device == "mps":
            expected_slice = np.array([0.6071, 0.5035, 0.4378, 0.5776, 0.5753, 0.4316, 0.4513, 0.5263, 0.4546])
        else:
            expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 0.4654, 0.3786, 0.5077, 0.4655])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_depth2img_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        if torch_device == "mps":
            expected_slice = np.array([0.6296, 0.5125, 0.3890, 0.4456, 0.5955, 0.4621, 0.3810, 0.5310, 0.4626])
        else:
            expected_slice = np.array([0.6012, 0.4507, 0.3769, 0.4121, 0.5566, 0.4585, 0.3803, 0.5045, 0.4631])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_depth2img_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = 2 * [inputs["image"]]
        image = pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 32, 32, 3)

        if torch_device == "mps":
            expected_slice = np.array([0.6501, 0.5150, 0.4939, 0.6688, 0.5437, 0.5758, 0.5115, 0.4406, 0.4551])
        else:
            expected_slice = np.array([0.6557, 0.6214, 0.6254, 0.5775, 0.4785, 0.5949, 0.5904, 0.4785, 0.4730])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_depth2img_pil(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = StableDiffusionDepth2ImgPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        if torch_device == "mps":
            expected_slice = np.array([0.53232, 0.47015, 0.40868, 0.45651, 0.4891, 0.4668, 0.4287, 0.48822, 0.47439])
        else:
            expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 0.4654, 0.3786, 0.5077, 0.4655])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass()


@slow
@require_torch_gpu
class StableDiffusionDepth2ImgPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_depth2img_pipeline_default(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 0.4654, 0.3786, 0.5077, 0.4655])

        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_depth2img_pipeline_k_lms(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.6363, 0.6274, 0.6309, 0.6370, 0.6226, 0.6286, 0.6213, 0.6453, 0.6306])

        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_depth2img_pipeline_ddim(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.6424, 0.6524, 0.6249, 0.6041, 0.6634, 0.6420, 0.6522, 0.6555, 0.6436])

        assert np.abs(expected_slice - image_slice).max() < 1e-4

    def test_stable_diffusion_depth2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: torch.FloatTensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 60, 80)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.7168, -1.5137, -0.1418, -2.9219, -2.7266, -2.4414, -2.1035, -3.0078, -1.7051]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 60, 80)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array(
                    [-0.7109, -1.5068, -0.1403, -2.9160, -2.7207, -2.4414, -2.1035, -3.0059, -1.7090]
                )

                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(dtype=torch.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == 2

    def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None, torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing(1)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs(dtype=torch.float16)
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 2.9 GB is allocated
        assert mem_bytes < 2.9 * 10**9


@nightly
@require_torch_gpu
class StableDiffusionImg2ImgPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_depth2img_pndm(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_depth2img_ddim(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_ddim.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_lms(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_lms.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_img2img_dpm(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
