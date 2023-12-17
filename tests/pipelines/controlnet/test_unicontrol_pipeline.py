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
import tempfile
import traceback
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionUniControlPipeline,
    UNet2DConditionModel,
)

from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_image,
    load_numpy,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# Will be run via run_test_in_subprocess
def _test_stable_diffusion_compile(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        controlnet = ControlNetModel.from_pretrained("NagaSaiAbhinay/unicontrol-v1.1-test")

        pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=None)

        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        pipe.controlnet.to(memory_format=torch.channels_last)
        pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=True)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        task = "canny"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))

        output = pipe(task, prompt, image, num_inference_steps=10, generator=generator, output_type="np")
        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out_full.npy"
        )
        expected_image = np.resize(expected_image, (512, 512, 3))

        assert np.abs(expected_image - image).max() < 1.0

    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class ControlNetPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionUniControlPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
            controlnet_conditioning_embedding = "UniControlTaskMOEEmbedding"
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
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
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        controlnet_embedder_scale_factor = 2
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(device),
        )

        inputs = {
            "task" : "canny",
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "image": image,
        }

        return inputs

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)


@slow
@require_torch_gpu
class ControlNetPipelineSlowTests(unittest.TestCase):

    controlnet = None
    pipe = None
    
    def setUp(self) -> None:
        super().setUp()
        self.controlnet = ControlNetModel.from_pretrained("NagaSaiAbhinay/unicontrol-v1.1-test", subfolder='controlnet')
        self.pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=self.controlnet
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=None)
    
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_hed(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "hed"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_canny(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        task = "canny"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out.npy"
        )

        assert np.abs(expected_image - image).max() < 9e-2

    def test_seg(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "seg"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_segbase(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "segbase"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_depth(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "normal"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_normal(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "normal"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_openpose(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "openpose"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_bbox(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "bbox"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_grayscale(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "grayscale"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1
    
    def test_outpainting(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "outpainting"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_sketch(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "hedsketch"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1
    
    def test_inpainting(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "inpainting"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_blur(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        task = "blur"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = self.pipe(task, prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")

        pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

        prompt = "house"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )

        _ = pipe(
            prompt,
            image,
            num_inference_steps=2,
            output_type="np",
        )

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 7 GB is allocated
        assert mem_bytes < 4 * 10**9

    def test_canny_guess_mode(self):
        controlnet = ControlNetModel.from_pretrained("NagaSaiAbhinay/unicontrol-v1.1-test")

        pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = ""
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=3.0,
            guess_mode=True,
        )

        image = output.images[0]
        assert image.shape == (768, 512, 3)

        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array([0.2724, 0.2846, 0.2724, 0.3843, 0.3682, 0.2736, 0.4675, 0.3862, 0.2887])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_canny_guess_mode_euler(self):
        controlnet = ControlNetModel.from_pretrained("NagaSaiAbhinay/unicontrol-v1.1-test")

        pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = ""
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=3.0,
            guess_mode=True,
        )

        image = output.images[0]
        assert image.shape == (768, 512, 3)

        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array([0.1655, 0.1721, 0.1623, 0.1685, 0.1711, 0.1646, 0.1651, 0.1631, 0.1494])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @require_python39_or_higher
    @require_torch_2
    def test_stable_diffusion_compile(self):
        run_test_in_subprocess(test_case=self, target_func=_test_stable_diffusion_compile, inputs=None)

    def test_v11_shuffle_global_pool_conditions(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle")

        pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "New York"
        image = load_image(
            "https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/control.png"
        )

        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=7.0,
        )

        image = output.images[0]
        assert image.shape == (512, 640, 3)

        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array([0.1338, 0.1597, 0.1202, 0.1687, 0.1377, 0.1017, 0.2070, 0.1574, 0.1348])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch_gpu
class StableDiffusionMultiTaskUniControlPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_pose_and_canny(self):
        controlnet = ControlNetModel.from_pretrained("NagaSaiAbhinay/unicontrol-v1.1-test")

        pipe = StableDiffusionUniControlPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird and Chef"
        task = ["pose", "canny"]
        image_canny = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        image_pose = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )

        output = pipe(task, prompt, [image_pose, image_canny], generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose_canny_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-2

