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
import traceback
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    ControlNetXSAdapter,
    DDIMScheduler,
    LCMScheduler,
    StableDiffusionControlNetXSPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    is_torch_compile,
    load_image,
    load_numpy,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ...models.autoencoders.test_models_vae import (
    get_asym_autoencoder_kl_config,
    get_autoencoder_kl_config,
    get_autoencoder_tiny_config,
    get_consistency_vae_config,
)
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
    SDFunctionTesterMixin,
)


enable_full_determinism()


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


# Will be run via run_test_in_subprocess
def _test_stable_diffusion_compile(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)

        controlnet = ControlNetXSAdapter.from_pretrained(
            "UmerHA/Testing-ConrolNetXS-SD2.1-canny", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=None)

        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))

        output = pipe(prompt, image, num_inference_steps=10, generator=generator, output_type="np")
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


class ControlNetXSPipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionControlNetXSPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    test_attention_slicing = False

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=16,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=8,
            norm_num_groups=4,
            time_cond_proj_dim=time_cond_proj_dim,
            use_linear_projection=True,
        )
        torch.manual_seed(0)
        controlnet = ControlNetXSAdapter.from_unet(
            unet=unet,
            size_ratio=1,
            learn_time_embedding=True,
            conditioning_embedding_out_channels=(2, 2),
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
            hidden_size=8,
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
            (1, 3, 8 * controlnet_embedder_scale_factor, 8 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(device),
        )

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "image": image,
        }

        return inputs

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    def test_controlnet_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=8)
        sd_pipe = StableDiffusionControlNetXSPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 16, 16, 3)
        expected_slice = np.array([0.745, 0.753, 0.767, 0.543, 0.523, 0.502, 0.314, 0.521, 0.478])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # pipeline creates a new UNetControlNetXSModel under the hood. So we need to check the dtype from pipe.components
        model_dtypes = [component.dtype for component in pipe.components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

        pipe.to(dtype=torch.float16)
        model_dtypes = [component.dtype for component in pipe.components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))

    def test_multi_vae(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        block_out_channels = pipe.vae.config.block_out_channels
        norm_num_groups = pipe.vae.config.norm_num_groups

        vae_classes = [AutoencoderKL, AsymmetricAutoencoderKL, ConsistencyDecoderVAE, AutoencoderTiny]
        configs = [
            get_autoencoder_kl_config(block_out_channels, norm_num_groups),
            get_asym_autoencoder_kl_config(block_out_channels, norm_num_groups),
            get_consistency_vae_config(block_out_channels, norm_num_groups),
            get_autoencoder_tiny_config(block_out_channels),
        ]

        out_np = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]

        for vae_cls, config in zip(vae_classes, configs):
            vae = vae_cls(**config)
            vae = vae.to(torch_device)
            components["vae"] = vae
            vae_pipe = self.pipeline_class(**components)

            # pipeline creates a new UNetControlNetXSModel under the hood, which aren't on device.
            # So we need to move the new pipe to device.
            vae_pipe.to(torch_device)
            vae_pipe.set_progress_bar_config(disable=None)

            out_vae_np = vae_pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]

            assert out_vae_np.shape == out_np.shape

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        # pipeline creates a new UNetControlNetXSModel under the hood. So we need to check the device from pipe.components
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to("cuda")
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        self.assertTrue(all(device == "cuda" for device in model_devices))

        output_cuda = pipe(**self.get_dummy_inputs("cuda"))[0]
        self.assertTrue(np.isnan(to_np(output_cuda)).sum() == 0)


@slow
@require_torch_gpu
class ControlNetXSPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_canny(self):
        controlnet = ControlNetXSAdapter.from_pretrained(
            "UmerHA/Testing-ConrolNetXS-SD2.1-canny", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array([0.1963, 0.229, 0.2659, 0.2109, 0.2332, 0.2827, 0.2534, 0.2422, 0.2808])
        assert np.allclose(original_image, expected_image, atol=1e-04)

    def test_depth(self):
        controlnet = ControlNetXSAdapter.from_pretrained(
            "UmerHA/Testing-ConrolNetXS-SD2.1-depth", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array([0.4844, 0.4937, 0.4956, 0.4663, 0.5039, 0.5044, 0.4565, 0.4883, 0.4941])
        assert np.allclose(original_image, expected_image, atol=1e-04)

    @is_torch_compile
    @require_torch_2
    def test_stable_diffusion_compile(self):
        run_test_in_subprocess(test_case=self, target_func=_test_stable_diffusion_compile, inputs=None)
