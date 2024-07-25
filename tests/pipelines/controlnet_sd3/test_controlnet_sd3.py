# coding=utf-8
# Copyright 2024 HuggingFace Inc and The InstantX Team.
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
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class StableDiffusion3ControlNetPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = StableDiffusion3ControlNetPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=4,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
        )

        torch.manual_seed(0)
        controlnet = SD3ControlNetModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=8,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
            "controlnet": controlnet,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        control_image = randn_tensor(
            (1, 3, 32, 32),
            generator=generator,
            device=torch.device(device),
            dtype=torch.float16,
        )

        controlnet_conditioning_scale = 0.5

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "control_image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        return inputs

    def test_controlnet_sd3(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusion3ControlNetPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device, dtype=torch.float16)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.5767, 0.7100, 0.5981, 0.5674, 0.5952, 0.4102, 0.5093, 0.5044, 0.6030])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f"Expected: {expected_slice}, got: {image_slice.flatten()}"

    @unittest.skip("xFormersAttnProcessor does not work with SD3 Joint Attention")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass


@slow
@require_torch_gpu
class StableDiffusion3ControlNetPipelineSlowTests(unittest.TestCase):
    pipeline_class = StableDiffusion3ControlNetPipeline

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_canny(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array(
            [0.20947266, 0.1574707, 0.19897461, 0.15063477, 0.1418457, 0.17285156, 0.14160156, 0.13989258, 0.30810547]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_pose(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array(
            [0.8671875, 0.86621094, 0.91015625, 0.8491211, 0.87890625, 0.9140625, 0.8300781, 0.8334961, 0.8623047]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_tile(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX//SD3-Controlnet-Tile", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array(
            [0.6982422, 0.7011719, 0.65771484, 0.6904297, 0.7416992, 0.6904297, 0.6977539, 0.7080078, 0.6386719]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_multi_controlnet(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)
        controlnet = SD3MultiControlNetModel([controlnet, controlnet])

        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=[control_image, control_image],
            controlnet_conditioning_scale=[0.25, 0.25],
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array(
            [0.7451172, 0.7416992, 0.7158203, 0.7792969, 0.7607422, 0.7089844, 0.6855469, 0.71777344, 0.7314453]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2
