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
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import Kandinsky3UNet, KandinskyV3Pipeline, VQModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_image,
    require_torch_gpu,
    slow,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class KandinskyV3PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV3Pipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS
    test_xformers_attention = False

    @property
    def dummy_movq_kwargs(self):
        return {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": [
                "AttnUpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "vq_embed_dim": 4,
        }

    @property
    def dummy_movq(self):
        torch.manual_seed(0)
        model = VQModel(**self.dummy_movq_kwargs)
        return model

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = Kandinsky3UNet(
            model_channels=16,
            init_channels=16,
            num_channels=4,
            time_embed_dim=4,
            groups=2,
            head_dim=4,
            expansion_ratio=4,
            compression_ratio=2,
            dim_mult=(1, 2),
            num_blocks=(1, 2),
            model_dim=32,
            context_dim=4,
            add_cross_attention=(False, True),
        )
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            thresholding=False,
        )
        torch.manual_seed(0)
        movq = self.dummy_movq
        torch.manual_seed(0)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "movq": movq,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "width": 16,
            "height": 16,
        }
        return inputs

    def test_kandinsky3(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 16, 16, 3)

        print(torch.from_numpy(image_slice.flatten()))
        expected_slice = np.array([0.4141, 0.2835, 0.4614, 0.5420, 0.3982, 0.4227, 0.4909, 0.4793, 0.5141])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=1e-1)

@slow
@require_torch_gpu
class KandinskyV3PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_kandinskyV3(self):
        from safetensors.torch import load_file

        from ...convert_kandinsky3_unet import convert_state_dict

        state_dict = load_file("/home/patrick/kandinsky-3/unet/diffusion_pytorch_model.fp16.safetensors")
        state_dict = convert_state_dict(state_dict)
        unet = Kandinsky3UNet()
        unet.load_state_dict(state_dict)
        unet.to(torch.float16)
        pipe = KandinskyV3Pipeline.from_pretrained("/home/patrick/kandinsky-3", unet=unet, variant="fp16", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

        generator = torch.Generator(device="cpu").manual_seed(0)

        image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]

        assert image.size == (1024, 1024)

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png"
        )

        image_processor = VaeImageProcessor()

        image_np = image_processor.pil_to_numpy(image)
        expected_image_np = image_processor.pil_to_numpy(expected_image)

        self.assertTrue(np.allclose(image_np, expected_image_np, atol=5e-2))
