# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoPipelineForImage2Image,
    Kandinsky3Img2ImgPipeline,
    Kandinsky3UNet,
    VQModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Kandinsky3Img2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Kandinsky3Img2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS
    test_xformers_attention = False
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "output_type",
            "return_dict",
        ]
    )

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
            in_channels=4,
            time_embedding_dim=4,
            groups=2,
            attention_head_dim=4,
            layers_per_block=3,
            block_out_channels=(32, 64),
            cross_attention_dim=4,
            encoder_hid_dim=32,
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
        # create init_image
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "generator": generator,
            "strength": 0.75,
            "num_inference_steps": 10,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_dict_tuple_outputs_equivalent(self):
        expected_slice = None
        if torch_device == "cpu":
            expected_slice = np.array([0.5762, 0.6112, 0.4150, 0.6018, 0.6167, 0.4626, 0.5426, 0.5641, 0.6536])
        super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)

    def test_kandinsky3_img2img(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array(
            [0.576259, 0.6132097, 0.41703486, 0.603196, 0.62062526, 0.4655338, 0.5434324, 0.5660727, 0.65433365]
        )

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=1e-1)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-2)


@slow
@require_torch_gpu
class Kandinsky3Img2ImgPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_kandinskyV3_img2img(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png"
        )
        w, h = 512, 512
        image = image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
        prompt = "A painting of the inside of a subway train with tiny raccoons."

        image = pipe(prompt, image=image, strength=0.75, num_inference_steps=5, generator=generator).images[0]

        assert image.size == (512, 512)

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/i2i.png"
        )

        image_processor = VaeImageProcessor()

        image_np = image_processor.pil_to_numpy(image)
        expected_image_np = image_processor.pil_to_numpy(expected_image)

        self.assertTrue(np.allclose(image_np, expected_image_np, atol=5e-2))
