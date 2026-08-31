# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    Kandinsky3Pipeline,
    Kandinsky3UNet,
    VQModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    load_image,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class Kandinsky3PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Kandinsky3Pipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS
    output_shape = (3, 16, 16)

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
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config).eval()

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

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "width": 16,
            "height": 16,
        }


class TestKandinsky3Pipeline(Kandinsky3PipelineTesterConfig, PipelineTesterMixin):
    def test_kandinsky3(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # This pipeline only denormalizes the decoded image for `output_type` "np"/"pil", so `"pt"` hands back the
        # raw decoder output. Map it into the [0, 1] range the expected slice below was recorded in.
        image = (image * 0.5 + 0.5).clamp(0, 1)

        # fmt: off
        expected_slice = torch.tensor([0.3944, 0.3680, 0.4842, 0.5333, 0.4412, 0.4812, 0.5089, 0.5381, 0.5578])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-1)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-2):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestKandinsky3PipelineMemory(Kandinsky3PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Kandinsky 3 pipeline."""


@slow
@require_torch_accelerator
class TestKandinsky3PipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_kandinskyV3(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

        generator = torch.Generator(device="cpu").manual_seed(0)

        image = pipe(prompt, num_inference_steps=5, generator=generator).images[0]

        assert image.size == (1024, 1024)

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png"
        )

        image_processor = VaeImageProcessor()

        image_np = image_processor.pil_to_numpy(image)
        expected_image_np = image_processor.pil_to_numpy(expected_image)

        assert np.allclose(image_np, expected_image_np, atol=5e-2)

    def test_kandinskyV3_img2img(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-3", variant="fp16", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
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

        assert np.allclose(image_np, expected_image_np, atol=5e-2)
