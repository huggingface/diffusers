# coding=utf-8
# Copyright 2025 HuggingFace Inc and Tencent Hunyuan Team.
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
from transformers import AutoConfig, AutoTokenizer, BertModel, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    HunyuanDiT2DModel,
    HunyuanDiTControlNetPipeline,
)
from diffusers.models import HunyuanDiT2DControlNetModel, HunyuanDiT2DMultiControlNetModel
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class HunyuanDiTControlNetPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanDiTControlNetPipeline
    required_input_params_in_call_signature = frozenset(
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
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = HunyuanDiT2DModel(
            sample_size=16,
            num_layers=4,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            in_channels=4,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            hidden_size=24,
            activation_fn="gelu-approximate",
        )

        torch.manual_seed(0)
        controlnet = HunyuanDiT2DControlNetModel(
            sample_size=16,
            transformer_num_layers=4,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            in_channels=4,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            hidden_size=24,
            activation_fn="gelu-approximate",
        )

        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDPMScheduler()
        text_encoder = BertModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder_2 = T5EncoderModel(config).eval()
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "safety_checker": None,
            "feature_extractor": None,
            "controlnet": controlnet,
        }

    def get_dummy_inputs(self):
        # The control image is drawn from the same generator that is handed to the pipeline, so the pipeline sees an
        # already-advanced generator state — keep the order to stay comparable with the expected slice below.
        generator = self.get_generator(0)
        control_image = randn_tensor(
            (1, 3, 16, 16),
            generator=generator,
            device=torch.device(torch_device),
            dtype=torch.float32,
        )

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "control_image": control_image,
            "controlnet_conditioning_scale": 0.5,
        }


class TestHunyuanDiTControlNetPipeline(HunyuanDiTControlNetPipelineTesterConfig, PipelineTesterMixin):
    def test_controlnet_hunyuandit(self):
        pipe = self.get_pipeline().to(torch_device, dtype=torch.float32)

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.5925, 0.5392, 0.4450, 0.7140, 0.3954, 0.3553, 0.3842, 0.5994, 0.3765])
        # fmt: on

        assert_tensors_close(image_slice.flatten().cpu(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("TODO(YiYi) need to fix later")
    def test_save_load_optional_components(self):
        pass

    @pytest.mark.skip(
        "Test not supported as `encode_prompt` is called two times separately which deivates from about 99% of the pipelines we have."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestHunyuanDiTControlNetPipelineMemory(HunyuanDiTControlNetPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanDiT ControlNet pipeline."""

    @pytest.mark.skip("TODO(YiYi) need to fix later")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @pytest.mark.skip("TODO(YiYi) need to fix later")
    def test_sequential_offload_forward_pass_twice(self):
        pass


@slow
@require_torch_accelerator
class TestHunyuanDiTControlNetPipelineSlow:
    pipeline_class = HunyuanDiTControlNetPipeline

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_canny(self):
        controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny", torch_dtype=torch.float16
        )
        pipe = HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."
        n_prompt = ""
        control_image = load_image(
            "https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true"
        )

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
            [0.43652344, 0.4399414, 0.44921875, 0.45043945, 0.45703125, 0.44873047, 0.43579102, 0.44018555, 0.42578125]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_pose(self):
        controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose", torch_dtype=torch.float16
        )
        pipe = HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "An Asian woman, dressed in a green top, wearing a purple headscarf and a purple scarf, stands in front of a blackboard. The background is the blackboard. The photo is presented in a close-up, eye-level, and centered composition, adopting a realistic photographic style"
        n_prompt = ""
        control_image = load_image(
            "https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose/resolve/main/pose.jpg?download=true"
        )

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
            [0.4091797, 0.4177246, 0.39526367, 0.4194336, 0.40356445, 0.3857422, 0.39208984, 0.40429688, 0.37451172]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_depth(self):
        controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth", torch_dtype=torch.float16
        )
        pipe = HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "In the dense forest, a black and white panda sits quietly in green trees and red flowers, surrounded by mountains, rivers, and the ocean. The background is the forest in a bright environment."
        n_prompt = ""
        control_image = load_image(
            "https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth/resolve/main/depth.jpg?download=true"
        )

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
            [0.31982422, 0.32177734, 0.30126953, 0.3190918, 0.3100586, 0.31396484, 0.3232422, 0.33544922, 0.30810547]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_multi_controlnet(self):
        controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny", torch_dtype=torch.float16
        )
        controlnet = HunyuanDiT2DMultiControlNetModel([controlnet, controlnet])

        pipe = HunyuanDiTControlNetPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."
        n_prompt = ""
        control_image = load_image(
            "https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true"
        )

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
            [0.43652344, 0.44018555, 0.4494629, 0.44995117, 0.45654297, 0.44848633, 0.43603516, 0.4404297, 0.42626953]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2
