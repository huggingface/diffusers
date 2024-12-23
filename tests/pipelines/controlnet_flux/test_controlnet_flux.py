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
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetPipeline,
    FluxTransformer2DModel,
)
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_big_gpu_with_torch_cuda,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class FluxControlNetPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlNetPipeline

    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"])
    batch_params = frozenset(["prompt"])

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )

        torch.manual_seed(0)
        controlnet = FluxControlNetModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
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
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
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
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
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
            "guidance_scale": 3.5,
            "output_type": "np",
            "control_image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        return inputs

    def test_controlnet_flux(self):
        components = self.get_dummy_components()
        flux_pipe = FluxControlNetPipeline(**components)
        flux_pipe = flux_pipe.to(torch_device, dtype=torch.float16)
        flux_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = flux_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array(
            [0.47387695, 0.63134766, 0.5605469, 0.61621094, 0.7207031, 0.7089844, 0.70410156, 0.6113281, 0.64160156]
        )

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f"Expected: {expected_slice}, got: {image_slice.flatten()}"

    @unittest.skip("xFormersAttnProcessor does not work with SD3 Joint Attention")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update(
                {
                    "control_image": randn_tensor(
                        (1, 3, height, width),
                        device=torch_device,
                        dtype=torch.float16,
                    )
                }
            )
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)


@slow
@require_big_gpu_with_torch_cuda
@pytest.mark.big_gpu_with_torch_cuda
class FluxControlNetPipelineSlowTests(unittest.TestCase):
    pipeline_class = FluxControlNetPipeline

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_canny(self):
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        control_image = load_image(
            "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
        ).resize((512, 512))

        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        )
        pooled_prompt_embeds = torch.load(
            hf_hub_download(
                repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/pooled_prompt_embeds.pt"
            )
        )

        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=2,
            guidance_scale=3.5,
            max_sequence_length=256,
            output_type="np",
            height=512,
            width=512,
            generator=generator,
        )

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array([0.2734, 0.2852, 0.2852, 0.2734, 0.2754, 0.2891, 0.2617, 0.2637, 0.2773])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 1e-2
