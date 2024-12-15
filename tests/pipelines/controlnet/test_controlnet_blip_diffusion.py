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
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer
from transformers.models.blip_2.configuration_blip_2 import Blip2Config
from transformers.models.clip.configuration_clip import CLIPTextConfig

from diffusers import (
    AutoencoderKL,
    BlipDiffusionControlNetPipeline,
    ControlNetModel,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import enable_full_determinism, torch_device
from src.diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from src.diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from src.diffusers.pipelines.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class BlipDiffusionControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BlipDiffusionControlNetPipeline
    params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
        "condtioning_image",
    ]
    batch_params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
        "condtioning_image",
    ]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "num_inference_steps",
        "neg_prompt",
        "guidance_scale",
        "prompt_strength",
        "prompt_reps",
    ]

    def get_dummy_components(self):
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            vocab_size=1000,
            hidden_size=16,
            intermediate_size=16,
            projection_dim=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            max_position_embeddings=77,
        )
        text_encoder = ContextCLIPTextModel(text_encoder_config)

        vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            act_fn="silu",
            latent_channels=4,
            norm_num_groups=16,
            sample_size=16,
        )

        blip_vision_config = {
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "image_size": 224,
            "patch_size": 14,
            "hidden_act": "quick_gelu",
        }

        blip_qformer_config = {
            "vocab_size": 1000,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 16,
            "max_position_embeddings": 512,
            "cross_attention_frequency": 1,
            "encoder_hidden_size": 16,
        }
        qformer_config = Blip2Config(
            vision_config=blip_vision_config,
            qformer_config=blip_qformer_config,
            num_query_tokens=16,
            tokenizer="hf-internal-testing/tiny-random-bert",
        )
        qformer = Blip2QFormerModel(qformer_config)

        unet = UNet2DConditionModel(
            block_out_channels=(4, 16),
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=16,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=16,
        )
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            skip_prk_steps=True,
        )
        controlnet = ControlNetModel(
            block_out_channels=(4, 16),
            layers_per_block=1,
            in_channels=4,
            norm_num_groups=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=16,
            conditioning_embedding_out_channels=(8, 16),
        )

        vae.eval()
        qformer.eval()
        text_encoder.eval()

        image_processor = BlipImageProcessor()

        components = {
            "text_encoder": text_encoder,
            "vae": vae,
            "qformer": qformer,
            "unet": unet,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "controlnet": controlnet,
            "image_processor": image_processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        np.random.seed(seed)
        reference_image = np.random.rand(32, 32, 3) * 255
        reference_image = Image.fromarray(reference_image.astype("uint8")).convert("RGBA")
        cond_image = np.random.rand(32, 32, 3) * 255
        cond_image = Image.fromarray(cond_image.astype("uint8")).convert("RGBA")

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "swimming underwater",
            "generator": generator,
            "reference_image": reference_image,
            "condtioning_image": cond_image,
            "source_subject_category": "dog",
            "target_subject_category": "dog",
            "height": 32,
            "width": 32,
            "guidance_scale": 7.5,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_dict_tuple_outputs_equivalent(self):
        expected_slice = None
        if torch_device == "cpu":
            expected_slice = np.array([0.4803, 0.3865, 0.1422, 0.6119, 0.2283, 0.6365, 0.5453, 0.5205, 0.3581])
        super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)

    def test_blipdiffusion_controlnet(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_dummy_inputs(device))[0]
        image_slice = image[0, -3:, -3:, 0]

        assert image.shape == (1, 16, 16, 4)
        expected_slice = np.array([0.7953, 0.7136, 0.6597, 0.4779, 0.7389, 0.4111, 0.5826, 0.4150, 0.8422])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
