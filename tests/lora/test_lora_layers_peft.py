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
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
)
from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.testing_utils import floats_tensor, require_peft_backend, require_torch_gpu, slow


if is_peft_available():
    from peft import LoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import get_peft_model_state_dict


def create_unet_lora_layers(unet: nn.Module):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_processor_class = (
            LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        )
        lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
    unet_lora_layers = AttnProcsLayers(lora_attn_procs)
    return lora_attn_procs, unet_lora_layers


@require_peft_backend
class PeftLoraLoaderMixinTests:
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_class = None
    scheduler_cls = None
    scheduler_kwargs = None
    has_two_text_encoders = False
    unet_kwargs = None
    vae_kwargs = None

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(**self.unet_kwargs)
        scheduler = self.scheduler_cls(**self.scheduler_kwargs)
        torch.manual_seed(0)
        vae = AutoencoderKL(**self.vae_kwargs)
        text_encoder = CLIPTextModel.from_pretrained("peft-internal-testing/tiny-clip-text-2")
        tokenizer = CLIPTokenizer.from_pretrained("peft-internal-testing/tiny-clip-text-2")

        if self.has_two_text_encoders:
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("peft-internal-testing/tiny-clip-text-2")
            tokenizer_2 = CLIPTokenizer.from_pretrained("peft-internal-testing/tiny-clip-text-2")

        text_lora_config = LoraConfig(
            r=4, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], init_lora_weights=False
        )

        unet_lora_attn_procs, unet_lora_layers = create_unet_lora_layers(unet)

        if self.has_two_text_encoders:
            pipeline_components = {
                "unet": unet,
                "scheduler": scheduler,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "text_encoder_2": text_encoder_2,
                "tokenizer_2": tokenizer_2,
            }
        else:
            pipeline_components = {
                "unet": unet,
                "scheduler": scheduler,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "safety_checker": None,
                "feature_extractor": None,
            }
        lora_components = {
            "unet_lora_layers": unet_lora_layers,
            "unet_lora_attn_procs": unet_lora_attn_procs,
        }
        return pipeline_components, lora_components, text_lora_config

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    # copied from: https://colab.research.google.com/gist/sayakpaul/df2ef6e1ae6d8c10a49d859883b10860/scratchpad.ipynb
    def get_dummy_tokens(self):
        max_seq_length = 77

        inputs = torch.randint(2, 56, size=(1, max_seq_length), generator=torch.manual_seed(0))

        prepared_inputs = {}
        prepared_inputs["input_ids"] = inputs
        return prepared_inputs

    def check_if_lora_correctly_set(self, model) -> bool:
        """
        Checks if the LoRA layers are correctly set with peft
        """
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                return True
        return False

    def test_simple_inference(self):
        """
        Tests a simple inference and makes sure it works as expected
        """
        components, _, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)

        _, _, inputs = self.get_dummy_inputs()
        output_no_lora = pipe(**inputs).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

    def test_simple_inference_with_text_lora(self):
        """
        Tests a simple inference with lora attached on the text encoder
        and makes sure it works as expected
        """
        components, _, text_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        output_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

    def test_simple_inference_with_text_lora_and_scale(self):
        """
        Tests a simple inference with lora attached on the text encoder + scale argument
        and makes sure it works as expected
        """
        components, _, text_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        output_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

        output_lora_scale = pipe(
            **inputs, generator=torch.manual_seed(0), cross_attention_kwargs={"scale": 0.5}
        ).images
        self.assertTrue(
            not np.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )

        output_lora_0_scale = pipe(
            **inputs, generator=torch.manual_seed(0), cross_attention_kwargs={"scale": 0.0}
        ).images
        self.assertTrue(
            np.allclose(output_no_lora, output_lora_0_scale, atol=1e-3, rtol=1e-3),
            "Lora + 0 scale should lead to same result as no LoRA",
        )

    def test_simple_inference_with_text_lora_fused(self):
        """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected
        """
        components, _, text_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        pipe.fuse_lora()
        # Fusing should still keep the LoRA layers
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        ouput_fused = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertFalse(
            np.allclose(ouput_fused, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

    def test_simple_inference_with_text_lora_unloaded(self):
        """
        Tests a simple inference with lora attached to text encoder, then unloads the lora weights
        and makes sure it works as expected
        """
        components, _, text_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        self.assertFalse(
            self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder"
        )

        if self.has_two_text_encoders:
            self.assertFalse(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly unloaded in text encoder 2"
            )

        ouput_unloaded = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(
            np.allclose(ouput_unloaded, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

    def test_simple_inference_with_text_lora_save_load(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA.
        """
        components, _, text_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            text_encoder_state_dict = get_peft_model_state_dict(pipe.text_encoder)
            if self.has_two_text_encoders:
                text_encoder_2_state_dict = get_peft_model_state_dict(pipe.text_encoder_2)

                self.pipeline_class.save_lora_weights(
                    save_directory=tmpdirname,
                    text_encoder_lora_layers=text_encoder_state_dict,
                    text_encoder_2_lora_layers=text_encoder_2_state_dict,
                    safe_serialization=False,
                )
            else:
                self.pipeline_class.save_lora_weights(
                    save_directory=tmpdirname,
                    text_encoder_lora_layers=text_encoder_state_dict,
                    safe_serialization=False,
                )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            pipe.unload_lora_weights()

            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"))

        images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_save_pretrained(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained
        """
        components, _, text_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(self.torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(self.check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
            )

        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)

            pipe_from_pretrained = self.pipeline_class.from_pretrained(tmpdirname)
            pipe_from_pretrained.to(self.torch_device)

        self.assertTrue(
            self.check_if_lora_correctly_set(pipe_from_pretrained.text_encoder),
            "Lora not correctly set in text encoder",
        )

        if self.has_two_text_encoders:
            self.assertTrue(
                self.check_if_lora_correctly_set(pipe_from_pretrained.text_encoder_2),
                "Lora not correctly set in text encoder 2",
            )

        images_lora_save_pretrained = pipe_from_pretrained(**inputs, generator=torch.manual_seed(0)).images

        self.assertTrue(
            np.allclose(images_lora, images_lora_save_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )


class StableDiffusionLoRATests(PeftLoraLoaderMixinTests, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline
    scheduler_cls = DDIMScheduler
    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 1,
    }
    unet_kwargs = {
        "block_out_channels": (32, 64),
        "layers_per_block": 2,
        "sample_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "cross_attention_dim": 32,
    }
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
    }

    @slow
    @require_torch_gpu
    def test_integration_logits_with_scale(self):
        path = "runwayml/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.load_lora_weights(lora_id)
        pipe = pipe.to("cuda")

        self.assertTrue(
            self.check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder 2",
        )

        prompt = "a red sks dog"

        images = pipe(
            prompt=prompt,
            num_inference_steps=15,
            cross_attention_kwargs={"scale": 0.5},
            generator=torch.manual_seed(0),
            output_type="np",
        ).images

        expected_slice_scale = np.array([0.307, 0.283, 0.310, 0.310, 0.300, 0.314, 0.336, 0.314, 0.321])

        predicted_slice = images[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(expected_slice_scale, predicted_slice, atol=1e-3, rtol=1e-3))

    @slow
    @require_torch_gpu
    def test_integration_logits_no_scale(self):
        path = "runwayml/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32)
        pipe.load_lora_weights(lora_id)
        pipe = pipe.to("cuda")

        self.assertTrue(
            self.check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        prompt = "a red sks dog"

        images = pipe(prompt=prompt, num_inference_steps=30, generator=torch.manual_seed(0), output_type="np").images

        expected_slice_scale = np.array([0.074, 0.064, 0.073, 0.0842, 0.069, 0.0641, 0.0794, 0.076, 0.084])

        predicted_slice = images[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(expected_slice_scale, predicted_slice, atol=1e-3, rtol=1e-3))


class StableDiffusionXLLoRATests(PeftLoraLoaderMixinTests, unittest.TestCase):
    has_two_text_encoders = True
    pipeline_class = StableDiffusionXLPipeline
    scheduler_cls = EulerDiscreteScheduler
    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "timestep_spacing": "leading",
        "steps_offset": 1,
    }
    unet_kwargs = {
        "block_out_channels": (32, 64),
        "layers_per_block": 2,
        "sample_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "attention_head_dim": (2, 4),
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "transformer_layers_per_block": (1, 2),
        "projection_class_embeddings_input_dim": 80,  # 6 * 8 + 32
        "cross_attention_dim": 64,
    }
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
        "sample_size": 128,
    }
