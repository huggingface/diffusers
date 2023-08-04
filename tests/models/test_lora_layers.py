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
from huggingface_hub.repocard import RepoCard
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin, PatchedLoraProjection, text_encoder_attn_modules
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.utils import floats_tensor, torch_device
from diffusers.utils.testing_utils import require_torch_gpu, slow


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


def create_text_encoder_lora_attn_procs(text_encoder: nn.Module):
    text_lora_attn_procs = {}
    lora_attn_processor_class = (
        LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
    )
    for name, module in text_encoder_attn_modules(text_encoder):
        if isinstance(module.out_proj, nn.Linear):
            out_features = module.out_proj.out_features
        elif isinstance(module.out_proj, PatchedLoraProjection):
            out_features = module.out_proj.regular_linear_layer.out_features
        else:
            assert False, module.out_proj.__class__

        text_lora_attn_procs[name] = lora_attn_processor_class(hidden_size=out_features, cross_attention_dim=None)
    return text_lora_attn_procs


def create_text_encoder_lora_layers(text_encoder: nn.Module):
    text_lora_attn_procs = create_text_encoder_lora_attn_procs(text_encoder)
    text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs)
    return text_encoder_lora_layers


def set_lora_weights(lora_attn_parameters, randn_weight=False):
    with torch.no_grad():
        for parameter in lora_attn_parameters:
            if randn_weight:
                parameter[:] = torch.randn_like(parameter)
            else:
                torch.zero_(parameter)


class LoraLoaderMixinTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
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

        unet_lora_attn_procs, unet_lora_layers = create_unet_lora_layers(unet)
        text_encoder_lora_layers = create_text_encoder_lora_layers(text_encoder)

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
            "text_encoder_lora_layers": text_encoder_lora_layers,
            "unet_lora_attn_procs": unet_lora_attn_procs,
        }
        return pipeline_components, lora_components

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

    def create_lora_weight_file(self, tmpdirname):
        _, lora_components = self.get_dummy_components()
        LoraLoaderMixin.save_lora_weights(
            save_directory=tmpdirname,
            unet_lora_layers=lora_components["unet_lora_layers"],
            text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
        )
        self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))

    def test_lora_save_load(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs()

        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))

    def test_lora_save_load_safetensors(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs()

        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))

    def test_lora_save_load_legacy(self):
        pipeline_components, lora_components = self.get_dummy_components()
        unet_lora_attn_procs = lora_components["unet_lora_attn_procs"]
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs()

        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            unet = sd_pipe.unet
            unet.set_attn_processor(unet_lora_attn_procs)
            unet.save_attn_procs(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))

    def test_text_encoder_lora_monkey_patch(self):
        pipeline_components, _ = self.get_dummy_components()
        pipe = StableDiffusionPipeline(**pipeline_components)

        dummy_tokens = self.get_dummy_tokens()

        # inference without lora
        outputs_without_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_without_lora.shape == (1, 77, 32)

        # monkey patch
        params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)

        set_lora_weights(params, randn_weight=False)

        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == (1, 77, 32)

        assert torch.allclose(
            outputs_without_lora, outputs_with_lora
        ), "lora_up_weight are all zero, so the lora outputs should be the same to without lora outputs"

        # create lora_attn_procs with randn up.weights
        create_text_encoder_lora_attn_procs(pipe.text_encoder)

        # monkey patch
        params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)

        set_lora_weights(params, randn_weight=True)

        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == (1, 77, 32)

        assert not torch.allclose(
            outputs_without_lora, outputs_with_lora
        ), "lora_up_weight are not zero, so the lora outputs should be different to without lora outputs"

    def test_text_encoder_lora_remove_monkey_patch(self):
        pipeline_components, _ = self.get_dummy_components()
        pipe = StableDiffusionPipeline(**pipeline_components)

        dummy_tokens = self.get_dummy_tokens()

        # inference without lora
        outputs_without_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_without_lora.shape == (1, 77, 32)

        # monkey patch
        params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)

        set_lora_weights(params, randn_weight=True)

        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == (1, 77, 32)

        assert not torch.allclose(
            outputs_without_lora, outputs_with_lora
        ), "lora outputs should be different to without lora outputs"

        # remove monkey patch
        pipe._remove_text_encoder_monkey_patch()

        # inference with removed lora
        outputs_without_lora_removed = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_without_lora_removed.shape == (1, 77, 32)

        assert torch.allclose(
            outputs_without_lora, outputs_without_lora_removed
        ), "remove lora monkey patch should restore the original outputs"

    def test_text_encoder_lora_scale(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs()

        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        lora_images_with_scale = sd_pipe(**pipeline_inputs, cross_attention_kwargs={"scale": 0.5}).images
        lora_image_with_scale_slice = lora_images_with_scale[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(
            torch.allclose(torch.from_numpy(lora_image_slice), torch.from_numpy(lora_image_with_scale_slice))
        )

    def test_lora_unet_attn_processors(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.create_lora_weight_file(tmpdirname)

            pipeline_components, _ = self.get_dummy_components()
            sd_pipe = StableDiffusionPipeline(**pipeline_components)
            sd_pipe = sd_pipe.to(torch_device)
            sd_pipe.set_progress_bar_config(disable=None)

            # check if vanilla attention processors are used
            for _, module in sd_pipe.unet.named_modules():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, (AttnProcessor, AttnProcessor2_0))

            # load LoRA weight file
            sd_pipe.load_lora_weights(tmpdirname)

            # check if lora attention processors are used
            for _, module in sd_pipe.unet.named_modules():
                if isinstance(module, Attention):
                    attn_proc_class = (
                        LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                    )
                    self.assertIsInstance(module.processor, attn_proc_class)

    def test_unload_lora_sd(self):
        pipeline_components, lora_components = self.get_dummy_components()
        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        sd_pipe = StableDiffusionPipeline(**pipeline_components)

        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Unload LoRA parameters.
        sd_pipe.unload_lora_weights()
        original_images_two = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice_two = original_images_two[0, -3:, -3:, -1]

        assert not np.allclose(
            orig_image_slice, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert not np.allclose(
            orig_image_slice_two, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert np.allclose(
            orig_image_slice, orig_image_slice_two, atol=1e-3
        ), "Unloading LoRA parameters should lead to results similar to what was obtained with the pipeline without any LoRA parameters."

    @unittest.skipIf(torch_device != "cuda", "This test is supposed to run on GPU")
    def test_lora_unet_attn_processors_with_xformers(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.create_lora_weight_file(tmpdirname)

            pipeline_components, _ = self.get_dummy_components()
            sd_pipe = StableDiffusionPipeline(**pipeline_components)
            sd_pipe = sd_pipe.to(torch_device)
            sd_pipe.set_progress_bar_config(disable=None)

            # enable XFormers
            sd_pipe.enable_xformers_memory_efficient_attention()

            # check if xFormers attention processors are used
            for _, module in sd_pipe.unet.named_modules():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, XFormersAttnProcessor)

            # load LoRA weight file
            sd_pipe.load_lora_weights(tmpdirname)

            # check if lora attention processors are used
            for _, module in sd_pipe.unet.named_modules():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, LoRAXFormersAttnProcessor)

            # unload lora weights
            sd_pipe.unload_lora_weights()

            # check if attention processors are reverted back to xFormers
            for _, module in sd_pipe.unet.named_modules():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, XFormersAttnProcessor)

    @unittest.skipIf(torch_device != "cuda", "This test is supposed to run on GPU")
    def test_lora_save_load_with_xformers(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs()

        # enable XFormers
        sd_pipe.enable_xformers_memory_efficient_attention()

        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))


class SDXLLoraLoaderMixinTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        unet_lora_attn_procs, unet_lora_layers = create_unet_lora_layers(unet)
        text_encoder_one_lora_layers = create_text_encoder_lora_layers(text_encoder)
        text_encoder_two_lora_layers = create_text_encoder_lora_layers(text_encoder_2)

        pipeline_components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
        }
        lora_components = {
            "unet_lora_layers": unet_lora_layers,
            "text_encoder_one_lora_layers": text_encoder_one_lora_layers,
            "text_encoder_two_lora_layers": text_encoder_two_lora_layers,
            "unet_lora_attn_procs": unet_lora_attn_procs,
        }
        return pipeline_components, lora_components

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

    def test_lora_save_load(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs()

        original_images = sd_pipe(**pipeline_inputs).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))

    def test_unload_lora_sdxl(self):
        pipeline_components, lora_components = self.get_dummy_components()
        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)

        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        # Emulate training.
        set_lora_weights(lora_components["unet_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_one_lora_layers"].parameters(), randn_weight=True)
        set_lora_weights(lora_components["text_encoder_two_lora_layers"].parameters(), randn_weight=True)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_layers"],
                text_encoder_lora_layers=lora_components["text_encoder_one_lora_layers"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Unload LoRA parameters.
        sd_pipe.unload_lora_weights()
        original_images_two = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice_two = original_images_two[0, -3:, -3:, -1]

        assert not np.allclose(
            orig_image_slice, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert not np.allclose(
            orig_image_slice_two, lora_image_slice
        ), "LoRA parameters should lead to a different image slice."
        assert np.allclose(
            orig_image_slice, orig_image_slice_two, atol=1e-3
        ), "Unloading LoRA parameters should lead to results similar to what was obtained with the pipeline without any LoRA parameters."


@slow
@require_torch_gpu
class LoraIntegrationTests(unittest.TestCase):
    def test_dreambooth_old_format(self):
        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "hf-internal-testing/lora_dreambooth_dog_example"
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe(
            "A photo of a sks dog floating in the river", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.7207, 0.6787, 0.6010, 0.7478, 0.6838, 0.6064, 0.6984, 0.6443, 0.5785])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_dreambooth_text_encoder_new_format(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/lora-trained"
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe("A photo of a sks dog", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.6628, 0.6138, 0.5390, 0.6625, 0.6130, 0.5463, 0.6166, 0.5788, 0.5359])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_a1111(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None).to(
            torch_device
        )
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_vanilla_funetuning(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/sd-model-finetuned-lora-t4"
        card = RepoCard.load(lora_model_id)
        base_model_id = card.data.to_dict()["base_model"]

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe = pipe.to(torch_device)
        pipe.load_lora_weights(lora_model_id)

        images = pipe("A pokemon with blue eyes.", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.7406, 0.699, 0.5963, 0.7493, 0.7045, 0.6096, 0.6886, 0.6388, 0.583])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_unload_kohya_lora(self):
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None).to(
            torch_device
        )
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=1e-3))

    def test_load_unload_load_kohya_lora(self):
        # This test ensures that a Kohya-style LoRA can be safely unloaded and then loaded
        # without introducing any side-effects. Even though the test uses a Kohya-style
        # LoRA, the underlying adapter handling mechanism is format-agnostic.
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None).to(
            torch_device
        )
        initial_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=1e-3))

        # make sure we can load a LoRA again after unloading and they don't have
        # any undesired effects.
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images_again = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        ).images
        lora_images_again = lora_images_again[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(lora_images, lora_images_again, atol=1e-3))
