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
import copy
import os
import random
import tempfile
import time
import unittest

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub.repocard import RepoCard
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin, StableDiffusionXLLoraLoaderMixin
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    deprecate_after_peft_backend,
    floats_tensor,
    load_image,
    nightly,
    require_torch_gpu,
    slow,
    torch_device,
)


def text_encoder_attn_modules(text_encoder: nn.Module):
    """Fetches the attention modules from `text_encoder`."""
    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_lora_state_dict(text_encoder: nn.Module):
    """Returns the LoRA state dict of the `text_encoder`. Assumes that `_modify_text_encoder()` was already called on it."""
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


def create_unet_lora_layers(unet: nn.Module, rank=4, mock_weights=True):
    """Creates and returns the LoRA state dict for the UNet."""
    # So that we accidentally don't end up using the in-place modified UNet.
    unet_lora_parameters = []

    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                rank=rank,
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                rank=rank,
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                rank=rank,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=rank,
            )
        )

        if mock_weights:
            with torch.no_grad():
                attn_module.to_q.lora_layer.up.weight += 1
                attn_module.to_k.lora_layer.up.weight += 1
                attn_module.to_v.lora_layer.up.weight += 1
                attn_module.to_out[0].lora_layer.up.weight += 1

        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

    unet_lora_sd = unet_lora_state_dict(unet)
    # Unload LoRA.
    unet.unload_lora()

    return unet_lora_parameters, unet_lora_sd


def create_3d_unet_lora_layers(unet: nn.Module, rank=4, mock_weights=True):
    """Creates and returns the LoRA state dict for the 3D UNet."""
    for attn_processor_name in unet.attn_processors.keys():
        has_cross_attention = attn_processor_name.endswith("attn2.processor") and not (
            attn_processor_name.startswith("transformer_in") or "temp_attentions" in attn_processor_name.split(".")
        )
        cross_attention_dim = unet.config.cross_attention_dim if has_cross_attention else None

        if attn_processor_name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif attn_processor_name.startswith("up_blocks"):
            block_id = int(attn_processor_name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif attn_processor_name.startswith("down_blocks"):
            block_id = int(attn_processor_name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        elif attn_processor_name.startswith("transformer_in"):
            # Note that the `8 * ...` comes from: https://github.com/huggingface/diffusers/blob/7139f0e874f10b2463caa8cbd585762a309d12d6/src/diffusers/models/unet_3d_condition.py#L148
            hidden_size = 8 * unet.config.attention_head_dim

        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=min(attn_module.to_q.in_features, hidden_size),
                out_features=attn_module.to_q.out_features
                if cross_attention_dim is None
                else max(attn_module.to_q.out_features, cross_attention_dim),
                rank=rank,
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=min(attn_module.to_k.in_features, hidden_size),
                out_features=attn_module.to_k.out_features
                if cross_attention_dim is None
                else max(attn_module.to_k.out_features, cross_attention_dim),
                rank=rank,
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=min(attn_module.to_v.in_features, hidden_size),
                out_features=attn_module.to_v.out_features
                if cross_attention_dim is None
                else max(attn_module.to_v.out_features, cross_attention_dim),
                rank=rank,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=min(attn_module.to_out[0].in_features, hidden_size),
                out_features=attn_module.to_out[0].out_features
                if cross_attention_dim is None
                else max(attn_module.to_out[0].out_features, cross_attention_dim),
                rank=rank,
            )
        )

        if mock_weights:
            with torch.no_grad():
                attn_module.to_q.lora_layer.up.weight += 1
                attn_module.to_k.lora_layer.up.weight += 1
                attn_module.to_v.lora_layer.up.weight += 1
                attn_module.to_out[0].lora_layer.up.weight += 1

    unet_lora_sd = unet_lora_state_dict(unet)

    # Unload LoRA.
    unet.unload_lora()

    return unet_lora_sd


def set_lora_weights(lora_attn_parameters, randn_weight=False, var=1.0):
    """Randomizes the LoRA params if specified."""
    if not isinstance(lora_attn_parameters, dict):
        with torch.no_grad():
            for parameter in lora_attn_parameters:
                if randn_weight:
                    parameter[:] = torch.randn_like(parameter) * var
                else:
                    torch.zero_(parameter)
    else:
        if randn_weight:
            modified_state_dict = {k: torch.rand_like(v) * var for k, v in lora_attn_parameters.items()}
        else:
            modified_state_dict = {k: torch.zeros_like(v) * var for k, v in lora_attn_parameters.items()}
        return modified_state_dict


def state_dicts_almost_equal(sd1, sd2):
    sd1 = dict(sorted(sd1.items()))
    sd2 = dict(sorted(sd2.items()))

    models_are_equal = True
    for ten1, ten2 in zip(sd1.values(), sd2.values()):
        if (ten1 - ten2).abs().max() > 1e-3:
            models_are_equal = False

    return models_are_equal


@deprecate_after_peft_backend
class LoraLoaderMixinTests(unittest.TestCase):
    lora_rank = 4

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

        unet_lora_raw_params, unet_lora_params = create_unet_lora_layers(unet, rank=self.lora_rank)
        text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(
            text_encoder, dtype=torch.float32, rank=self.lora_rank
        )
        text_encoder_lora_params = text_encoder_lora_state_dict(text_encoder)
        # We call this to ensure that the effects of the in-place `_modify_text_encoder` have been erased.
        LoraLoaderMixin._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        pipeline_components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        lora_components = {
            "unet_lora_raw_params": unet_lora_raw_params,
            "unet_lora_params": unet_lora_params,
            "text_encoder_lora_params": text_encoder_lora_params,
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
            unet_lora_layers=lora_components["unet_lora_params"],
            text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
        )
        self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))

    @unittest.skipIf(not torch.cuda.is_available() or not is_xformers_available(), reason="xformers requires cuda")
    def test_stable_diffusion_xformers_attn_processors(self):
        # disable_full_determinism()
        device = "cuda"  # ensure determinism for the device-dependent torch.Generator
        components, _ = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, inputs = self.get_dummy_inputs()

        # run xformers attention
        sd_pipe.enable_xformers_memory_efficient_attention()
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

    def test_stable_diffusion_lora(self):
        components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)
        sd_pipe.unet.set_default_attn_processor()

        # forward 1
        _, _, inputs = self.get_dummy_inputs()

        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        # set lora layers
        sd_pipe.unet.load_attn_procs(lora_components["unet_lora_params"])

        # forward 2
        _, _, inputs = self.get_dummy_inputs()

        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.0})
        image = output.images
        image_slice_1 = image[0, -3:, -3:, -1]

        # forward 3
        _, _, inputs = self.get_dummy_inputs()

        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.5})
        image = output.images
        image_slice_2 = image[0, -3:, -3:, -1]

        assert np.abs(image_slice - image_slice_1).max() < 1e-2
        assert np.abs(image_slice - image_slice_2).max() > 1e-2

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
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))

    def test_lora_save_load_no_safe_serialization(self):
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
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                safe_serialization=False,
            )

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
        text_encoder_lora_params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)
        text_encoder_lora_params = set_lora_weights(
            text_encoder_lora_state_dict(pipe.text_encoder), randn_weight=False
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=None,
                text_encoder_lora_layers=text_encoder_lora_params,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.load_lora_weights(tmpdirname)

        # inference with lora
        outputs_with_lora = pipe.text_encoder(**dummy_tokens)[0]
        assert outputs_with_lora.shape == (1, 77, 32)

        assert torch.allclose(
            outputs_without_lora, outputs_with_lora
        ), "lora_up_weight are all zero, so the lora outputs should be the same to without lora outputs"

        # monkey patch
        pipeline_components, _ = self.get_dummy_components()
        pipe = StableDiffusionPipeline(**pipeline_components)

        text_encoder_lora_params = pipe._modify_text_encoder(pipe.text_encoder, pipe.lora_scale)
        text_encoder_lora_params = set_lora_weights(
            text_encoder_lora_state_dict(pipe.text_encoder), randn_weight=True, var=0.1
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=None,
                text_encoder_lora_layers=text_encoder_lora_params,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.load_lora_weights(tmpdirname)

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
        params = set_lora_weights(text_encoder_lora_state_dict(pipe.text_encoder), var=0.1, randn_weight=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=None,
                text_encoder_lora_layers=params,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.load_lora_weights(tmpdirname)

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
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
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
                    self.assertIsNotNone(module.to_q.lora_layer)
                    self.assertIsNotNone(module.to_k.lora_layer)
                    self.assertIsNotNone(module.to_v.lora_layer)
                    self.assertIsNotNone(module.to_out[0].lora_layer)

    def test_unload_lora_sd(self):
        pipeline_components, lora_components = self.get_dummy_components()
        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        sd_pipe = StableDiffusionPipeline(**pipeline_components)
        sd_pipe.unet.set_default_attn_processor()

        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            LoraLoaderMixin.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
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

    @unittest.skipIf(torch_device != "cuda" or not is_xformers_available(), "This test is supposed to run on GPU")
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
                    self.assertIsNotNone(module.to_q.lora_layer)
                    self.assertIsNotNone(module.to_k.lora_layer)
                    self.assertIsNotNone(module.to_v.lora_layer)
                    self.assertIsNotNone(module.to_out[0].lora_layer)

            # unload lora weights
            sd_pipe.unload_lora_weights()

            # check if attention processors are reverted back to xFormers
            for _, module in sd_pipe.unet.named_modules():
                if isinstance(module, Attention):
                    self.assertIsInstance(module.processor, XFormersAttnProcessor)

    @unittest.skipIf(torch_device != "cuda" or not is_xformers_available(), "This test is supposed to run on GPU")
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
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_layers"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))


@deprecate_after_peft_backend
class SDInpaintLoraMixinTests(unittest.TestCase):
    lora_rank = 4

    def get_dummy_inputs(self, device, seed=0, img_res=64, output_pil=True):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        if output_pil:
            # Get random floats in [0, 1] as image
            image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
            image = image.cpu().permute(0, 2, 3, 1)[0]
            mask_image = torch.ones_like(image)
            # Convert image and mask_image to [0, 255]
            image = 255 * image
            mask_image = 255 * mask_image
            # Convert to PIL image
            init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((img_res, img_res))
            mask_image = Image.fromarray(np.uint8(mask_image)).convert("RGB").resize((img_res, img_res))
        else:
            # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
            image = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)).to(device)
            # Convert image to [-1, 1]
            init_image = 2.0 * image - 1.0
            mask_image = torch.ones((1, 1, img_res, img_res), device=device)

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
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
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        unet_lora_raw_params, unet_lora_params = create_unet_lora_layers(unet, rank=self.lora_rank)
        text_encoder_lora_params = StableDiffusionXLLoraLoaderMixin._modify_text_encoder(
            text_encoder, dtype=torch.float32, rank=self.lora_rank
        )
        text_encoder_lora_params = set_lora_weights(
            text_encoder_lora_state_dict(text_encoder), randn_weight=True, var=0.1
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        lora_components = {
            "unet_lora_raw_params": unet_lora_raw_params,
            "unet_lora_params": unet_lora_params,
            "text_encoder_lora_params": text_encoder_lora_params,
        }
        return components, lora_components

    def test_stable_diffusion_inpaint_lora(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)
        sd_pipe.unet.set_default_attn_processor()

        # forward 1
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        # set lora layers
        sd_pipe.unet.load_attn_procs(lora_components["unet_lora_params"])

        # forward 2
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.0})
        image = output.images
        image_slice_1 = image[0, -3:, -3:, -1]

        # forward 3
        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.5})
        image = output.images
        image_slice_2 = image[0, -3:, -3:, -1]

        assert np.abs(image_slice - image_slice_1).max() < 1e-2
        assert np.abs(image_slice - image_slice_2).max() > 1e-2


@deprecate_after_peft_backend
class SDXLLoraLoaderMixinTests(unittest.TestCase):
    lora_rank = 4

    def get_dummy_components(self, modify_text_encoder=True):
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

        _, unet_lora_params = create_unet_lora_layers(unet, rank=self.lora_rank)

        if modify_text_encoder:
            _ = StableDiffusionXLLoraLoaderMixin._modify_text_encoder(
                text_encoder, dtype=torch.float32, rank=self.lora_rank
            )
            text_encoder_lora_params = text_encoder_lora_state_dict(text_encoder)
            StableDiffusionXLLoraLoaderMixin._remove_text_encoder_monkey_patch_classmethod(text_encoder)

            _ = StableDiffusionXLLoraLoaderMixin._modify_text_encoder(
                text_encoder_2, dtype=torch.float32, rank=self.lora_rank
            )
            text_encoder_two_lora_params = text_encoder_lora_state_dict(text_encoder_2)
            StableDiffusionXLLoraLoaderMixin._remove_text_encoder_monkey_patch_classmethod(text_encoder_2)
        else:
            text_encoder_lora_params = None
            text_encoder_two_lora_params = None

        pipeline_components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }
        lora_components = {
            "unet_lora_params": unet_lora_params,
            "text_encoder_lora_params": text_encoder_lora_params,
            "text_encoder_two_lora_params": text_encoder_two_lora_params,
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
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(tmpdirname)

        lora_images = sd_pipe(**pipeline_inputs).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Outputs shouldn't match.
        self.assertFalse(torch.allclose(torch.from_numpy(orig_image_slice), torch.from_numpy(lora_image_slice)))

    def test_unload_lora_sdxl(self):
        pipeline_components, lora_components = self.get_dummy_components()
        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe.unet.set_default_attn_processor()

        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
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

    def test_load_lora_locally(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=False,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"))

        sd_pipe.unload_lora_weights()

    def test_text_encoder_lora_state_dict_unchanged(self):
        pipeline_components, lora_components = self.get_dummy_components(modify_text_encoder=False)
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)

        text_encoder_1_sd_keys = sorted(sd_pipe.text_encoder.state_dict().keys())
        text_encoder_2_sd_keys = sorted(sd_pipe.text_encoder_2.state_dict().keys())

        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # Modify the text encoder.
        _ = StableDiffusionXLLoraLoaderMixin._modify_text_encoder(
            sd_pipe.text_encoder, dtype=torch.float32, rank=self.lora_rank
        )
        lora_components["text_encoder_lora_params"] = set_lora_weights(
            text_encoder_lora_state_dict(sd_pipe.text_encoder), randn_weight=True, var=0.1
        )
        _ = StableDiffusionXLLoraLoaderMixin._modify_text_encoder(
            sd_pipe.text_encoder_2, dtype=torch.float32, rank=self.lora_rank
        )
        lora_components["text_encoder_two_lora_params"] = set_lora_weights(
            text_encoder_lora_state_dict(sd_pipe.text_encoder_2), randn_weight=True, var=0.1
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=False,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"))

            text_encoder_1_sd_keys_2 = sorted(sd_pipe.text_encoder.state_dict().keys())
            text_encoder_2_sd_keys_2 = sorted(sd_pipe.text_encoder_2.state_dict().keys())

        sd_pipe.unload_lora_weights()

        text_encoder_1_sd_keys_3 = sorted(sd_pipe.text_encoder.state_dict().keys())
        text_encoder_2_sd_keys_3 = sorted(sd_pipe.text_encoder_2.state_dict().keys())

        # default & unloaded LoRA weights should have identical state_dicts
        assert text_encoder_1_sd_keys == text_encoder_1_sd_keys_3
        # default & loaded LoRA weights should NOT have identical state_dicts
        assert text_encoder_1_sd_keys != text_encoder_1_sd_keys_2

        # default & unloaded LoRA weights should have identical state_dicts
        assert text_encoder_2_sd_keys == text_encoder_2_sd_keys_3
        # default & loaded LoRA weights should NOT have identical state_dicts
        assert text_encoder_2_sd_keys != text_encoder_2_sd_keys_2

    def test_load_lora_locally_safetensors(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.unload_lora_weights()

    def test_lora_fuse_nan(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        # corrupt one LoRA weight with `inf` values
        with torch.no_grad():
            sd_pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.to_q.lora_layer.down.weight += float(
                "NaN"
            )

        # with `safe_fusing=True` we should see an Error
        with self.assertRaises(ValueError):
            sd_pipe.fuse_lora(safe_fusing=True)

        # without we should not see an error, but every image will be black
        sd_pipe.fuse_lora(safe_fusing=False)

        out = sd_pipe("test", num_inference_steps=2, output_type="np").images

        assert np.isnan(out).all()

    def test_lora_fusion(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.fuse_lora()
        lora_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        self.assertFalse(np.allclose(orig_image_slice, lora_image_slice, atol=1e-3))

    def test_unfuse_lora(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)
        sd_pipe.unet.set_default_attn_processor()

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.fuse_lora()
        lora_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Reverse LoRA fusion.
        sd_pipe.unfuse_lora()
        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        orig_image_slice_two = original_images[0, -3:, -3:, -1]

        assert not np.allclose(
            orig_image_slice, lora_image_slice
        ), "Fusion of LoRAs should lead to a different image slice."
        assert not np.allclose(
            orig_image_slice_two, lora_image_slice
        ), "Fusion of LoRAs should lead to a different image slice."
        assert np.allclose(
            orig_image_slice, orig_image_slice_two, atol=1e-3
        ), "Reversing LoRA fusion should lead to results similar to what was obtained with the pipeline without any LoRA parameters."

    def test_lora_fusion_is_not_affected_by_unloading(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        _ = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.fuse_lora()
        lora_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice = lora_images[0, -3:, -3:, -1]

        # Unload LoRA parameters.
        sd_pipe.unload_lora_weights()
        images_with_unloaded_lora = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        images_with_unloaded_lora_slice = images_with_unloaded_lora[0, -3:, -3:, -1]

        assert (
            np.abs(lora_image_slice - images_with_unloaded_lora_slice).max() < 2e-1
        ), "`unload_lora_weights()` should have not effect on the semantics of the results as the LoRA parameters were fused."

    def test_fuse_lora_with_different_scales(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        _ = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.fuse_lora(lora_scale=1.0)
        lora_images_scale_one = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice_scale_one = lora_images_scale_one[0, -3:, -3:, -1]

        # Reverse LoRA fusion.
        sd_pipe.unfuse_lora()

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.fuse_lora(lora_scale=0.5)
        lora_images_scale_0_5 = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice_scale_0_5 = lora_images_scale_0_5[0, -3:, -3:, -1]

        assert not np.allclose(
            lora_image_slice_scale_one, lora_image_slice_scale_0_5, atol=1e-03
        ), "Different LoRA scales should influence the outputs accordingly."

    def test_with_different_scales(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)
        sd_pipe.unet.set_default_attn_processor()

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)
        original_images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        original_imagee_slice = original_images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        lora_images_scale_one = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice_scale_one = lora_images_scale_one[0, -3:, -3:, -1]

        lora_images_scale_0_5 = sd_pipe(
            **pipeline_inputs, generator=torch.manual_seed(0), cross_attention_kwargs={"scale": 0.5}
        ).images
        lora_image_slice_scale_0_5 = lora_images_scale_0_5[0, -3:, -3:, -1]

        lora_images_scale_0_0 = sd_pipe(
            **pipeline_inputs, generator=torch.manual_seed(0), cross_attention_kwargs={"scale": 0.0}
        ).images
        lora_image_slice_scale_0_0 = lora_images_scale_0_0[0, -3:, -3:, -1]

        assert not np.allclose(
            lora_image_slice_scale_one, lora_image_slice_scale_0_5, atol=1e-03
        ), "Different LoRA scales should influence the outputs accordingly."

        assert np.allclose(
            original_imagee_slice, lora_image_slice_scale_0_0, atol=1e-03
        ), "LoRA scale of 0.0 shouldn't be different from the results without LoRA."

    def test_with_different_scales_fusion_equivalence(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)
        sd_pipe.unet.set_default_attn_processor()

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        images = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        images_slice = images[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        lora_images_scale_0_5 = sd_pipe(
            **pipeline_inputs, generator=torch.manual_seed(0), cross_attention_kwargs={"scale": 0.5}
        ).images
        lora_image_slice_scale_0_5 = lora_images_scale_0_5[0, -3:, -3:, -1]

        sd_pipe.fuse_lora(lora_scale=0.5)
        lora_images_scale_0_5_fusion = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice_scale_0_5_fusion = lora_images_scale_0_5_fusion[0, -3:, -3:, -1]

        assert np.allclose(
            lora_image_slice_scale_0_5, lora_image_slice_scale_0_5_fusion, atol=1e-03
        ), "Fusion shouldn't affect the results when calling the pipeline with a non-default LoRA scale."

        sd_pipe.unfuse_lora()
        images_unfused = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        images_slice_unfused = images_unfused[0, -3:, -3:, -1]

        assert np.allclose(images_slice, images_slice_unfused, atol=1e-03), "Unfused should match no LoRA"

        assert not np.allclose(
            images_slice, lora_image_slice_scale_0_5, atol=1e-03
        ), "0.5 scale and no scale shouldn't match"

    def test_save_load_fused_lora_modules(self):
        pipeline_components, lora_components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**pipeline_components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        _, _, pipeline_inputs = self.get_dummy_inputs(with_generator=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=tmpdirname,
                unet_lora_layers=lora_components["unet_lora_params"],
                text_encoder_lora_layers=lora_components["text_encoder_lora_params"],
                text_encoder_2_lora_layers=lora_components["text_encoder_two_lora_params"],
                safe_serialization=True,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            sd_pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        sd_pipe.fuse_lora()
        lora_images_fusion = sd_pipe(**pipeline_inputs, generator=torch.manual_seed(0)).images
        lora_image_slice_fusion = lora_images_fusion[0, -3:, -3:, -1]

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd_pipe.save_pretrained(tmpdirname)
            sd_pipe_loaded = StableDiffusionXLPipeline.from_pretrained(tmpdirname).to(torch_device)

        loaded_lora_images = sd_pipe_loaded(**pipeline_inputs, generator=torch.manual_seed(0)).images
        loaded_lora_image_slice = loaded_lora_images[0, -3:, -3:, -1]

        assert np.allclose(
            lora_image_slice_fusion, loaded_lora_image_slice, atol=1e-03
        ), "The pipeline was serialized with LoRA parameters fused inside of the respected modules. The loaded pipeline should yield proper outputs, henceforth."


@deprecate_after_peft_backend
class UNet2DConditionLoRAModelTests(unittest.TestCase):
    model_class = UNet2DConditionModel
    main_input_name = "sample"
    lora_rank = 4

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32), rng=random.Random(0)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_lora_at_different_scales(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        _, lora_params = create_unet_lora_layers(model, rank=self.lora_rank)

        # make sure we can set a list of attention processors
        model.load_attn_procs(lora_params)
        model.to(torch_device)

        with torch.no_grad():
            sample2 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
            sample3 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
            sample4 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample1 - sample2).abs().max() < 3e-3
        assert (sample3 - sample4).abs().max() < 3e-3

        # sample 2 and sample 3 should be different
        assert (sample2 - sample3).abs().max() > 1e-4

    def test_lora_on_off(self, expected_max_diff=1e-3):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = (8, 16)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            old_sample = model(**inputs_dict).sample

        _, lora_params = create_unet_lora_layers(model, rank=self.lora_rank)
        model.load_attn_procs(lora_params)

        with torch.no_grad():
            sample = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample

        # Unload LoRA.
        model.unload_lora()

        with torch.no_grad():
            new_sample = model(**inputs_dict).sample

        max_diff_new_sample = (sample - new_sample).abs().max()
        max_diff_old_sample = (sample - old_sample).abs().max()

        assert max_diff_new_sample < expected_max_diff
        assert max_diff_old_sample < expected_max_diff

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_lora_xformers_on_off(self, expected_max_diff=6e-4):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = (8, 16)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)
        _, lora_params = create_unet_lora_layers(model, rank=self.lora_rank)
        model.load_attn_procs(lora_params)

        # default
        with torch.no_grad():
            sample = model(**inputs_dict).sample

            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample

            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample

        max_diff_on_sample = (sample - on_sample).abs().max()
        max_diff_off_sample = (sample - off_sample).abs().max()

        assert max_diff_on_sample < expected_max_diff
        assert max_diff_off_sample < expected_max_diff


@deprecate_after_peft_backend
class UNet3DConditionLoRAModelTests(unittest.TestCase):
    model_class = UNet3DConditionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        num_frames = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels, num_frames) + sizes, rng=random.Random(0)).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 32), rng=random.Random(0)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 4, 32, 32)

    @property
    def output_shape(self):
        return (4, 4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_lora_at_different_scales(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = 8

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        unet_lora_params = create_3d_unet_lora_layers(model)

        # make sure we can set a list of attention processors
        model.load_attn_procs(unet_lora_params)
        model.to(torch_device)

        with torch.no_grad():
            sample2 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.0}).sample
            sample3 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample
            sample4 = model(**inputs_dict, cross_attention_kwargs={"scale": 0.5}).sample

        assert (sample1 - sample2).abs().max() < 3e-3
        assert (sample3 - sample4).abs().max() < 3e-3

        # sample 2 and sample 3 should be different
        assert (sample2 - sample3).abs().max() > 3e-3


@slow
@deprecate_after_peft_backend
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

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_lycoris(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/Amixx", safety_checker=None, use_safetensors=True, variant="fp16"
        ).to(torch_device)
        lora_model_id = "hf-internal-testing/edgLycorisMugler-light"
        lora_filename = "edgLycorisMugler-light.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.6463, 0.658, 0.599, 0.6542, 0.6512, 0.6213, 0.658, 0.6485, 0.6017])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_a1111_with_model_cpu_offload(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_a1111_with_sequential_cpu_offload(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/Counterfeit-V2.5", safety_checker=None)
        pipe.enable_sequential_cpu_offload()
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_kohya_sd_v15_with_higher_dimensions(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None).to(
            torch_device
        )
        lora_model_id = "hf-internal-testing/urushisato-lora"
        lora_filename = "urushisato_v15.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7165, 0.6616, 0.5833, 0.7504, 0.6718, 0.587, 0.6871, 0.6361, 0.5694])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

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

    def test_sdxl_0_9_lora_one(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
        lora_model_id = "hf-internal-testing/sdxl-0.9-daiton-lora"
        lora_filename = "daiton-xl-lora-test.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3838, 0.3482, 0.3588, 0.3162, 0.319, 0.3369, 0.338, 0.3366, 0.3213])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_sdxl_0_9_lora_two(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
        lora_model_id = "hf-internal-testing/sdxl-0.9-costumes-lora"
        lora_filename = "saijo.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3137, 0.3269, 0.3355, 0.255, 0.2577, 0.2563, 0.2679, 0.2758, 0.2626])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_sdxl_0_9_lora_three(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9")
        lora_model_id = "hf-internal-testing/sdxl-0.9-kamepan-lora"
        lora_filename = "kame_sdxl_v2-000020-16rank.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4015, 0.3761, 0.3616, 0.3745, 0.3462, 0.3337, 0.3564, 0.3649, 0.3468])

        self.assertTrue(np.allclose(images, expected, atol=5e-3))

    def test_sdxl_1_0_lora(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_sdxl_1_0_lora_fusion(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        # This way we also test equivalence between LoRA fusion and the non-fusion behaviour.
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        self.assertTrue(np.allclose(images, expected, atol=1e-4))

    def test_sdxl_1_0_lora_unfusion(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_with_fusion = images[0, -3:, -3:, -1].flatten()

        pipe.unfuse_lora()
        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_without_fusion = images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(images_with_fusion, images_without_fusion, atol=1e-3))

    def test_sdxl_1_0_lora_unfusion_effectivity(self):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        original_image_slice = images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        _ = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        pipe.unfuse_lora()
        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_without_fusion_slice = images[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(original_image_slice, images_without_fusion_slice, atol=1e-3))

    def test_sdxl_1_0_lora_fusion_efficiency(self):
        generator = torch.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        start_time = time.time()
        for _ in range(3):
            pipe(
                "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
            ).images
        end_time = time.time()
        elapsed_time_non_fusion = end_time - start_time

        del pipe

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, torch_dtype=torch.float16)
        pipe.fuse_lora()
        pipe.enable_model_cpu_offload()

        generator = torch.Generator().manual_seed(0)
        start_time = time.time()
        for _ in range(3):
            pipe(
                "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
            ).images
        end_time = time.time()
        elapsed_time_fusion = end_time - start_time

        self.assertTrue(elapsed_time_fusion < elapsed_time_non_fusion)

    def test_sdxl_1_0_last_ben(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "TheLastBen/Papercut_SDXL"
        lora_filename = "papercut.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe("papercut.safetensors", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.5244, 0.4347, 0.4312, 0.4246, 0.4398, 0.4409, 0.4884, 0.4938, 0.4094])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_sdxl_1_0_fuse_unfuse_all(self):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        text_encoder_1_sd = copy.deepcopy(pipe.text_encoder.state_dict())
        text_encoder_2_sd = copy.deepcopy(pipe.text_encoder_2.state_dict())
        unet_sd = copy.deepcopy(pipe.unet.state_dict())

        pipe.load_lora_weights(
            "davizca87/sun-flower", weight_name="snfw3rXL-000004.safetensors", torch_dtype=torch.float16
        )
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        pipe.unfuse_lora()

        assert state_dicts_almost_equal(text_encoder_1_sd, pipe.text_encoder.state_dict())
        assert state_dicts_almost_equal(text_encoder_2_sd, pipe.text_encoder_2.state_dict())
        assert state_dicts_almost_equal(unet_sd, pipe.unet.state_dict())

    def test_sdxl_1_0_lora_with_sequential_cpu_offloading(self):
        generator = torch.Generator().manual_seed(0)

        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_sequential_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        self.assertTrue(np.allclose(images, expected, atol=1e-3))

    def test_canny_lora(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet
        )
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
        pipe.enable_sequential_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "corgi"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (768, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4574, 0.4461, 0.4435, 0.4462, 0.4396, 0.439, 0.4474, 0.4486, 0.4333])
        assert np.allclose(original_image, expected_image, atol=1e-04)

    @nightly
    def test_sequential_fuse_unfuse(self):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

        # 1. round
        pipe.load_lora_weights("Pclanglais/TintinIA")
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        image_slice = images[0, -3:, -3:, -1].flatten()

        pipe.unfuse_lora()

        # 2. round
        pipe.load_lora_weights("ProomptEngineer/pe-balloon-diffusion-style")
        pipe.fuse_lora()
        pipe.unfuse_lora()

        # 3. round
        pipe.load_lora_weights("ostris/crayon_style_lora_sdxl")
        pipe.fuse_lora()
        pipe.unfuse_lora()

        # 4. back to 1st round
        pipe.load_lora_weights("Pclanglais/TintinIA")
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        images_2 = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        image_slice_2 = images_2[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(image_slice, image_slice_2, atol=1e-3))
