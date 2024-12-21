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

import sys
import unittest

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    ConsisIDPipeline,
    ConsisIDTransformer3DModel,
)
from diffusers.utils.testing_utils import (
    floats_tensor,
    is_torch_version,
    require_peft_backend,
    skip_mps,
    torch_device,
)


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class ConsisIDLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = ConsisIDPipeline
    scheduler_cls = CogVideoXDPMScheduler
    scheduler_kwargs = {"timestep_spacing": "trailing"}
    scheduler_classes = [CogVideoXDPMScheduler]

    transformer_kwargs = {
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "in_channels": 8,
        "out_channels": 4,
        "time_embed_dim": 2,
        "text_embed_dim": 32,
        "num_layers": 1,
        "sample_width": 16,
        "sample_height": 16,
        "sample_frames": 9,
        "patch_size": 2,
        "temporal_compression_ratio": 4,
        "max_text_seq_length": 16,
        "cross_attn_interval": 1,
        "is_kps": False,
        "is_train_face": True,
        "cross_attn_dim_head": 1,
        "cross_attn_num_heads": 1,
        "LFE_id_dim": 2,
        "LFE_vit_dim": 2,
        "LFE_depth": 5,
        "LFE_dim_head": 8,
        "LFE_num_heads": 2,
        "LFE_num_id_token": 1,
        "LFE_num_querie": 1,
        "LFE_output_dim": 21,
        "LFE_ff_mult": 1,
    }
    transformer_cls = ConsisIDTransformer3DModel
    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        "up_block_types": (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        "block_out_channels": (8, 8, 8, 8),
        "latent_channels": 4,
        "layers_per_block": 1,
        "norm_num_groups": 2,
        "temporal_compression_ratio": 4,
    }
    vae_cls = AutoencoderKLCogVideoX
    tokenizer_cls, tokenizer_id = AutoTokenizer, "/storage/ysh/Ckpts/hf-internal-testing/tiny-random-t5/"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "/storage/ysh/Ckpts/hf-internal-testing/tiny-random-t5/"

    text_encoder_target_modules = ["q", "k", "v", "o"]

    @property
    def output_shape(self):
        return (1, 9, 16, 16, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 9
        num_latent_frames = 3
        sizes = (2, 2)

        generator = torch.manual_seed(0)
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        noise = floats_tensor((batch_size, num_latent_frames, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)
        id_vit_hidden = [torch.ones([batch_size, 2, 2]).to(torch_device)] * 5
        id_cond = torch.ones(batch_size, 2).to(torch_device)

        pipeline_inputs = {
            "image": image,
            "prompt": "dance monkey",
            "num_frames": num_frames,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": sequence_length,
            "id_vit_hidden": id_vit_hidden,
            "id_cond": id_cond,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    @skip_mps
    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cpu" and is_torch_version(">=", "2.5"),
        reason="Test currently fails on CPU and PyTorch 2.5.1 but not on PyTorch 2.4.1.",
        strict=True,
    )
    def test_lora_fuse_nan(self):
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)
            
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")

            self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

            # corrupt one LoRA weight with `inf` values
            with torch.no_grad():
                pipe.transformer.transformer_blocks[0].attn1.to_q.lora_A["adapter-1"].weight += float("inf")

            # with `safe_fusing=True` we should see an Error
            with self.assertRaises(ValueError):
                pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

            # without we should not see an error, but every image will be black
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)

            out = pipe(
                image=inputs["image"],
                prompt=inputs["prompt"],
                num_frames=inputs["num_frames"],
                num_inference_steps=inputs["num_inference_steps"],
                guidance_scale=inputs["guidance_scale"],
                height=inputs["height"],
                width=inputs["width"],
                max_sequence_length=inputs["max_sequence_length"],
                id_vit_hidden=inputs["id_vit_hidden"],
                id_cond=inputs["id_cond"],
                output_type=inputs["output_type"],
            )[0]

            self.assertTrue(np.isnan(out).all())

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    @unittest.skip("Not supported in ConsisID.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in ConsisID.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in ConsisID.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ConsisID.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ConsisID.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ConsisID.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ConsisID.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in ConsisID.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass
