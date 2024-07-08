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
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.utils.testing_utils import is_peft_available, require_peft_backend, require_torch_gpu, torch_device


if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

sys.path.append(".")

from utils import check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class SD3LoRATests(unittest.TestCase):
    pipeline_class = StableDiffusion3Pipeline

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_projection_dim=32,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            out_channels=4,
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
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def get_lora_config_for_transformer(self):
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        return lora_config

    def get_lora_config_for_text_encoders(self):
        text_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        return text_lora_config

    def test_simple_inference_with_transformer_lora_save_load(self):
        components = self.get_dummy_components()
        transformer_config = self.get_lora_config_for_transformer()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        pipe.transformer.add_adapter(transformer_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        inputs = self.get_dummy_inputs(torch_device)
        images_lora = pipe(**inputs).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            transformer_state_dict = get_peft_model_state_dict(pipe.transformer)

            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname,
                transformer_lora_layers=transformer_state_dict,
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.unload_lora_weights()

            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        inputs = self.get_dummy_inputs(torch_device)
        images_lora_from_pretrained = pipe(**inputs).images
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_clip_encoders_lora_save_load(self):
        components = self.get_dummy_components()
        transformer_config = self.get_lora_config_for_transformer()
        text_encoder_config = self.get_lora_config_for_text_encoders()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)

        pipe.transformer.add_adapter(transformer_config)
        pipe.text_encoder.add_adapter(text_encoder_config)
        pipe.text_encoder_2.add_adapter(text_encoder_config)

        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder.")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2.")

        inputs = self.get_dummy_inputs(torch_device)
        images_lora = pipe(**inputs).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            transformer_state_dict = get_peft_model_state_dict(pipe.transformer)
            text_encoder_one_state_dict = get_peft_model_state_dict(pipe.text_encoder)
            text_encoder_two_state_dict = get_peft_model_state_dict(pipe.text_encoder_2)

            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname,
                transformer_lora_layers=transformer_state_dict,
                text_encoder_lora_layers=text_encoder_one_state_dict,
                text_encoder_2_lora_layers=text_encoder_two_state_dict,
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.unload_lora_weights()

            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

        inputs = self.get_dummy_inputs(torch_device)
        images_lora_from_pretrained = pipe(**inputs).images
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text_encoder_one")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text_encoder_two")

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_transformer_lora_and_scale(self):
        components = self.get_dummy_components()
        transformer_lora_config = self.get_lora_config_for_transformer()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_no_lora = pipe(**inputs).images

        pipe.transformer.add_adapter(transformer_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        inputs = self.get_dummy_inputs(torch_device)
        output_lora = pipe(**inputs).images
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

        inputs = self.get_dummy_inputs(torch_device)
        output_lora_scale = pipe(**inputs, joint_attention_kwargs={"scale": 0.5}).images
        self.assertTrue(
            not np.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )

        inputs = self.get_dummy_inputs(torch_device)
        output_lora_0_scale = pipe(**inputs, joint_attention_kwargs={"scale": 0.0}).images
        self.assertTrue(
            np.allclose(output_no_lora, output_lora_0_scale, atol=1e-3, rtol=1e-3),
            "Lora + 0 scale should lead to same result as no LoRA",
        )

    def test_simple_inference_with_clip_encoders_lora_and_scale(self):
        components = self.get_dummy_components()
        transformer_lora_config = self.get_lora_config_for_transformer()
        text_encoder_config = self.get_lora_config_for_text_encoders()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_no_lora = pipe(**inputs).images

        pipe.transformer.add_adapter(transformer_lora_config)
        pipe.text_encoder.add_adapter(text_encoder_config)
        pipe.text_encoder_2.add_adapter(text_encoder_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text_encoder_one")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text_encoder_two")

        inputs = self.get_dummy_inputs(torch_device)
        output_lora = pipe(**inputs).images
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

        inputs = self.get_dummy_inputs(torch_device)
        output_lora_scale = pipe(**inputs, joint_attention_kwargs={"scale": 0.5}).images
        self.assertTrue(
            not np.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )

        inputs = self.get_dummy_inputs(torch_device)
        output_lora_0_scale = pipe(**inputs, joint_attention_kwargs={"scale": 0.0}).images
        self.assertTrue(
            np.allclose(output_no_lora, output_lora_0_scale, atol=1e-3, rtol=1e-3),
            "Lora + 0 scale should lead to same result as no LoRA",
        )

    def test_simple_inference_with_transformer_fused(self):
        components = self.get_dummy_components()
        transformer_lora_config = self.get_lora_config_for_transformer()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_no_lora = pipe(**inputs).images

        pipe.transformer.add_adapter(transformer_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        pipe.fuse_lora()
        # Fusing should still keep the LoRA layers
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        inputs = self.get_dummy_inputs(torch_device)
        ouput_fused = pipe(**inputs).images
        self.assertFalse(
            np.allclose(ouput_fused, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

    def test_simple_inference_with_transformer_fused_with_no_fusion(self):
        components = self.get_dummy_components()
        transformer_lora_config = self.get_lora_config_for_transformer()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_no_lora = pipe(**inputs).images

        pipe.transformer.add_adapter(transformer_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        inputs = self.get_dummy_inputs(torch_device)
        ouput_lora = pipe(**inputs).images

        pipe.fuse_lora()
        # Fusing should still keep the LoRA layers
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        inputs = self.get_dummy_inputs(torch_device)
        ouput_fused = pipe(**inputs).images
        self.assertFalse(
            np.allclose(ouput_fused, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )
        self.assertTrue(
            np.allclose(ouput_fused, ouput_lora, atol=1e-3, rtol=1e-3),
            "Fused lora output should be changed when LoRA isn't fused but still effective.",
        )

    def test_simple_inference_with_transformer_fuse_unfuse(self):
        components = self.get_dummy_components()
        transformer_lora_config = self.get_lora_config_for_transformer()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_no_lora = pipe(**inputs).images

        pipe.transformer.add_adapter(transformer_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        pipe.fuse_lora()
        # Fusing should still keep the LoRA layers
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        inputs = self.get_dummy_inputs(torch_device)
        ouput_fused = pipe(**inputs).images
        self.assertFalse(
            np.allclose(ouput_fused, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

        pipe.unfuse_lora()
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")
        inputs = self.get_dummy_inputs(torch_device)
        output_unfused_lora = pipe(**inputs).images
        self.assertTrue(
            np.allclose(ouput_fused, output_unfused_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

    @require_torch_gpu
    def test_sd3_lora(self):
        """
        Test loading the loras that are saved with the diffusers and peft formats.
        Related PR: https://github.com/huggingface/diffusers/pull/8584
        """
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        lora_model_id = "hf-internal-testing/tiny-sd3-loras"

        lora_filename = "lora_diffusers_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.unload_lora_weights()

        lora_filename = "lora_peft_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
