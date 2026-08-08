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
import sys
import unittest

import numpy as np
import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Pipeline, Flux2Transformer2DModel

from ..testing_utils import floats_tensor, require_peft_backend, torch_device


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class Flux2LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = Flux2Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 16,
        "timestep_guidance_channels": 256,
        "axes_dims_rope": [4, 4, 4, 4],
    }
    transformer_cls = Flux2Transformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",),
        "up_block_types": ("UpDecoderBlock2D",),
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 1,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }
    vae_cls = AutoencoderKLFlux2

    tokenizer_cls, tokenizer_id = AutoProcessor, "hf-internal-testing/tiny-mistral3-diffusers"
    text_encoder_cls, text_encoder_id = Mistral3ForConditionalGeneration, "hf-internal-testing/tiny-mistral3-diffusers"
    denoiser_target_modules = ["to_qkv_mlp_proj", "to_k"]

    supports_text_encoder_loras = False

    @property
    def output_shape(self):
        return (1, 8, 8, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "a dog is dancing",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 8,
            "output_type": "np",
            "text_encoder_out_layers": (1,),
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    # Overriding because (1) text encoder LoRAs are not supported in Flux 2 and (2) because the Flux 2 single block
    # QKV projections are always fused, it has no `to_q` param as expected by the original test.
    def test_lora_fuse_nan(self):
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        # corrupt one LoRA weight with `inf` values
        with torch.no_grad():
            possible_tower_names = ["transformer_blocks", "single_transformer_blocks"]
            filtered_tower_names = [
                tower_name for tower_name in possible_tower_names if hasattr(pipe.transformer, tower_name)
            ]
            if len(filtered_tower_names) == 0:
                reason = f"`pipe.transformer` didn't have any of the following attributes: {possible_tower_names}."
                raise ValueError(reason)
            for tower_name in filtered_tower_names:
                transformer_tower = getattr(pipe.transformer, tower_name)
                is_single = "single" in tower_name
                if is_single:
                    transformer_tower[0].attn.to_qkv_mlp_proj.lora_A["adapter-1"].weight += float("inf")
                else:
                    transformer_tower[0].attn.to_k.lora_A["adapter-1"].weight += float("inf")

        # with `safe_fusing=True` we should see an Error
        with self.assertRaises(ValueError):
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

        # without we should not see an error, but every image will be black
        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)
        out = pipe(**inputs)[0]

        self.assertTrue(np.isnan(out).all())

    @unittest.skip("Not supported in Flux2.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Flux2.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Flux2.")
    def test_modify_padding_mode(self):
        pass


@require_peft_backend
class Flux2LoKrTests(unittest.TestCase):
    factor = 4

    def get_transformer(self):
        return Flux2Transformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=256,
            axes_dims_rope=[4, 4, 4, 4],
        )

    def make_factors(self, out_features, in_features):
        from peft.tuners.lokr.layer import factorization

        (out_l, out_k) = factorization(out_features, self.factor)
        (in_m, in_n) = factorization(in_features, self.factor)
        return torch.randn(out_l, in_m), torch.randn(out_k, in_n)

    def test_lokr_bfl_checkpoint_fuses_qkv_and_loads(self):
        # ai-toolkit Flux2 LoKr checkpoints use BFL layer names with LoKr applied to the fused QKV projections
        # of the double blocks; loading fuses the model's QKV projections and maps the adapter 1:1.
        from peft.tuners.lokr.layer import LoKrLayer

        torch.manual_seed(0)
        transformer = self.get_transformer()
        named = dict(transformer.named_modules())
        state_dict = {}
        expected_deltas = {}

        for bfl_path, diffusers_path, target in [
            ("single_blocks.0.linear1", "single_transformer_blocks.0.attn.to_qkv_mlp_proj", None),
            ("single_blocks.0.linear2", "single_transformer_blocks.0.attn.to_out", None),
            ("double_blocks.0.img_attn.proj", "transformer_blocks.0.attn.to_out.0", None),
            ("double_blocks.0.img_mlp.0", "transformer_blocks.0.ff.linear_in", None),
            ("double_blocks.0.img_attn.qkv", "transformer_blocks.0.attn.to_qkv", "transformer_blocks.0.attn.to_q"),
            (
                "double_blocks.0.txt_attn.qkv",
                "transformer_blocks.0.attn.to_added_qkv",
                "transformer_blocks.0.attn.add_q_proj",
            ),
        ]:
            # fused QKV modules do not exist before fusion; derive their dimensions from the query projection
            linear = named[target if target is not None else diffusers_path]
            out_features = linear.out_features if target is None else 3 * linear.out_features
            w1, w2 = self.make_factors(out_features, linear.in_features)
            state_dict[f"diffusion_model.{bfl_path}.lokr_w1"] = w1
            state_dict[f"diffusion_model.{bfl_path}.lokr_w2"] = w2
            state_dict[f"diffusion_model.{bfl_path}.alpha"] = torch.tensor(9999220736.0)
            expected_deltas[diffusers_path] = torch.kron(w1, w2)

        pipe = Flux2Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=None,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        )
        pipe.load_lora_weights(state_dict, adapter_name="default")

        self.assertTrue(transformer.transformer_blocks[0].attn.fused_projections)
        named = dict(transformer.named_modules())
        for module, expected in expected_deltas.items():
            layer = named[module]
            self.assertIsInstance(layer, LoKrLayer, module)
            delta = layer.get_delta_weight("default")
            self.assertLess((delta - expected).abs().max().item(), 1e-5, module)

    def test_lokr_lycoris_checkpoint(self):
        # LyCORIS wraps the diffusers model directly and encodes module paths with underscores under a
        # `lycoris_` prefix.
        from peft.tuners.lokr.layer import LoKrLayer

        from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin

        torch.manual_seed(0)
        transformer = self.get_transformer()
        named = dict(transformer.named_modules())
        state_dict = {}
        expected_deltas = {}

        for lycoris_path, diffusers_path in [
            (
                "lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj",
                "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            ),
            ("lycoris_transformer_blocks_0_attn_to_q", "transformer_blocks.0.attn.to_q"),
            ("lycoris_transformer_blocks_0_attn_to_out_0", "transformer_blocks.0.attn.to_out.0"),
            ("lycoris_transformer_blocks_0_ff_linear_in", "transformer_blocks.0.ff.linear_in"),
        ]:
            linear = named[diffusers_path]
            w1, w2 = self.make_factors(linear.out_features, linear.in_features)
            state_dict[f"{lycoris_path}.lokr_w1"] = w1
            state_dict[f"{lycoris_path}.lokr_w2"] = w2
            state_dict[f"{lycoris_path}.alpha"] = torch.tensor(16.0)
            expected_deltas[diffusers_path] = torch.kron(w1, w2)

        converted = Flux2LoraLoaderMixin.lora_state_dict(state_dict)
        transformer.load_lora_adapter(converted, prefix="transformer", adapter_name="default")

        named = dict(transformer.named_modules())
        for module, expected in expected_deltas.items():
            layer = named[module]
            self.assertIsInstance(layer, LoKrLayer, module)
            delta = layer.get_delta_weight("default")
            self.assertLess((delta - expected).abs().max().item(), 1e-5, module)
