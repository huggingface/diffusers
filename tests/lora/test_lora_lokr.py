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
import unittest

import torch

from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Pipeline, Flux2Transformer2DModel, ZImageTransformer2DModel

from ..testing_utils import require_peft_backend


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

    def test_lokr_fused_qkv_checkpoint_refuses_when_unfused_projections_are_adapted(self):
        # A LoRA already injected on to_q would be orphaned by fuse_qkv_projections(), so loading must refuse.
        from peft import LoraConfig

        transformer = self.get_transformer()
        transformer.add_adapter(LoraConfig(r=2, target_modules=["to_q"]), adapter_name="plain_lora")
        linear = dict(transformer.named_modules())["transformer_blocks.0.attn.to_q"]
        w1, w2 = self.make_factors(3 * linear.out_features, linear.in_features)
        state_dict = {
            "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1": w1,
            "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": w2,
            "diffusion_model.double_blocks.0.img_attn.qkv.alpha": torch.tensor(9999220736.0),
        }
        pipe = Flux2Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=None,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        )
        with self.assertRaisesRegex(ValueError, "already loaded on the unfused projections"):
            pipe.load_lora_weights(state_dict, adapter_name="lokr")
        self.assertFalse(transformer.transformer_blocks[0].attn.fused_projections)

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


@require_peft_backend
class ZImageLoKrTests(unittest.TestCase):
    def get_transformer(self):
        return ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=16,
            dim=32,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=2,
            norm_eps=1e-5,
            qk_norm=True,
            cap_feat_dim=16,
            rope_theta=256.0,
            t_scale=1000.0,
            axes_dims=[8, 4, 4],
            axes_lens=[256, 32, 32],
        )

    def test_lokr_ai_toolkit_checkpoint(self):
        # ai-toolkit Z-Image LoKr checkpoints store dotted diffusers module paths under a `diffusion_model.`
        # prefix, with full Kronecker factors and a placeholder alpha, plus optionally rank-decomposed factors
        # with a meaningful alpha.
        from peft.tuners.lokr.layer import LoKrLayer, factorization

        from diffusers.loaders.lora_pipeline import ZImageLoraLoaderMixin

        torch.manual_seed(0)
        transformer = self.get_transformer()
        named = dict(transformer.named_modules())
        factor = 4
        state_dict = {}
        expected_deltas = {}

        for module in ["layers.0.attention.to_q", "layers.0.feed_forward.w1", "layers.0.adaLN_modulation.0"]:
            linear = named[module]
            (out_l, out_k) = factorization(linear.out_features, factor)
            (in_m, in_n) = factorization(linear.in_features, factor)
            w1 = torch.randn(out_l, in_m)
            w2 = torch.randn(out_k, in_n)
            state_dict[f"diffusion_model.{module}.lokr_w1"] = w1
            state_dict[f"diffusion_model.{module}.lokr_w2"] = w2
            # alpha applies no scaling when both factors are full matrices; ai-toolkit stores a placeholder
            state_dict[f"diffusion_model.{module}.alpha"] = torch.tensor(9999220736.0)
            expected_deltas[module] = torch.kron(w1, w2)

        # module with a rank-decomposed right factor and a meaningful alpha
        rank, alpha = 2, 1.0
        module = "layers.1.attention.to_v"
        linear = named[module]
        (out_l, out_k) = factorization(linear.out_features, factor)
        (in_m, in_n) = factorization(linear.in_features, factor)
        w1 = torch.randn(out_l, in_m)
        w2_a = torch.randn(out_k, rank)
        w2_b = torch.randn(rank, in_n)
        state_dict[f"diffusion_model.{module}.lokr_w1"] = w1
        state_dict[f"diffusion_model.{module}.lokr_w2_a"] = w2_a
        state_dict[f"diffusion_model.{module}.lokr_w2_b"] = w2_b
        state_dict[f"diffusion_model.{module}.alpha"] = torch.tensor(alpha)
        expected_deltas[module] = (alpha / rank) * torch.kron(w1, w2_a @ w2_b)

        converted = ZImageLoraLoaderMixin.lora_state_dict(state_dict)
        self.assertTrue(all(k.startswith("transformer.") and ".lokr_" in k for k in converted))

        transformer.load_lora_adapter(converted, prefix="transformer", adapter_name="default")

        named = dict(transformer.named_modules())
        wrapped = {name for name, module in transformer.named_modules() if isinstance(module, LoKrLayer)}
        self.assertEqual(wrapped, set(expected_deltas))
        for module, expected in expected_deltas.items():
            delta = named[module].get_delta_weight("default")
            self.assertLess((delta - expected).abs().max().item(), 1e-5, module)
