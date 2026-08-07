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

import tempfile

import pytest
import torch

from diffusers.modular_pipelines import MiniMaxH3Blocks, MiniMaxH3ModularPipeline
from diffusers.utils import is_peft_available

from ..testing_utils import require_peft_backend


if is_peft_available():
    from peft.utils import get_peft_model_state_dict


@require_peft_backend
class TestMiniMaxH3LoraLayers:
    """
    The MiniMax-H3 LoRA surface that is specific to this model and its two checkpoint partitions: the conversion of
    the two circulating non-diffusers formats (fused projections, original module names, no alpha keys), the
    `alpha == rank` metadata synthesis for alpha-less files, and the routing between the `transformer` and
    `transformer_ref` partitions. Generic LoRA behavior is not tested here.
    """

    pipeline_class = MiniMaxH3ModularPipeline
    pipeline_blocks_class = MiniMaxH3Blocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-minimax-h3-modular-pipe"

    def get_pipeline(self):
        pipeline = self.pipeline_blocks_class().init_pipeline(self.pretrained_model_name_or_path)
        pipeline.load_components(dtype=torch.float32)
        pipeline.set_progress_bar_config(disable=None)
        return pipeline

    def get_dummy_lora_state_dict(self, prefix="diffusion_model.", rank=4, adaln_rank=None):
        r"""
        A LoRA in the layout both real-world producers emit: the *original* checkpoint's module names, fused
        `attn.qkv_proj` and `mlp.fc1`, and no `.alpha`. ai-toolkit prefixes them with `diffusion_model.`; the one
        public H3 LoRA carries no prefix at all, which `prefix=""` reproduces. `adaln_rank` makes the file mixed-rank,
        as that LoRA is.
        """
        transformer = self.get_pipeline().transformer
        config = transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim
        adaln_rank = adaln_rank or rank

        state_dict = {}
        for source, in_features, out_features, module_rank in [
            ("blocks.0.attn.qkv_proj", hidden, 3 * inner, rank),
            ("blocks.0.attn.out_proj", inner, hidden, rank),
            ("blocks.0.mlp.fc1", hidden, 2 * config.ffn_dim, rank),
            ("blocks.0.mlp.fc2", config.ffn_dim, hidden, rank),
            ("blocks.0.adaln_proj.linear", config.time_embed_dim, 6 * 3 * hidden, adaln_rank),
            ("token_refiner.blocks.0.attn.qkv_proj", hidden, 3 * inner, rank),
            ("final_layer.adaln_proj.linear", config.time_embed_dim, 2 * hidden, adaln_rank),
        ]:
            state_dict[f"{prefix}{source}.lora_A.weight"] = torch.randn(module_rank, in_features)
            state_dict[f"{prefix}{source}.lora_B.weight"] = torch.randn(out_features, module_rank)
        return state_dict

    def test_lora_state_dict_conversion(self):
        r"""The original module names map onto the diffusers ones, fused projections split, `mlp.fc1` halves swap."""
        state_dict = self.get_dummy_lora_state_dict()
        rank = state_dict["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].shape[0]
        fused_up = state_dict["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
        qkv_up = state_dict["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]

        converted = self.pipeline_class.lora_state_dict(state_dict)

        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in converted
        assert "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight" in converted
        assert "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight" in converted
        assert "transformer.transformer_blocks.0.ff.net.2.lora_B.weight" in converted
        assert "transformer.transformer_blocks.0.adaln_proj.linear.lora_A.weight" in converted
        assert "transformer.token_refiner.refiner_blocks.0.attn.to_v.lora_B.weight" in converted
        assert "transformer.norm_out.linear.lora_B.weight" in converted
        assert not any("qkv_proj" in key or "fc1" in key or "final_layer" in key for key in converted)

        # The fused QKV splits into three row blocks that share `lora_A`.
        inner = qkv_up.shape[0] // 3
        for index, projection in enumerate(["to_q", "to_k", "to_v"]):
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            assert torch.equal(converted[f"{prefix}.lora_B.weight"], qkv_up[index * inner : (index + 1) * inner])
            assert torch.equal(
                converted[f"{prefix}.lora_A.weight"],
                converted["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"],
            )

        # `mlp.fc1` is `[gate; value]` and `SwiGLU.proj` is `[value; gate]`, so `lora_B`'s halves swap and `lora_A`
        # is untouched. A key-name-only assertion would pass with the swap missing, which is a silent quality bug.
        ffn_dim = fused_up.shape[0] // 2
        swapped = converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
        assert torch.equal(swapped, torch.cat([fused_up[ffn_dim:], fused_up[:ffn_dim]]))
        assert torch.equal(
            converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight"],
            state_dict["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"],
        )
        assert all(value.shape[1] == rank or value.shape[0] == rank for value in converted.values())

    def test_lora_state_dict_conversion_without_a_prefix(self):
        r"""The one public H3 LoRA has no prefix at all, so the module names are what identifies the format."""
        converted = self.pipeline_class.lora_state_dict(self.get_dummy_lora_state_dict(prefix=""))

        assert "transformer.transformer_blocks.0.attn.to_k.lora_B.weight" in converted
        assert all(key.startswith("transformer.") for key in converted)

    def test_lora_state_dict_conversion_raises_on_an_unknown_module(self):
        state_dict = self.get_dummy_lora_state_dict()
        state_dict["diffusion_model.blocks.0.not_a_module.lora_A.weight"] = torch.randn(4, 8)
        state_dict["diffusion_model.blocks.0.not_a_module.lora_B.weight"] = torch.randn(8, 4)

        with pytest.raises(ValueError, match="not_a_module"):
            self.pipeline_class.lora_state_dict(state_dict)

    def test_lora_state_dict_synthesizes_unit_scale_metadata(self):
        r"""
        A non-diffusers H3 LoRA has no alpha information and applies as `W + lora_B @ lora_A`. `get_peft_kwargs` reads
        `lora_alpha` off the first rank it sees and never re-derives it, so a mixed-rank file — which the public turbo
        LoRA is — would have its majority-rank modules scaled by `alpha / r`. The converted metadata pins
        `alpha == rank` for every module instead.
        """
        state_dict = self.get_dummy_lora_state_dict(rank=8, adaln_rank=2)

        _, metadata = self.pipeline_class.lora_state_dict(state_dict, return_lora_metadata=True)

        assert metadata["transformer.r"] == 8
        assert metadata["transformer.lora_alpha"] == 8
        assert metadata["transformer.alpha_pattern"] == metadata["transformer.rank_pattern"]
        assert set(metadata["transformer.rank_pattern"].values()) == {2}
        assert "^norm_out.linear" in metadata["transformer.rank_pattern"]

    def get_dummy_diffusers_lora_state_dict(self, prefix="transformer", rank=8, adaln_rank=2):
        r"""
        The same adapter already converted to diffusers keys and republished — which is how the public turbo LoRA also
        circulates. Mixed-rank, still alpha-less, so it needs the same treatment as the original layout even though no
        conversion runs.
        """
        transformer = self.get_pipeline().transformer
        config = transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim

        state_dict = {}
        for module, in_features, out_features, module_rank in [
            ("transformer_blocks.0.attn.to_q", hidden, inner, rank),
            ("transformer_blocks.0.attn.to_out.0", inner, hidden, rank),
            ("transformer_blocks.0.ff.net.0.proj", hidden, 2 * config.ffn_dim, rank),
            ("transformer_blocks.0.ff.net.2", config.ffn_dim, hidden, rank),
            ("transformer_blocks.0.adaln_proj.linear", config.time_embed_dim, 6 * 3 * hidden, adaln_rank),
            ("norm_out.linear", config.time_embed_dim, 2 * hidden, adaln_rank),
        ]:
            state_dict[f"{prefix}.{module}.lora_A.weight"] = torch.randn(module_rank, in_features)
            state_dict[f"{prefix}.{module}.lora_B.weight"] = torch.randn(out_features, module_rank)
        return state_dict

    @pytest.mark.parametrize("prefix", ["transformer", "transformer_ref"], ids=["transformer", "transformer_ref"])
    def test_load_lora_weights_diffusers_format_mixed_rank(self, prefix):
        r"""
        A mixed-rank, alpha-less adapter already in diffusers keys bypasses the converter, so the alpha handling cannot
        live there: without it `get_peft_kwargs` takes `lora_alpha` from whichever rank it sees first and one of the
        two rank groups is applied at `alpha / r`.
        """
        pipe = self.get_pipeline()

        pipe.load_lora_weights(self.get_dummy_diffusers_lora_state_dict(prefix=prefix), adapter_name="dummy")

        component = getattr(pipe, prefix)
        injected = [module for module in component.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 6
        assert {module.scaling["dummy"] for module in injected} == {1.0}
        for module in injected:
            assert module.lora_alpha["dummy"] == module.r["dummy"]
        assert component.transformer_blocks[0].attn.to_q.r["dummy"] == 8
        assert component.transformer_blocks[0].adaln_proj.linear.r["dummy"] == 2
        assert component.norm_out.linear.r["dummy"] == 2

    def test_lora_state_dict_respects_existing_metadata(self):
        r"""A file that carries diffusers' own `lora_adapter_metadata` must not have it overwritten."""
        pipe = self.get_pipeline()
        pipe.load_lora_weights(self.get_dummy_diffusers_lora_state_dict(), adapter_name="dummy")
        layers = get_peft_model_state_dict(pipe.transformer, adapter_name="dummy")
        saved_metadata = {"r": 8, "lora_alpha": 8, "rank_pattern": {}, "alpha_pattern": {}, "target_modules": ["x"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline_class.save_lora_weights(
                tmpdir, transformer_lora_layers=layers, transformer_lora_adapter_metadata=saved_metadata
            )
            _, metadata = self.pipeline_class.lora_state_dict(tmpdir, return_lora_metadata=True)

        assert metadata["transformer.target_modules"] == ["x"]

    @pytest.mark.parametrize("prefix", ["diffusion_model.", ""], ids=["ai_toolkit", "unprefixed"])
    def test_load_lora_weights(self, prefix):
        r"""
        A mixed-rank, alpha-less file — the public turbo LoRA's shape — has to reach every module at its own rank and
        at an effective scale of exactly 1.0.
        """
        pipe = self.get_pipeline()

        pipe.load_lora_weights(
            self.get_dummy_lora_state_dict(prefix=prefix, rank=8, adaln_rank=2), adapter_name="dummy"
        )

        assert "dummy" in pipe.transformer.peft_config
        # Both partitions are loaded here and the file does not say which one it targets, so only `transformer` gets it.
        assert "dummy" not in getattr(pipe.transformer_ref, "peft_config", {})

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        # 3 split qkv + to_out.0 + the two ff Linears + adaln, the refiner's 3 split qkv, and norm_out
        assert len(injected) == 11
        assert {module.scaling["dummy"] for module in injected} == {1.0}
        for module in injected:
            assert module.lora_A["dummy"].weight.shape[0] == module.r["dummy"]
            assert module.lora_alpha["dummy"] == module.r["dummy"]
        assert pipe.transformer.transformer_blocks[0].attn.to_q.r["dummy"] == 8
        assert pipe.transformer.transformer_blocks[0].adaln_proj.linear.r["dummy"] == 2
        assert pipe.transformer.norm_out.linear.r["dummy"] == 2

    def test_load_lora_weights_into_transformer_ref(self):
        pipe = self.get_pipeline()

        pipe.load_lora_weights(self.get_dummy_lora_state_dict(), adapter_name="dummy", load_into_transformer_ref=True)

        assert "dummy" in pipe.transformer_ref.peft_config
        assert "dummy" not in getattr(pipe.transformer, "peft_config", {})

    def test_save_load_lora_weights_round_trip(self):
        r"""
        `save_lora_weights` is the only mechanism that records which partition a LoRA belongs to, so the round trip
        has to preserve it.
        """
        pipe = self.get_pipeline()
        pipe.load_lora_weights(self.get_dummy_lora_state_dict(), adapter_name="dummy", load_into_transformer_ref=True)
        layers = get_peft_model_state_dict(pipe.transformer_ref, adapter_name="dummy")

        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline_class.save_lora_weights(tmpdir, transformer_ref_lora_layers=layers)
            reloaded = self.pipeline_class.lora_state_dict(tmpdir)

        assert reloaded
        assert all(key.startswith("transformer_ref.") for key in reloaded)

        fresh = self.get_pipeline()
        fresh.load_lora_weights(reloaded, adapter_name="dummy")
        assert "dummy" in fresh.transformer_ref.peft_config
        assert "dummy" not in getattr(fresh.transformer, "peft_config", {})

    def test_load_lora_weights_routes_to_the_only_partition(self):
        r"""
        `workflow="ref2va"` loads `transformer_ref` and no `transformer`, and nothing in a published H3 LoRA says which
        partition it targets, so the one partition that is present is the unambiguous destination.
        """
        pipe = self.pipeline_blocks_class().get_workflow("ref2va").init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        assert getattr(pipe, "transformer", None) is None

        state_dict = self.get_dummy_lora_state_dict()
        pipe.load_lora_weights(state_dict, adapter_name="dummy")

        assert "dummy" in pipe.transformer_ref.peft_config

    def test_load_lora_weights_raises_without_the_requested_partition(self):
        pipe = self.pipeline_blocks_class().get_workflow("t2va").init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        assert getattr(pipe, "transformer_ref", None) is None

        state_dict = self.get_dummy_lora_state_dict()
        with pytest.raises(ValueError, match="load_into_transformer_ref"):
            pipe.load_lora_weights(state_dict, load_into_transformer_ref=True)
