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

import os
import tempfile

import pytest
import safetensors.torch
import torch

from diffusers.modular_pipelines import MiniMaxH3Blocks, MiniMaxH3ModularPipeline
from diffusers.utils import is_peft_available, logging

from ..testing_utils import CaptureLogger, require_peft_backend


if is_peft_available():
    from peft.utils import get_peft_model_state_dict


@require_peft_backend
class TestMiniMaxH3LoraLayers:
    """
    The MiniMax-H3 LoRA surface that is specific to this model and its two checkpoint partitions: loading the layouts
    that circulate, the alpha handling that gets each of them to its trained scale, and the routing between the
    `transformer` and `transformer_ref` partitions. Generic LoRA behavior is not tested here.
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

    def get_dummy_flattened_lora_state_dict(self, rank=4, alpha=4.0):
        r"""
        musubi-tuner's layout: one flat `lora_unet_` module name per key with every `.` collapsed to `_`, kohya
        `lora_down`/`lora_up` tensors and an explicit scalar `.alpha`. The module names carry underscores of their own,
        so every name the un-flattening has to disambiguate is present here, not just the four the published LoRA in
        this layout happens to train.
        """
        config = self.get_pipeline().transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim
        video_patch_dim = config.in_channels * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]

        state_dict = {}
        for module, in_features, out_features in [
            ("blocks_0_attn_qkv_proj", hidden, 3 * inner),
            ("blocks_0_attn_out_proj", inner, hidden),
            ("blocks_0_mlp_fc1", hidden, 2 * config.ffn_dim),
            ("blocks_0_mlp_fc2", config.ffn_dim, hidden),
            ("blocks_0_adaln_proj_linear", config.time_embed_dim, 6 * 3 * hidden),
            ("token_refiner_blocks_0_attn_qkv_proj", hidden, 3 * inner),
            ("token_refiner_blocks_0_mlp_fc2", config.ffn_dim, hidden),
            ("video_patch_proj", video_patch_dim, hidden),
            ("audio_patch_proj", config.audio_in_channels, hidden),
            ("condition_proj", config.text_dim, hidden),
            ("time_embedder_proj_in", config.freq_dim, config.time_embed_hidden_dim),
            ("time_embedder_proj_out", config.time_embed_hidden_dim, config.time_embed_dim),
            ("final_layer_adaln_proj_linear", config.time_embed_dim, 2 * hidden),
            ("final_layer_video_out", hidden, video_patch_dim),
            ("final_layer_audio_out", hidden, config.audio_in_channels),
        ]:
            state_dict[f"lora_unet_{module}.lora_down.weight"] = torch.randn(rank, in_features)
            state_dict[f"lora_unet_{module}.lora_up.weight"] = torch.randn(out_features, rank)
            state_dict[f"lora_unet_{module}.alpha"] = torch.tensor(alpha)
        return state_dict

    def test_load_lora_weights_flattened_layout(self):
        pipe = self.get_pipeline()

        pipe.load_lora_weights(self.get_dummy_flattened_lora_state_dict(), adapter_name="dummy")

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 19
        assert {module.scaling["dummy"] for module in injected} == {1.0}

    def test_load_lora_weights_folds_explicit_alphas_once(self):
        r"""
        Files that ship explicit `.alpha` scalars must apply at `alpha / rank` per module. That ratio is folded into the
        weights during conversion, not carried to peft as a network alpha, so the loaded adapter shows `alpha == rank`
        and a peft scale of 1.0 while the update itself carries the ratio. Both halves are asserted: dropping the fold
        under-scales the update, and passing the alpha on as well would square the ratio. Two modules of the same rank
        carry different alphas here, which no rank-keyed `alpha_pattern` could express.
        """
        pipe = self.get_pipeline()
        config = pipe.transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim

        state_dict, expected = {}, {}
        for source, target, in_features, out_features, rank, alpha in [
            ("blocks.0.attn.out_proj", "transformer_blocks.0.attn.to_out.0", inner, hidden, 8, 2.0),
            ("blocks.0.mlp.fc2", "transformer_blocks.0.ff.net.2", config.ffn_dim, hidden, 8, 4.0),
            (
                "blocks.0.adaln_proj.linear",
                "transformer_blocks.0.adaln_proj.linear",
                config.time_embed_dim,
                6 * 3 * hidden,
                4,
                2.0,
            ),
            ("final_layer.adaln_proj.linear", "norm_out.linear", config.time_embed_dim, 2 * hidden, 4, 1.0),
        ]:
            down, up = torch.randn(rank, in_features), torch.randn(out_features, rank)
            state_dict[f"diffusion_model.{source}.lora_A.weight"] = down
            state_dict[f"diffusion_model.{source}.lora_B.weight"] = up
            state_dict[f"diffusion_model.{source}.alpha"] = torch.tensor(alpha)
            expected[target] = (alpha / rank) * (up @ down)

        pipe.load_lora_weights(state_dict, adapter_name="dummy")

        for target, delta in expected.items():
            module = pipe.transformer.get_submodule(target)
            assert module.scaling["dummy"] == 1.0
            assert module.lora_alpha["dummy"] == module.r["dummy"]
            applied = module.scaling["dummy"] * (module.lora_B["dummy"].weight @ module.lora_A["dummy"].weight)
            assert torch.allclose(applied, delta, atol=1e-5), target

    def get_dummy_unprefixed_diffusers_lora_state_dict(self, rank=4):
        r"""
        One producer publishes its own converter's output: diffusers module names with split q/k/v, peft's
        `.default.` adapter-name infix left in every key, and no component prefix.
        """
        config = self.get_pipeline().transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim

        state_dict = {}
        for module, in_features, out_features in [
            ("transformer_blocks.0.attn.to_q", hidden, inner),
            ("transformer_blocks.0.attn.to_k", hidden, inner),
            ("transformer_blocks.0.attn.to_v", hidden, inner),
            ("transformer_blocks.0.attn.to_out.0", inner, hidden),
            ("transformer_blocks.0.ff.net.0.proj", hidden, 2 * config.ffn_dim),
            ("transformer_blocks.0.ff.net.2", config.ffn_dim, hidden),
            ("token_refiner.refiner_blocks.0.attn.to_q", hidden, inner),
            ("token_refiner.refiner_blocks.0.ff.net.2", config.ffn_dim, hidden),
        ]:
            state_dict[f"{module}.lora_A.default.weight"] = torch.randn(rank, in_features)
            state_dict[f"{module}.lora_B.default.weight"] = torch.randn(out_features, rank)
        return state_dict

    def test_load_lora_weights_unprefixed_diffusers_format(self):
        pipe = self.get_pipeline()

        pipe.load_lora_weights(self.get_dummy_unprefixed_diffusers_lora_state_dict(), adapter_name="dummy")

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 8
        assert {module.scaling["dummy"] for module in injected} == {1.0}

    def test_load_lora_weights_warns_when_nothing_is_targeted(self):
        r"""
        An unrecognized layout that keeps the substring `lora` in every key passes the format check and then filters to
        nothing in both partitions. Neither partition branch would run, so without this warning the load is a silent
        no-op. The message follows `PeftAdapterMixin.load_lora_adapter`'s wording, which is what every single-denoiser
        model emits in the same situation.
        """
        pipe = self.get_pipeline()
        state_dict = {
            "some_other_model.layers.0.lora_A.weight": torch.randn(4, 24),
            "some_other_model.layers.0.lora_B.weight": torch.randn(24, 4),
        }

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.WARNING)
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(state_dict, adapter_name="dummy")

        assert cap_logger.out.startswith("No LoRA keys associated to MiniMaxH3ModularPipeline")
        assert "some_other_model.layers.0.lora_A.weight" in cap_logger.out
        # Nothing was loaded into either partition.
        for component in [pipe.transformer, pipe.transformer_ref]:
            assert "dummy" not in getattr(component, "peft_config", {})
            assert not [module for module in component.modules() if "dummy" in getattr(module, "scaling", {})]

    def save_with_file_metadata(self, state_dict, tmpdir, alpha="8"):
        r"""
        One producer records the alpha it trained with in the safetensors `__metadata__` instead of in per-module
        scalars, so the value exists only on disk — a state dict handed over in memory cannot carry it.
        """
        weight_name = "pytorch_lora_weights.safetensors"
        safetensors.torch.save_file(
            state_dict, os.path.join(tmpdir, weight_name), metadata={"floating_dtype": "bfloat16", "alpha": alpha}
        )
        return weight_name

    def test_load_lora_weights_honors_the_metadata_alpha(self):
        r"""
        The 8-step turbo LoRA's header in miniature: diffusers module names with peft's `.default.` infix, one uniform
        rank, no `.alpha` scalars anywhere, and `alpha` "8" in the file's own `__metadata__`. Rank 128 against alpha 8
        is a trained scale of 0.0625; synthesizing `alpha == rank` instead applies the adapter 16x too strongly.
        """
        state_dict = self.get_dummy_unprefixed_diffusers_lora_state_dict(rank=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_name = self.save_with_file_metadata(state_dict, tmpdir)

            pipe = self.get_pipeline()
            pipe.load_lora_weights(tmpdir, weight_name=weight_name, adapter_name="dummy")

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 8
        assert {module.scaling["dummy"] for module in injected} == {0.0625}

    def test_load_lora_weights_prefers_alpha_tensors_over_the_metadata_alpha(self):
        r"""
        A file carrying both must not apply an alpha twice. The converter folds each module's `.alpha` into the weights,
        so the `__metadata__` entry is ignored and the adapter loads at `alpha == rank`. The mixed rank is what makes
        this discriminating: honoring `alpha` "8" here would scale the rank-2 modules by 4.0.
        """
        state_dict = self.get_dummy_lora_state_dict(rank=8, adaln_rank=2)
        for key in [k for k in state_dict if k.endswith(".lora_A.weight")]:
            state_dict[f"{key.removesuffix('.lora_A.weight')}.alpha"] = torch.tensor(2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_name = self.save_with_file_metadata(state_dict, tmpdir)

            pipe = self.get_pipeline()
            pipe.load_lora_weights(tmpdir, weight_name=weight_name, adapter_name="dummy")

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        assert {module.scaling["dummy"] for module in injected} == {1.0}
        assert pipe.transformer.transformer_blocks[0].attn.to_q.r["dummy"] == 8
        assert pipe.transformer.norm_out.linear.r["dummy"] == 2

    def test_load_lora_weights_warns_on_a_non_numeric_metadata_alpha(self):
        r"""
        `alpha` is a generic word for a header entry, so a file can carry one that means something else entirely. Such a
        value is warned about and ignored rather than refused — the file still loads, at the `alpha == rank` the
        alpha-less synthesis pins.
        """
        state_dict = self.get_dummy_unprefixed_diffusers_lora_state_dict(rank=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_name = self.save_with_file_metadata(state_dict, tmpdir, alpha="high")

            logger = logging.get_logger("diffusers.loaders.lora_pipeline")
            logger.setLevel(logging.WARNING)
            pipe = self.get_pipeline()
            with CaptureLogger(logger) as cap_logger:
                pipe.load_lora_weights(tmpdir, weight_name=weight_name, adapter_name="dummy")

        assert "'high'" in cap_logger.out
        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 8
        assert {module.scaling["dummy"] for module in injected} == {1.0}

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
