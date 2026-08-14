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
        A file with one uniform rank, no `.alpha` scalars and `alpha` "8" in its own `__metadata__`: rank 128
        against alpha 8 is a trained scale of 0.0625; synthesizing `alpha == rank` would apply the adapter 16x too
        strongly.
        """
        state_dict = self.get_dummy_diffusers_lora_state_dict(prefix="transformer", rank=128, adaln_rank=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_name = self.save_with_file_metadata(state_dict, tmpdir)

            pipe = self.get_pipeline()
            pipe.load_lora_weights(tmpdir, weight_name=weight_name, adapter_name="dummy")

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 6
        assert {module.scaling["dummy"] for module in injected} == {0.0625}

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

    def test_load_lora_weights_into_transformer_ref(self):
        pipe = self.get_pipeline()

        pipe.load_lora_weights(
            self.get_dummy_diffusers_lora_state_dict(), adapter_name="dummy", load_into_transformer_ref=True
        )

        assert "dummy" in pipe.transformer_ref.peft_config
        assert "dummy" not in getattr(pipe.transformer, "peft_config", {})

    def test_save_load_lora_weights_round_trip(self):
        r"""
        `save_lora_weights` is the only mechanism that records which partition a LoRA belongs to, so the round trip
        has to preserve it.
        """
        pipe = self.get_pipeline()
        pipe.load_lora_weights(
            self.get_dummy_diffusers_lora_state_dict(), adapter_name="dummy", load_into_transformer_ref=True
        )
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

        state_dict = self.get_dummy_diffusers_lora_state_dict()
        pipe.load_lora_weights(state_dict, adapter_name="dummy")

        assert "dummy" in pipe.transformer_ref.peft_config

    def test_load_lora_weights_raises_without_the_requested_partition(self):
        pipe = self.pipeline_blocks_class().get_workflow("t2va").init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        assert getattr(pipe, "transformer_ref", None) is None

        state_dict = self.get_dummy_diffusers_lora_state_dict()
        with pytest.raises(ValueError, match="load_into_transformer_ref"):
            pipe.load_lora_weights(state_dict, load_into_transformer_ref=True)
