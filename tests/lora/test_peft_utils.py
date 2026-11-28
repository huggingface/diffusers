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

import logging

import torch

from diffusers.utils import peft_utils
from diffusers.utils.peft_utils import _create_lora_config, get_peft_kwargs

from ..testing_utils import CaptureLogger, require_peft_backend


def _rank_dict(module_ranks):
    return {f"{module}.lora_B.weight": rank for module, rank in module_ranks.items()}


def _peft_state_dict(module_ranks):
    state_dict = {}
    for module in module_ranks:
        state_dict[f"{module}.lora_A.weight"] = None
        state_dict[f"{module}.lora_B.weight"] = None
    return state_dict


def _effective_scale(kwargs, module):
    alpha = kwargs["alpha_pattern"].get(module, kwargs["lora_alpha"])
    rank = kwargs["rank_pattern"].get(module, kwargs["r"])
    return alpha / rank


def test_mixed_ranks_without_alphas_apply_at_scale_one():
    # An adapter with per-module ranks and no alpha keys means `W_eff = W + lora_B @ lora_A`,
    # i.e. alpha == rank, so every module must come out at scale 1.0.
    module_ranks = {"blocks.0.adaln": 16, "blocks.0.to_q": 64, "blocks.0.to_v": 64}
    kwargs = get_peft_kwargs(_rank_dict(module_ranks), None, _peft_state_dict(module_ranks))
    for module in module_ranks:
        assert _effective_scale(kwargs, module) == 1.0


def test_mixed_ranks_without_alphas_are_order_independent():
    # The scale must not depend on which module happens to come first in the state dict.
    module_ranks = {"blocks.0.to_q": 64, "blocks.0.to_v": 64, "blocks.0.adaln": 16}
    reordered = dict(reversed(module_ranks.items()))
    for ranks in (module_ranks, reordered):
        kwargs = get_peft_kwargs(_rank_dict(ranks), None, _peft_state_dict(ranks))
        for module in ranks:
            assert _effective_scale(kwargs, module) == 1.0


def test_uniform_rank_without_alphas_unchanged():
    module_ranks = {"blocks.0.to_q": 32, "blocks.0.to_v": 32}
    kwargs = get_peft_kwargs(_rank_dict(module_ranks), None, _peft_state_dict(module_ranks))
    assert kwargs["r"] == kwargs["lora_alpha"] == 32
    assert kwargs["rank_pattern"] == {}
    assert kwargs["alpha_pattern"] == {}


def test_mixed_ranks_with_uniform_alpha_keep_declared_alpha():
    # A declared alpha must win over the alpha == rank convention: scale is alpha / rank per module.
    module_ranks = {"blocks.0.adaln": 16, "blocks.0.to_q": 64}
    network_alphas = {f"{module}.alpha": 32 for module in module_ranks}
    kwargs = get_peft_kwargs(_rank_dict(module_ranks), network_alphas, _peft_state_dict(module_ranks))
    assert kwargs["lora_alpha"] == 32
    assert kwargs["alpha_pattern"] == {}
    assert _effective_scale(kwargs, "blocks.0.adaln") == 2.0
    assert _effective_scale(kwargs, "blocks.0.to_q") == 0.5


def test_mixed_ranks_with_per_module_alphas_unchanged():
    module_ranks = {"blocks.0.adaln": 16, "blocks.0.to_q": 64}
    network_alphas = {"blocks.0.adaln.alpha": 16, "blocks.0.to_q.alpha": 64}
    kwargs = get_peft_kwargs(_rank_dict(module_ranks), network_alphas, _peft_state_dict(module_ranks))
    for module in module_ranks:
        assert _effective_scale(kwargs, module) == 1.0


def test_text_encoder_alpha_pattern_names_lora_modules():
    # Regression test for https://github.com/huggingface/diffusers/issues/14970:
    # text-encoder `alpha_pattern` keys must name the LoRA module itself (e.g. `k_proj`),
    # not its parent block, otherwise PEFT's end-of-name matching misses and the
    # default (most common) alpha is applied to every module.
    peft_state_dict = {
        "text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": None,
        "text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": None,
        "text_model.encoder.layers.1.self_attn.k_proj.lora_A.weight": None,
        "text_model.encoder.layers.1.self_attn.k_proj.lora_B.weight": None,
    }
    rank_dict = {k.replace(".lora_A.", ".lora_B."): 4 for k in peft_state_dict if ".lora_A." in k}
    network_alphas = {
        "text_model.encoder.layers.0.self_attn.to_q_lora.down.weight.alpha": 4.0,
        "text_model.encoder.layers.1.self_attn.to_k_lora.down.weight.alpha": 8.0,
    }
    kwargs = get_peft_kwargs(rank_dict, network_alphas, peft_state_dict, is_unet=False)
    assert kwargs["lora_alpha"] == 4.0
    assert kwargs["alpha_pattern"] == {"encoder.layers.1.self_attn.k_proj": 8.0}
    assert _effective_scale(kwargs, "encoder.layers.1.self_attn.k_proj") == 2.0
    assert _effective_scale(kwargs, "encoder.layers.0.self_attn.q_proj") == 1.0


def test_text_encoder_alpha_pattern_handles_mlp_and_text_projection():
    peft_state_dict = {"m.lora_A.weight": None, "m.lora_B.weight": None}
    network_alphas = {
        "text_model.encoder.layers.0.mlp.fc1.lora_linear_layer.down.weight.alpha": 8.0,
        "text_model.text_projection.alpha": 8.0,
        "text_model.encoder.layers.0.mlp.fc2.lora_linear_layer.down.weight.alpha": 4.0,
    }
    kwargs = get_peft_kwargs({"m.lora_B.weight": 4}, network_alphas, peft_state_dict, is_unet=False)
    assert kwargs["lora_alpha"] == 8.0
    assert kwargs["alpha_pattern"] == {"encoder.layers.0.mlp.fc2": 4.0}


class _TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4)
        self.to_v = torch.nn.Linear(4, 4)


def _lora_metadata(**extra):
    return {"r": 4, "lora_alpha": 8, "target_modules": ["to_q", "to_v"], **extra}


@require_peft_backend
def test_lora_metadata_megatron_fields_are_ignored():
    # A checkpoint's adapter metadata must not be able to switch peft to Megatron layers, which imports the
    # module named by `megatron_core`.
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import Linear as LoraLinear

    metadata = _lora_metadata(megatron_core="megatron.core", megatron_config={"params_dtype": "float32"})
    peft_utils.logger.setLevel(logging.WARNING)
    with CaptureLogger(peft_utils.logger) as cap_logger:
        config = _create_lora_config(state_dict={}, network_alphas=None, metadata=metadata, rank_pattern_dict={})

    assert config.megatron_config is None
    assert config.megatron_core == LoraConfig().megatron_core
    assert (config.r, config.lora_alpha, set(config.target_modules)) == (4, 8, {"to_q", "to_v"})
    assert "megatron_config" in metadata
    assert "Ignoring `megatron_config`" in cap_logger.out

    model = inject_adapter_in_model(config, _TinyAttention(), adapter_name="default")
    assert isinstance(model.to_q, LoraLinear)


@require_peft_backend
def test_lora_metadata_with_default_megatron_fields_does_not_warn():
    # `LoraConfig.to_dict()` always includes the Megatron fields, so a regularly saved adapter carries them with
    # their default values.
    from peft import LoraConfig

    metadata = _lora_metadata(megatron_core=LoraConfig().megatron_core, megatron_config=None)
    peft_utils.logger.setLevel(logging.WARNING)
    with CaptureLogger(peft_utils.logger) as cap_logger:
        _create_lora_config(state_dict={}, network_alphas=None, metadata=metadata, rank_pattern_dict={})

    assert "megatron" not in cap_logger.out
