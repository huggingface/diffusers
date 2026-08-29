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

from diffusers.utils.peft_utils import get_peft_kwargs


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
