# Copyright 2025 HuggingFace Inc.
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

from diffusers.hooks.taylorseer_cache import (
    TaylorSeerCacheConfig,
    TaylorSeerCacheHook,
    TaylorSeerState,
    _apply_taylorseer_cache_hook,
    apply_taylorseer_cache,
)
from diffusers.hooks.hooks import StateManager
from diffusers.models import ModelMixin


class DummyAttnBlock(torch.nn.Module):
    """A simple attention-like block whose output is 2x the input."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states * 2.0


class DummyTransformer(ModelMixin):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyAttnBlock()])

    def forward(self, hidden_states):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _make_hook(
    cache_interval: int = 5,
    disable_cache_before_step: int = 3,
    disable_cache_after_step: int | None = None,
) -> TaylorSeerCacheHook:
    """Construct a TaylorSeerCacheHook with a fresh StateManager."""
    state_manager = StateManager(
        TaylorSeerState,
        init_kwargs={
            "taylor_factors_dtype": torch.float32,
            "max_order": 1,
            "is_inactive": False,
        },
    )
    return TaylorSeerCacheHook(
        cache_interval=cache_interval,
        disable_cache_before_step=disable_cache_before_step,
        taylor_factors_dtype=torch.float32,
        state_manager=state_manager,
        disable_cache_after_step=disable_cache_after_step,
    )


class TaylorSeerCacheTests(unittest.TestCase):
    def test_compute_schedule_first_post_warmup_step_triggers_compute(self):
        """
        The first step at or after disable_cache_before_step must always trigger
        a full forward pass (should_compute=True), not a cached prediction.

        With disable_cache_before_step=3 and cache_interval=5 the expected
        compute steps are: 0, 1, 2 (warmup), 3, 8, 13, ...

        The off-by-one bug `(step - disable - 1) % interval` shifts this to
        4, 9, 14, ... causing step 3 to wrongly return should_compute=False.
        """
        hook = _make_hook(cache_interval=5, disable_cache_before_step=3)

        expected = {
            0: True,   # warmup
            1: True,   # warmup
            2: True,   # warmup
            3: True,   # first post-warmup step — must compute, not predict
            4: False,  # cache reuse
            5: False,
            6: False,
            7: False,
            8: True,   # next compute refresh at disable + cache_interval = 3 + 5
            9: False,
        }

        for step, should_compute_expected in expected.items():
            should_compute, _ = hook._measure_should_compute()
            self.assertEqual(
                should_compute,
                should_compute_expected,
                f"Step {step}: expected should_compute={should_compute_expected}, got {should_compute}",
            )

    def test_compute_schedule_disable_cache_after_step(self):
        """
        Steps at or beyond disable_cache_after_step must always compute
        regardless of cache_interval position.
        """
        hook = _make_hook(
            cache_interval=5,
            disable_cache_before_step=2,
            disable_cache_after_step=6,
        )

        # Steps 0-1 warmup, step 2 first refresh, steps 3-4 cache, step 5 cache,
        # step 6+ cooldown (always compute).
        expected = {
            0: True,   # warmup
            1: True,   # warmup
            2: True,   # first post-warmup compute (disable_cache_before_step=2)
            3: False,
            4: False,
            5: False,
            6: True,   # cooldown — always compute
            7: True,   # cooldown
        }

        for step, should_compute_expected in expected.items():
            should_compute, _ = hook._measure_should_compute()
            self.assertEqual(
                should_compute,
                should_compute_expected,
                f"Step {step}: expected should_compute={should_compute_expected}, got {should_compute}",
            )


if __name__ == "__main__":
    unittest.main()
