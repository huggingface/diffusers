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

import torch

from diffusers.hooks.taylorseer_cache import TaylorSeerState


def _step(state: TaylorSeerState, features: torch.Tensor) -> None:
    state.current_step += 1
    state.update((features,))


def test_update_restarts_when_feature_shape_changes():
    # Qwen-Image 2.1 with the KV cache returns prefix + target tokens on the prefill step and target tokens only
    # afterwards, so the hooked module sees two different sequence lengths within one denoising loop.
    state = TaylorSeerState(taylor_factors_dtype=torch.float32, max_order=1)

    _step(state, torch.ones(1, 10, 4))
    _step(state, torch.ones(1, 6, 4))

    assert set(state.taylor_factors[0]) == {0}
    assert state.taylor_factors[0][0].shape == (1, 6, 4)

    _step(state, torch.full((1, 6, 4), 3.0))

    assert set(state.taylor_factors[0]) == {0, 1}
    torch.testing.assert_close(state.taylor_factors[0][1], torch.full((1, 6, 4), 2.0))

    state.current_step += 1
    (prediction,) = state.predict()
    torch.testing.assert_close(prediction, torch.full((1, 6, 4), 5.0))


def test_update_keeps_finite_difference_when_shape_is_stable():
    state = TaylorSeerState(taylor_factors_dtype=torch.float32, max_order=1)

    _step(state, torch.zeros(1, 6, 4))
    _step(state, torch.full((1, 6, 4), 2.0))

    torch.testing.assert_close(state.taylor_factors[0][1], torch.full((1, 6, 4), 2.0))
