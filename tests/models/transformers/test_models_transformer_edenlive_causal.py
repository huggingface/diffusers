# Copyright 2026 The VAST-AI Team. All rights reserved.
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

import pytest
import torch

from diffusers.models.transformers import transformer_edenlive

from ...testing_utils import require_torch_multi_gpu
from .test_models_transformer_edenlive import EdenLiveTransformerTesterConfig


class TestEdenLiveCausalCacheOrder:
    @pytest.mark.parametrize("use_rotary_emb", [False, True])
    @torch.no_grad()
    def test_without_cache_matches_attention_reference(self, use_rotary_emb):
        torch.manual_seed(0)
        attn = transformer_edenlive.EdenLiveAttention(
            dim=8, heads=2, dim_head=4, processor=transformer_edenlive.EdenLiveCausalAttnProcessor()
        ).eval()
        hidden_states = torch.randn(1, 6, 8)
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(-1, (2, 4))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(-1, (2, 4))
        value = attn.to_v(hidden_states).unflatten(-1, (2, 4))
        rotary_emb = None
        if use_rotary_emb:
            angles = torch.randn(1, 6, 1, 2)
            rotations = torch.polar(torch.ones_like(angles), angles)
            query = torch.view_as_real(torch.view_as_complex(query.unflatten(-1, (2, 2))) * rotations).flatten(-2)
            key = torch.view_as_real(torch.view_as_complex(key.unflatten(-1, (2, 2))) * rotations).flatten(-2)
            rotary_emb = (angles.cos().repeat_interleave(2, dim=-1), angles.sin().repeat_interleave(2, dim=-1))
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            )
            .transpose(1, 2)
            .flatten(2)
        )
        expected = attn.to_out[1](attn.to_out[0](expected))
        actual = attn(hidden_states, rotary_emb=rotary_emb)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_local_chunk_order_survives_eviction_and_short_tail(self):
        torch.manual_seed(0)
        attn = transformer_edenlive.EdenLiveAttention(
            dim=2,
            heads=1,
            dim_head=2,
            processor=transformer_edenlive.EdenLiveCausalAttnProcessor(),
        ).eval()
        cache = transformer_edenlive._EdenLiveLayerCache(torch.zeros(1, 6, 1, 2), torch.zeros(1, 6, 1, 2))

        def hidden_states(ids):
            values = torch.tensor(ids, dtype=torch.float32)
            return torch.stack((values / 16, (values.remainder(3) - 1) / 2), dim=-1).unsqueeze(0)

        def call(ids, expected_ids, commit):
            hidden = hidden_states(ids)
            visible = hidden_states(expected_ids)
            rope = (torch.ones(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
            stored_key, stored_value = cache.key.clone(), cache.value.clone()
            stored_count = cache.num_tokens
            output = attn(
                hidden,
                rotary_emb=rope,
                key_rotary_emb=rope,
                layer_cache=cache,
                cache_update=commit,
                cache_window_tokens=6,
                cache_sink_tokens=2,
            )
            query = attn.norm_q(attn.to_q(hidden)).unsqueeze(1)
            key = attn.norm_k(attn.to_k(visible)).unsqueeze(1)
            value = attn.to_v(visible).unsqueeze(1)
            expected = torch.nn.functional.scaled_dot_product_attention(query, key, value).squeeze(1)
            expected = attn.to_out[1](attn.to_out[0](expected))
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
            if commit:
                assert cache.num_tokens == len(expected_ids)
                torch.testing.assert_close(cache.key[:, : cache.num_tokens], key.transpose(1, 2), rtol=0, atol=0)
                torch.testing.assert_close(cache.value[:, : cache.num_tokens], value.transpose(1, 2), rtol=0, atol=0)
            else:
                assert cache.num_tokens == stored_count
                torch.testing.assert_close(cache.key, stored_key, rtol=0, atol=0)
                torch.testing.assert_close(cache.value, stored_value, rtol=0, atol=0)

        call([0, 1], [0, 1], True)
        call([4, 5], [0, 1, 4, 5], True)
        call([8, 9], [0, 1, 4, 5, 8, 9], True)
        call([12, 13], [0, 1, 8, 9, 12, 13], True)
        call([16], [0, 1, 9, 12, 13, 16], False)
        call([16], [0, 1, 9, 12, 13, 16], True)


class TestEdenLiveCausalDeviceMap(EdenLiveTransformerTesterConfig):
    @require_torch_multi_gpu
    @pytest.mark.parametrize("block_device", ["cuda:1", "cpu"])
    @pytest.mark.parametrize("head_device", ["cuda:0", "cuda:1"])
    @torch.no_grad()
    def test_causal_rollout_matches_single_device(self, tmp_path, block_device, head_device):
        torch.manual_seed(0)
        baseline = self.model_class(**self.get_init_dict()).eval()
        baseline.save_pretrained(tmp_path)
        device_map = {name: "cuda:0" for name, _ in baseline.named_children() if name != "blocks"}
        device_map.update(
            {
                "scale_shift_table": "cuda:0",
                "blocks.0": "cuda:0",
                "blocks.1": block_device,
                "norm_out": head_device,
                "proj_out": head_device,
            }
        )
        dispatched = self.model_class.from_pretrained(tmp_path, device_map=device_map).eval()
        baseline.to("cuda:0")
        inputs = {name: value.to("cuda:0") for name, value in self.get_dummy_inputs(height=4, width=4).items()}
        torch.testing.assert_close(dispatched(**inputs).sample, baseline(**inputs).sample, atol=1e-5, rtol=1e-5)
        state_args = {
            "batch_size": 1,
            "height": 4,
            "width": 4,
            "chunk_size": 3,
            "max_context_frames": 6,
            "sink_frames": 3,
        }
        reference_state = baseline.create_causal_state(**state_args, device="cuda:0")
        state = dispatched.create_causal_state(**state_args, device="cuda:0")
        expected_devices = ["cuda:0", "cuda:0" if block_device == "cpu" else block_device]
        assert [str(layer.key.device) for layer in state.layers] == expected_devices
        assert [str(layer.value.device) for layer in state.layers] == expected_devices

        for num_frames in (3, 3, 3, 1):
            chunk_inputs = dict(inputs)
            chunk_inputs["hidden_states"] = inputs["hidden_states"][:, :, :num_frames]
            chunk_inputs["proxy_features"] = inputs["proxy_features"][:, :, :num_frames]
            for commit in (False, True):
                chunk_inputs["timestep"] = torch.tensor([0.0 if commit else 500.0], device="cuda:0")
                expected = baseline(**chunk_inputs, causal_state=reference_state, cache_update=commit).sample
                actual = dispatched(**chunk_inputs, causal_state=state, cache_update=commit).sample
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                assert state.next_frame == reference_state.next_frame
                for layer, reference_layer in zip(state.layers, reference_state.layers):
                    assert layer.num_tokens == reference_layer.num_tokens
                    torch.testing.assert_close(layer.key.cpu(), reference_layer.key.cpu(), atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(layer.value.cpu(), reference_layer.value.cpu(), atol=1e-5, rtol=1e-5)

        state.reset()
        reference_state.reset()
        assert state.next_frame == 0 and all(layer.num_tokens == 0 for layer in state.layers)
        torch.testing.assert_close(
            dispatched(**inputs, causal_state=state).sample,
            baseline(**inputs, causal_state=reference_state).sample,
            atol=1e-5,
            rtol=1e-5,
        )
