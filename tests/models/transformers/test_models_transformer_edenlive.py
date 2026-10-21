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

from diffusers import EdenLiveTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class EdenLiveTransformerTesterConfig(BaseModelTesterConfig):
    model_class = EdenLiveTransformer3DModel
    main_input_name = "hidden_states"
    input_shape = (4, 3, 8, 8)
    output_shape = (4, 3, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 4,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 2,
            "proxy_channels": 8,
            "proxy_input_channels": 16,
            "rope_max_seq_len": 64,
        }

    def get_dummy_inputs(self, batch_size=1, height=8, width=8):
        generator = self.generator
        return {
            "hidden_states": randn_tensor(
                (batch_size, 4, 3, height, width), generator=generator, device=torch_device, dtype=self.torch_dtype
            ),
            "proxy_features": randn_tensor(
                (batch_size, 16, 3, 4, 4), generator=generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, 12, 16), generator=generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([500.0], device=torch_device).expand(batch_size),
        }


class TestEdenLiveTransformerModel(EdenLiveTransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("proxy_dtype", [torch.float32, torch.bfloat16])
    @torch.no_grad()
    def test_proxy_projection_changes_conditioning(self, proxy_dtype):
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["proxy_features"] = inputs["proxy_features"].to(proxy_dtype)
        expected = model(**inputs).sample
        model.proxy_projection.weight.zero_()
        model.proxy_projection.bias.zero_()
        unconditioned = model(**inputs).sample
        assert torch.isfinite(expected).all() and torch.isfinite(unconditioned).all()
        assert (expected - unconditioned).abs().max() > 1e-6

    @torch.no_grad()
    def test_causal_state_commits_only_clean_chunks_and_preserves_sink(self):
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        inputs = self.get_dummy_inputs()
        tokens_per_frame = (inputs["hidden_states"].shape[-2] // model.config.patch_size[1]) * (
            inputs["hidden_states"].shape[-1] // model.config.patch_size[2]
        )
        state = model.create_causal_state(
            batch_size=1,
            height=8,
            width=8,
            chunk_size=3,
            max_context_frames=18,
            sink_frames=6,
            device=torch_device,
            dtype=torch.float32,
        )
        assert state.next_frame == 0
        model(**inputs, causal_state=state)
        assert state.next_frame == 0
        assert all(layer.num_tokens == 0 for layer in state.layers)
        sink_keys = None
        for index in range(14):
            inputs["timestep"].zero_()
            model(**inputs, causal_state=state, cache_update=True)
            assert state.next_frame == 3 * (index + 1)
            assert all(layer.num_tokens <= 18 * tokens_per_frame for layer in state.layers)
            if index == 1:
                sink_keys = [layer.key[:, : 6 * tokens_per_frame].clone() for layer in state.layers]
            elif index > 1:
                for expected, layer in zip(sink_keys, state.layers):
                    torch.testing.assert_close(layer.key[:, : 6 * tokens_per_frame], expected, atol=0, rtol=0)
        state.reset()
        assert state.next_frame == 0 and all(layer.num_tokens == 0 for layer in state.layers)
        assert not any("cache" in key for key in model.state_dict())


class TestEdenLiveTransformerMemory(EdenLiveTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestEdenLiveTransformerTorchCompile(EdenLiveTransformerTesterConfig, TorchCompileTesterMixin):
    # Keep all shapes in the proxy-resize branch for the zero-recompilation check.
    different_shapes_for_compilation = [(8, 8), (8, 16), (16, 16)]

    @torch.no_grad()
    def test_torch_compile_without_proxy_resize(self):
        model = self.model_class(**self.get_init_dict()).eval().to(torch_device)
        inputs = self.get_dummy_inputs(height=4, width=4)
        expected = model(**inputs).sample
        compiled = torch.compile(model, fullgraph=True)
        with torch._dynamo.config.patch(error_on_recompile=True):
            for _ in range(2):
                actual = compiled(**inputs).sample
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


class TestEdenLiveTransformerTraining(EdenLiveTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestEdenLiveTransformerAttention(EdenLiveTransformerTesterConfig, AttentionTesterMixin):
    pass
