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

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import ContextParallelConfig, EdenLiveTransformer3DModel
from diffusers.models.transformers import transformer_edenlive

from ...testing_utils import enable_full_determinism
from .test_models_transformer_edenlive import EdenLiveTransformerTesterConfig


class TestEdenLiveCausalCacheOrder:
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


def _parallel_outputs(model, inputs):
    with torch.no_grad():
        output = model(**inputs, return_dict=False)[0]
        state = model.create_causal_state(
            batch_size=1,
            height=8,
            width=8,
            chunk_size=3,
            max_context_frames=18,
            sink_frames=6,
            device=inputs["hidden_states"].device,
            dtype=inputs["hidden_states"].dtype,
        )
        clean_inputs = {**inputs, "timestep": torch.zeros_like(inputs["timestep"])}
        sink_keys = None
        for index in range(8):
            model(**clean_inputs, causal_state=state, cache_update=True)
            assert state.next_frame == 3 * (index + 1)
            assert all(layer.num_tokens <= 18 * state.tokens_per_frame // state.cp_size for layer in state.layers)
            sink_tokens = 6 * state.tokens_per_frame // state.cp_size
            if index == 1:
                sink_keys = [layer.key[:, :sink_tokens].clone() for layer in state.layers]
            elif index > 1:
                for expected, layer in zip(sink_keys, state.layers):
                    torch.testing.assert_close(layer.key[:, :sink_tokens], expected, rtol=0, atol=0)
        tail_inputs = {
            **inputs,
            "hidden_states": inputs["hidden_states"][:, :, :1],
            "proxy_features": inputs["proxy_features"][:, :, :1],
            "timestep": inputs["timestep"][:, :16] if inputs["timestep"].ndim == 2 else inputs["timestep"],
        }
        tail = model(**tail_inputs, causal_state=state, return_dict=False)[0]
        assert state.next_frame == 24
        tail_clean = {**tail_inputs, "timestep": torch.zeros_like(tail_inputs["timestep"])}
        model(**tail_clean, causal_state=state, cache_update=True)
        assert state.next_frame == 25
        return [output.float().cpu(), tail.float().cpu()]


def _context_parallel_worker(rank, world_size, port, init_dict, state_dict, inputs, backend, use_fsdp, output_path):
    # Spawned workers must establish the same settings as the single-device reference.
    enable_full_determinism()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16 if backend == "_native_flash" else torch.float32
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=180))
        model = EdenLiveTransformer3DModel(**init_dict).eval()
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)
        model.set_attention_backend(backend)
        inputs = {
            name: tensor.to(device=device, dtype=torch.float32 if name == "timestep" else dtype)
            for name, tensor in inputs.items()
        }

        def check_timestep(_module, _args, kwargs):
            assert kwargs["timestep"].dtype == torch.float32
            assert torch.isin(kwargs["timestep"], torch.tensor([0.0, 375.0], device=device)).all()

        model.register_forward_pre_hook(check_timestep, with_kwargs=True)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("ring", "ulysses"))
        model.enable_parallelism(config=ContextParallelConfig(ring_degree=1, ulysses_degree=world_size, mesh=mesh))
        if use_fsdp:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            policy = MixedPrecisionPolicy(
                param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype, cast_forward_inputs=False
            )
            for block in model.blocks:
                fully_shard(block, mesh=mesh["ulysses"], mp_policy=policy, reshard_after_forward=True)
            fully_shard(model, mesh=mesh["ulysses"], mp_policy=policy, reshard_after_forward=True)
        outputs = _parallel_outputs(model, inputs)
        if rank == 0:
            torch.save(outputs, output_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestEdenLiveContextParallel(EdenLiveTransformerTesterConfig):
    @pytest.mark.context_parallel
    @pytest.mark.attention
    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize(
        "backend,use_fsdp", [("native", False), ("_native_flash", False), ("_native_flash", True)]
    )
    @pytest.mark.parametrize("num_layers,token_time", [(2, False), (1, True)])
    def test_context_parallel_inference(self, tmp_path, world_size, backend, use_fsdp, num_layers, token_time):
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry

        if not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < world_size:
            pytest.skip(f"This case requires {world_size} CUDA devices with NCCL.")
        previous_backend, _ = _AttentionBackendRegistry.get_active_backend()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            enable_full_determinism()
            config = {**self.get_init_dict(), "attention_head_dim": 16, "num_layers": num_layers}
            model = EdenLiveTransformer3DModel(**config).eval()
            state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}
            inputs = {name: tensor.cpu() for name, tensor in self.get_dummy_inputs().items()}
            # This value would be rounded to 376 by an indiscriminate BF16 input cast.
            inputs["timestep"] = torch.tensor([375.0], dtype=torch.float32)
            if token_time:
                inputs["timestep"] = torch.full((1, 48), 375.0)
                inputs["timestep"][:, :16] = 0
            dtype = torch.bfloat16 if backend == "_native_flash" else torch.float32
            model.to(device="cuda:0", dtype=dtype).set_attention_backend(backend)
            reference_inputs = {
                name: tensor.to(device="cuda:0", dtype=torch.float32 if name == "timestep" else dtype)
                for name, tensor in inputs.items()
            }
            expected = _parallel_outputs(model, reference_inputs)
            del model, reference_inputs
            torch.cuda.empty_cache()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            output_path = str(tmp_path / "context_parallel.pt")
            mp.spawn(
                _context_parallel_worker,
                args=(world_size, port, config, state, inputs, backend, use_fsdp, output_path),
                nprocs=world_size,
                join=True,
            )
            actual = torch.load(output_path, map_location="cpu", weights_only=True)
            for reference, output in zip(expected, actual):
                assert torch.isfinite(output).all()
                # Keep the existing small-model CP backend tolerance.
                torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
        finally:
            _AttentionBackendRegistry.set_active_backend(previous_backend)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
