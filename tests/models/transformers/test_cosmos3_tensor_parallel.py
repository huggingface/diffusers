# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from diffusers import Cosmos3OmniTransformer
from diffusers.hooks.tensor_parallel import resolve_tp_shard_specs
from diffusers.models._modeling_parallel import TensorParallelConfig
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME

from ...testing_utils import is_tensor_parallel, require_torch_multi_accelerator, torch_device
from .test_models_transformer_cosmos3 import Cosmos3OmniTransformerTesterConfig


class Cosmos3TensorParallelTesterConfig(Cosmos3OmniTransformerTesterConfig):
    def get_init_dict(self):
        return {
            **super().get_init_dict(),
            "head_dim": 8,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "rope_axes_dim": [2, 1, 1],
            "attention_bias": True,
        }

    def get_dummy_inputs(self, und_len=5, gen_len=7, dtype=torch.float32):
        inputs = super().get_dummy_inputs(width=gen_len)
        device = inputs["input_ids"].device
        indexes = torch.arange(und_len, und_len + gen_len, device=device)
        inputs.update(
            input_ids=torch.arange(und_len, device=device) % self.get_init_dict()["vocab_size"],
            text_indexes=torch.arange(und_len, device=device),
            position_ids=torch.arange(und_len + gen_len, device=device).expand(3, -1),
            und_len=und_len,
            sequence_length=und_len + gen_len,
            vision_tokens=[tokens.to(dtype=dtype) for tokens in inputs["vision_tokens"]],
            vision_sequence_indexes=indexes,
            vision_mse_loss_indexes=indexes,
        )
        return inputs

    @pytest.fixture(params=["silu", "relu2"])
    def model_config(self, request):
        activation = request.param
        return {
            **self.get_init_dict(),
            "hidden_act": activation,
            "qk_norm_for_text": activation == "silu",
            "use_und_k_norm_for_gen": activation == "relu2",
        }


def _cosmos_tp_worker(rank, world_size, rendezvous, checkpoint, dtype):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(minutes=3)
    )
    try:
        torch.manual_seed(2026)
        reference = Cosmos3OmniTransformer.from_pretrained(checkpoint, dtype=dtype).to(device).eval()
        native = copy.deepcopy(reference)
        specs = resolve_tp_shard_specs(native, native._tp_plan, world_size)
        # Planned weights come from rank zero even when other ranks start with different values.
        if rank != 0:
            with torch.no_grad():
                for name, parameter in native.named_parameters():
                    if name in specs:
                        parameter.add_(1)
        native.enable_parallelism(config=TensorParallelConfig(tp_degree=world_size))
        loaded = Cosmos3OmniTransformer.from_pretrained(
            checkpoint, dtype=dtype, parallel_config=TensorParallelConfig(tp_degree=world_size)
        ).eval()

        assert loaded.rotary_emb.inv_freq.dtype == torch.float32
        torch.testing.assert_close(loaded.rotary_emb.inv_freq.to(device), reference.rotary_emb.inv_freq)
        for name, parameter in loaded.named_parameters():
            other = native.get_parameter(name)
            assert isinstance(parameter, DTensor) == (name in specs)
            assert parameter.device == device
            if isinstance(parameter, DTensor):
                assert parameter.placements == other.placements
                parameter, other = parameter.to_local(), other.to_local()
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)
            if name.startswith("time_embedder."):
                assert parameter.dtype == torch.float32

        tolerance = 0.025 if dtype == torch.bfloat16 else 3e-5
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for u, g in ((5, 7), (8, 12), (7, 5)):
                inputs = Cosmos3TensorParallelTesterConfig().get_dummy_inputs(und_len=u, gen_len=g, dtype=dtype)
                expected = reference(**inputs).sample[0]
                actual = native(**inputs).sample[0]
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(loaded(**inputs).sample[0], actual, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


class TestCosmos3TensorParallelConfig(Cosmos3TensorParallelTesterConfig):
    def test_plan_covers_both_streams(self, model_config):
        model = self.model_class(**model_config)
        specs = resolve_tp_shard_specs(model, model._tp_plan, 2)
        for name, parameter in model.named_parameters():
            if name.startswith("layers.") and parameter.ndim == 2:
                assert name in specs
        assert "embed_tokens.weight" not in specs
        assert any("gate_proj" in key for key in specs) == (model_config["hidden_act"] == "silu")
        assert any("gate_proj" in key for key in Cosmos3OmniTransformer._tp_plan)

    @pytest.mark.parametrize("heads,kv_heads,degree", [(8, 1, 2), (6, 4, 4), (8, 4, 3)])
    def test_invalid_head_split(self, heads, kv_heads, degree):
        config = self.get_init_dict()
        config.update(num_attention_heads=heads, num_key_value_heads=kv_heads)
        model = Cosmos3OmniTransformer(**config)
        with pytest.raises(ValueError, match="both query and KV"):
            model._resolve_parallel_config(TensorParallelConfig(tp_degree=degree))
        assert model._parallel_config is None


@is_tensor_parallel
@require_torch_multi_accelerator
class TestCosmos3TensorParallelLoading(Cosmos3TensorParallelTesterConfig):
    """Cosmos' nested modality inputs and list outputs require a worker beyond the flat-tensor TP mixin."""

    @pytest.fixture
    def checkpoint(self, tmp_path, model_config):
        torch.manual_seed(2026)
        model = self.model_class(**model_config)
        checkpoint = tmp_path / "checkpoint"
        model.save_pretrained(checkpoint, max_shard_size="100KB")
        return checkpoint

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_checkpoint_and_loaded_model(self, tmp_path, checkpoint, dtype):
        if torch_device != "cuda":
            pytest.skip("This worker uses NCCL.")
        assert (checkpoint / SAFE_WEIGHTS_INDEX_NAME).is_file()
        mp.spawn(
            _cosmos_tp_worker,
            args=(2, (tmp_path / "rendezvous").as_uri(), str(checkpoint), dtype),
            nprocs=2,
            join=True,
        )
