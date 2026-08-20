# coding=utf-8
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
"""Guards rejecting tensor parallelism combined with quantization, offloading, or LoRA adapters.

Unlike the rest of the tensor-parallel suite in `testing_utils/parallelism.py`, these tests need neither an
accelerator nor more than one rank: every case asserts that a call raises before any collective is issued. They run
single-process on gloo, so they run in ordinary CI.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models._modeling_parallel import ParallelConfig, TensorParallelConfig
from diffusers.models.modeling_utils import ModelMixin


class TinyTPModel(ModelMixin, ConfigMixin):
    """Smallest model carrying a `_tp_plan`: one colwise Linear feeding one rowwise Linear."""

    config_name = "config.json"
    _tp_plan = {"linear_1": "colwise", "linear_2": "rowwise"}
    _supports_group_offloading = True

    @register_to_config
    def __init__(self, hidden_size: int = 8, num_attention_heads: int = 2):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.linear_2(self.linear_1(hidden_states))


class _SerializableQuantizer:
    """Stand-in that passes `save_pretrained`'s serializability check, so the TP guard is what raises."""

    is_serializable = True
    supports_safetensors_serialization = True


@pytest.fixture(scope="module")
def gloo_process_group():
    """A single-rank CPU process group, enough for `_resolve_parallel_config` to build a mesh."""
    if not dist.is_available():
        pytest.skip("torch.distributed is not available.")
    already_initialized = dist.is_initialized()
    if not already_initialized:
        dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29591", world_size=1, rank=0)
    yield
    if not already_initialized:
        dist.destroy_process_group()


def _shard(model):
    model.enable_parallelism(config=TensorParallelConfig(tp_degree=1))


def _mark_as_tensor_parallel(model):
    """Put the model in the state it would be in after sharding, without needing a real mesh."""
    model._parallel_config = ParallelConfig(tensor_parallel_config=TensorParallelConfig(tp_degree=2))
    return model


class TestTensorParallelModelStateGuards:
    """`_check_tp_model_state` — a model whose parameters TP cannot take over."""

    def test_clean_model_reaches_the_device_check(self, gloo_process_group):
        """Ordering guard: with none of the bad states, the device-type check is what rejects CPU.

        This is what keeps the tests below meaningful. If `_check_tp_model_state` ran after the
        `_SUPPORTED_TP_DEVICES` check, every case would raise the device error instead of its own.
        """
        with pytest.raises(ValueError, match="not supported on device type"):
            _shard(TinyTPModel())

    def test_quantized_via_hf_quantizer(self, gloo_process_group):
        model = TinyTPModel()
        model.hf_quantizer = object()
        with pytest.raises(ValueError, match="is quantized"):
            _shard(model)

    def test_quantized_via_is_quantized(self, gloo_process_group):
        model = TinyTPModel()
        model.is_quantized = True
        with pytest.raises(ValueError, match="is quantized"):
            _shard(model)

    def test_device_map_dispatched(self, gloo_process_group):
        model = TinyTPModel()
        model.hf_device_map = {"": 0}
        with pytest.raises(ValueError, match="placed by accelerate"):
            _shard(model)

    def test_accelerate_hook_on_submodule(self, gloo_process_group):
        model = TinyTPModel()
        model.linear_1._hf_hook = object()
        with pytest.raises(ValueError, match="placed by accelerate"):
            _shard(model)

    def test_group_offloaded(self, gloo_process_group, monkeypatch):
        import diffusers.hooks.group_offloading as group_offloading

        monkeypatch.setattr(group_offloading, "_is_group_offload_enabled", lambda module: True)
        with pytest.raises(ValueError, match="group offloading enabled"):
            _shard(TinyTPModel())

    def test_peft_adapter_injected(self, gloo_process_group):
        peft = pytest.importorskip("peft")

        model = TinyTPModel()
        peft.inject_adapter_in_model(peft.LoraConfig(r=2, target_modules=["linear_1"]), model)
        with pytest.raises(ValueError, match=r"adapter \(LoRA\) layers injected"):
            _shard(model)


class TestTensorParallelReverseDirectionGuards:
    """The other order: a model already sharded, then asked to offload or take an adapter."""

    def test_enable_group_offload_on_tp_model(self):
        model = _mark_as_tensor_parallel(TinyTPModel())
        with pytest.raises(ValueError, match="sharded with tensor parallelism"):
            model.enable_group_offload(onload_device=torch.device("cpu"))

    def test_pipeline_offload_helper_detects_tp_component(self):
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline

        model = _mark_as_tensor_parallel(TinyTPModel())
        # The helper takes an explicit module, so it runs without building a whole pipeline.
        with pytest.raises(ValueError, match="sharded with tensor parallelism"):
            DiffusionPipeline._maybe_raise_error_if_tensor_parallel_active(
                DiffusionPipeline, raise_error=True, module=model
            )

    def test_pipeline_offload_helper_passes_for_plain_model(self):
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline

        assert not DiffusionPipeline._maybe_raise_error_if_tensor_parallel_active(
            DiffusionPipeline, raise_error=True, module=TinyTPModel()
        )


class TestTensorParallelSaveGuards:
    """`save_pretrained` must not write a checkpoint that silently drops quantization."""

    def test_dcp_save_rejects_quantized_model(self, tmp_path):
        model = _mark_as_tensor_parallel(TinyTPModel())
        model.hf_quantizer = _SerializableQuantizer()
        with pytest.raises(ValueError, match="quantized tensor-parallel model cannot be saved"):
            model.save_pretrained(str(tmp_path / "dcp"), dcp=True)

    def test_tp_save_rejects_quantized_model(self, tmp_path):
        model = _mark_as_tensor_parallel(TinyTPModel())
        model.hf_quantizer = _SerializableQuantizer()
        with pytest.raises(ValueError, match="quantized tensor-parallel model cannot be saved"):
            model.save_pretrained(str(tmp_path / "full"))
