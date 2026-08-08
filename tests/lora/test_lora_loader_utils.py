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
import logging
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import save_file

from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import StableDiffusionLoraLoaderMixin, lora_base
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin


LORA_KEY = "unet.test.lora_A.weight"


def _write_lora_weights(path):
    save_file({LORA_KEY: torch.ones(1)}, path)


@pytest.fixture
def lora_weight_path(tmp_path):
    weight_path = tmp_path / "adapter.safetensors"
    _write_lora_weights(weight_path)
    return weight_path


@pytest.fixture
def model_info_mock(monkeypatch):
    model_info_mock = Mock()
    monkeypatch.setattr(lora_base, "model_info", model_info_mock)
    return model_info_mock


def test_local_directory_in_offline_mode(lora_weight_path, monkeypatch, model_info_mock):
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_weight_path.parent)

    assert torch.equal(state_dict[LORA_KEY], torch.ones(1))
    model_info_mock.assert_not_called()


def test_local_directory_with_local_files_only(lora_weight_path, model_info_mock):
    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_weight_path.parent, local_files_only=True)

    assert torch.equal(state_dict[LORA_KEY], torch.ones(1))
    model_info_mock.assert_not_called()


def test_local_file_in_offline_mode(lora_weight_path, monkeypatch, model_info_mock):
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_weight_path)

    assert torch.equal(state_dict[LORA_KEY], torch.ones(1))
    model_info_mock.assert_not_called()


def test_remote_repository_in_offline_mode_requires_weight_name(monkeypatch, model_info_mock):
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    with pytest.raises(ValueError, match="offline mode.*weight_name"):
        StableDiffusionLoraLoaderMixin.lora_state_dict("organization/repository")

    model_info_mock.assert_not_called()


def test_local_directory_without_matching_files_returns_none(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").touch()
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)

    weight_name = lora_base._best_guess_weight_name(tmp_path)

    assert weight_name is None


def test_local_directory_with_multiple_files_warns_and_uses_first(tmp_path, monkeypatch, caplog):
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    first_path.touch()
    second_path.touch()
    monkeypatch.setattr(lora_base, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(lora_base.os, "listdir", lambda _: [first_path.name, second_path.name])
    monkeypatch.setattr(lora_base.logger, "propagate", True)

    with caplog.at_level(logging.WARNING, logger="diffusers.loaders.lora_base"):
        weight_name = lora_base._best_guess_weight_name(tmp_path)

    assert weight_name == first_path.name
    assert "contains more than one weights file" in caplog.text


def test_unfuse_lora_partial_components_keeps_merged_adapters_in_sync():
    """Regression test for gh-14214.

    Unfusing only a subset of components must keep _merged_adapters in sync
    with the adapters still physically fused in the remaining components.
    """

    class TinyModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
        config_name = "config.json"

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

    class FakePipeline(LoraBaseMixin):
        _lora_loadable_modules = ["unet", "text_encoder"]

        def __init__(self, unet, text_encoder):
            self._merged_adapters = set()
            self.unet, self.text_encoder = unet, text_encoder

    unet = TinyModel()
    text_encoder = TinyModel()
    config = LoraConfig(r=4, lora_alpha=4, target_modules=["linear"], init_lora_weights=False)
    unet.add_adapter(config, adapter_name="adapter")
    text_encoder.add_adapter(config, adapter_name="adapter")

    pipe = FakePipeline(unet, text_encoder)
    pipe.fuse_lora(components=["unet", "text_encoder"], adapter_names=["adapter"])
    assert pipe.num_fused_loras == 1

    pipe.unfuse_lora(components=["text_encoder"])
    assert "adapter" in pipe.fused_loras, "adapter should remain tracked while unet is still fused"
    assert pipe.num_fused_loras == 1

    unet_still_merged = any(isinstance(m, BaseTunerLayer) and len(m.merged_adapters) > 0 for m in unet.modules())
    assert unet_still_merged, "unet should still be physically merged at the PEFT level"

    pipe.unfuse_lora(components=["unet"])
    assert pipe.num_fused_loras == 0
