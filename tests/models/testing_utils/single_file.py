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

import functools
import gc

import huggingface_hub
import pytest
import torch
from huggingface_hub import HfApi
from safetensors.torch import _getdtype

from diffusers.loaders.single_file_utils import _extract_repo_id_and_weights_name

from ...testing_utils import (
    backend_empty_cache,
    is_single_file,
    nightly,
    torch_device,
)
from .common import check_device_map_is_respected


# Config entries that legitimately differ between a checkpoint that's loaded with `from_pretrained()` and the one loaded with `from_single_file()`.
PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]


@functools.lru_cache(maxsize=None)
def fetch_checkpoint_metadata(ckpt_path):
    """
    Fetch a single file checkpoint's keys, shapes and dtypes without downloading its weights.

    Only the safetensors header crosses the network, and the result is cached for the session because every test
    in a class asks for the same one. Returns `None` for checkpoints that are not safetensors.
    """
    pretrained_model_name_or_path, weight_name = _extract_repo_id_and_weights_name(ckpt_path)
    if not weight_name.endswith(".safetensors"):
        return None

    return HfApi().parse_safetensors_file_metadata(pretrained_model_name_or_path, weight_name)


@is_single_file
class SingleFileTesterMixin:
    """
    Mixin class for testing single file loading for models.

    Required properties (must be implemented by subclasses):
        - ckpt_path: Path or Hub path to the single file checkpoint

    Optional properties:
        - torch_dtype: torch dtype to use for testing (default: None)

    Expected from config mixin:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - pretrained_model_kwargs: Additional kwargs for from_pretrained (e.g., subfolder)

    Pytest mark: single_file
        Use `pytest -m "not single_file"` to skip these tests
    """

    # ==================== Required Properties ====================

    @property
    def ckpt_path(self) -> str:
        """Path or Hub path to the single file checkpoint. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the `ckpt_path` property.")

    # ==================== Optional Properties ====================

    @property
    def torch_dtype(self) -> torch.dtype | None:
        """torch dtype to use for single file testing."""
        return None

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def get_dummy_single_file_state_dict(self):
        """
        A stand-in for the single file checkpoint's state dict.

        Keys, shapes and dtypes match the real checkpoint exactly, while the tensors themselves are on meta and
        hold no data. Pass it to `from_single_file` with `device="meta"` to exercise the loading path without
        materializing a weight. The conversion functions consume the dict they are handed, so build a fresh one
        for every load.
        """
        metadata = fetch_checkpoint_metadata(self.ckpt_path)
        if metadata is None:
            pytest.skip(f"{self.ckpt_path} is not a safetensors checkpoint")

        return {
            name: torch.empty(tuple(tensor.shape), dtype=_getdtype(tensor.dtype), device="meta")
            for name, tensor in metadata.tensors.items()
        }

    def test_single_file_model_config(self):
        # An empty model supplies the registered defaults (`out_channels` and friends) that a bare config dict
        # would be missing.
        with torch.device("meta"):
            model = self.model_class.from_config(self.pretrained_model_name_or_path, **self.pretrained_model_kwargs)

        config = model.config
        model_single_file = self.model_class.from_single_file(self.get_dummy_single_file_state_dict(), device="meta")

        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert config[param_name] == param_value, (
                f"{param_name} differs between pretrained loading and single file loading: "
                f"pretrained={config[param_name]}, single_file={param_value}"
            )

    def test_single_file_loading_local_files_only(self, monkeypatch):
        state_dict = self.get_dummy_single_file_state_dict()

        # The checkpoint is already in memory, so the config is the only thing left to resolve. Fetch it into the
        # cache the way a previous run would have, and keep it as the reference to compare against.
        config = self.model_class.load_config(self.pretrained_model_name_or_path, **self.pretrained_model_kwargs)

        # Cut the Hub off, so anything the load does not find locally raises instead of quietly downloading.
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

        model_single_file = self.model_class.from_single_file(state_dict, local_files_only=True, device="meta")

        # Resolving from the cache has to land on the same config as resolving from the Hub.
        for param_name, param_value in config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert model_single_file.config[param_name] == param_value, (
                f"{param_name} differs when loading with local_files_only=True: "
                f"pretrained={param_value}, single_file={model_single_file.config[param_name]}"
            )

    def test_single_file_loading_with_diffusers_config(self):
        model_single_file = self.model_class.from_single_file(
            self.get_dummy_single_file_state_dict(),
            config=self.pretrained_model_name_or_path,
            device="meta",
            **self.pretrained_model_kwargs,
        )

        # An empty model supplies the registered defaults (`out_channels` and friends) that a bare config dict
        # would be missing.
        with torch.device("meta"):
            model = self.model_class.from_config(self.pretrained_model_name_or_path, **self.pretrained_model_kwargs)

        config = model.config

        # Compare configs
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert config[param_name] == param_value, (
                f"{param_name} differs: pretrained={config[param_name]}, single_file={param_value}"
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    def test_single_file_loading_dtype(self, dtype):
        model_single_file = self.model_class.from_single_file(
            self.get_dummy_single_file_state_dict(), device="meta", dtype=dtype
        )
        for name, param in model_single_file.named_parameters():
            assert param.dtype == dtype, f"Parameter {name} should be {dtype} but got {param.dtype}"

    @nightly
    def test_single_file_model_parameters(self):
        pretrained_kwargs = {**self.pretrained_model_kwargs}
        single_file_kwargs = {}

        if self.torch_dtype:
            pretrained_kwargs["dtype"] = self.torch_dtype
            single_file_kwargs["dtype"] = self.torch_dtype

        # Both models load onto the CPU, so keep only the state dicts and drop the models themselves.
        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, **pretrained_kwargs)
        state_dict = model.state_dict()
        del model
        gc.collect()

        model_single_file = self.model_class.from_single_file(self.ckpt_path, **single_file_kwargs)
        state_dict_single_file = model_single_file.state_dict()
        del model_single_file
        gc.collect()

        assert set(state_dict.keys()) == set(state_dict_single_file.keys()), (
            "Model parameters keys differ between pretrained and single file loading. "
            f"Missing in single file: {set(state_dict.keys()) - set(state_dict_single_file.keys())}. "
            f"Extra in single file: {set(state_dict_single_file.keys()) - set(state_dict.keys())}"
        )

        for key in state_dict.keys():
            param = state_dict[key]
            param_single_file = state_dict_single_file[key]

            assert param.shape == param_single_file.shape, (
                f"Parameter shape mismatch for {key}: "
                f"pretrained {param.shape} vs single file {param_single_file.shape}"
            )

            assert torch.equal(param, param_single_file), f"Parameter values differ for {key}"

    @nightly
    def test_single_file_loading_with_device_map(self):
        single_file_kwargs = {"device_map": "auto"}

        if self.torch_dtype:
            single_file_kwargs["dtype"] = self.torch_dtype

        model = self.model_class.from_single_file(self.ckpt_path, **single_file_kwargs)

        assert model is not None, "Failed to load model with device_map"
        assert hasattr(model, "hf_device_map"), "Model should have hf_device_map attribute when loaded with device_map"
        assert model.hf_device_map is not None, "hf_device_map should not be None when loaded with device_map"
        check_device_map_is_respected(model, model.hf_device_map)
