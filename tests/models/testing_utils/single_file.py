# coding=utf-8
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

import gc

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from diffusers.loaders.single_file_utils import _extract_repo_id_and_weights_name

from ...testing_utils import (
    backend_empty_cache,
    is_single_file,
    nightly,
    require_torch_accelerator,
    torch_device,
)
from .common import check_device_map_is_respected


def download_single_file_checkpoint(pretrained_model_name_or_path, filename, tmpdir):
    """Download a single file checkpoint from the Hub to a temporary directory."""
    path = hf_hub_download(pretrained_model_name_or_path, filename=filename, local_dir=tmpdir)
    return path


def download_diffusers_config(pretrained_model_name_or_path, tmpdir):
    """Download diffusers config files (excluding weights) from a repository."""
    path = snapshot_download(
        pretrained_model_name_or_path,
        ignore_patterns=[
            "**/*.ckpt",
            "*.ckpt",
            "**/*.bin",
            "*.bin",
            "**/*.pt",
            "*.pt",
            "**/*.safetensors",
            "*.safetensors",
        ],
        allow_patterns=["**/*.json", "*.json", "*.txt", "**/*.txt"],
        local_dir=tmpdir,
    )
    return path


@nightly
@require_torch_accelerator
@is_single_file
class SingleFileTesterMixin:
    """
    Mixin class for testing single file loading for models.

    Required properties (must be implemented by subclasses):
        - ckpt_path: Path or Hub path to the single file checkpoint

    Optional properties:
        - torch_dtype: torch dtype to use for testing (default: None)
        - alternate_ckpt_paths: List of alternate checkpoint paths for variant testing (default: None)

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

    @property
    def alternate_ckpt_paths(self) -> list[str] | None:
        """List of alternate checkpoint paths for variant testing."""
        return None

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_single_file_model_config(self):
        pretrained_kwargs = {"device": torch_device, **self.pretrained_model_kwargs}
        single_file_kwargs = {"device": torch_device}

        if self.torch_dtype:
            pretrained_kwargs["torch_dtype"] = self.torch_dtype
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, **pretrained_kwargs)
        model_single_file = self.model_class.from_single_file(self.ckpt_path, **single_file_kwargs)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert model.config[param_name] == param_value, (
                f"{param_name} differs between pretrained loading and single file loading: "
                f"pretrained={model.config[param_name]}, single_file={param_value}"
            )

    def test_single_file_model_parameters(self):
        pretrained_kwargs = {"device_map": str(torch_device), **self.pretrained_model_kwargs}
        single_file_kwargs = {"device": torch_device}

        if self.torch_dtype:
            pretrained_kwargs["torch_dtype"] = self.torch_dtype
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        # Load pretrained model, get state dict on CPU, then free GPU memory
        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, **pretrained_kwargs)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        del model
        gc.collect()
        backend_empty_cache(torch_device)

        # Load single file model, get state dict on CPU
        model_single_file = self.model_class.from_single_file(self.ckpt_path, **single_file_kwargs)
        state_dict_single_file = {k: v.cpu() for k, v in model_single_file.state_dict().items()}
        del model_single_file
        gc.collect()
        backend_empty_cache(torch_device)

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

    def test_single_file_loading_local_files_only(self, tmp_path):
        single_file_kwargs = {}

        if self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        pretrained_model_name_or_path, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
        local_ckpt_path = download_single_file_checkpoint(pretrained_model_name_or_path, weight_name, str(tmp_path))

        model_single_file = self.model_class.from_single_file(
            local_ckpt_path, local_files_only=True, **single_file_kwargs
        )

        assert model_single_file is not None, "Failed to load model with local_files_only=True"

    def test_single_file_loading_with_diffusers_config(self):
        single_file_kwargs = {}

        if self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype
        single_file_kwargs.update(self.pretrained_model_kwargs)

        # Load with config parameter
        model_single_file = self.model_class.from_single_file(
            self.ckpt_path, config=self.pretrained_model_name_or_path, **single_file_kwargs
        )

        # Load pretrained for comparison
        pretrained_kwargs = {**self.pretrained_model_kwargs}
        if self.torch_dtype:
            pretrained_kwargs["torch_dtype"] = self.torch_dtype

        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, **pretrained_kwargs)

        # Compare configs
        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert model.config[param_name] == param_value, (
                f"{param_name} differs: pretrained={model.config[param_name]}, single_file={param_value}"
            )

    def test_single_file_loading_with_diffusers_config_local_files_only(self, tmp_path):
        single_file_kwargs = {}

        if self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype
        single_file_kwargs.update(self.pretrained_model_kwargs)

        pretrained_model_name_or_path, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
        local_ckpt_path = download_single_file_checkpoint(pretrained_model_name_or_path, weight_name, str(tmp_path))
        local_diffusers_config = download_diffusers_config(self.pretrained_model_name_or_path, str(tmp_path))

        model_single_file = self.model_class.from_single_file(
            local_ckpt_path, config=local_diffusers_config, local_files_only=True, **single_file_kwargs
        )

        assert model_single_file is not None, "Failed to load model with config and local_files_only=True"

    def test_single_file_loading_dtype(self):
        for dtype in [torch.float32, torch.float16]:
            if torch_device == "mps" and dtype == torch.bfloat16:
                continue

            model_single_file = self.model_class.from_single_file(self.ckpt_path, torch_dtype=dtype)

            assert model_single_file.dtype == dtype, f"Expected dtype {dtype}, got {model_single_file.dtype}"

            # Cleanup
            del model_single_file
            gc.collect()
            backend_empty_cache(torch_device)

    def test_checkpoint_variant_loading(self):
        if not self.alternate_ckpt_paths:
            return

        for ckpt_path in self.alternate_ckpt_paths:
            backend_empty_cache(torch_device)

            single_file_kwargs = {}
            if self.torch_dtype:
                single_file_kwargs["torch_dtype"] = self.torch_dtype

            model = self.model_class.from_single_file(ckpt_path, **single_file_kwargs)

            assert model is not None, f"Failed to load checkpoint from {ckpt_path}"

            del model
            gc.collect()
            backend_empty_cache(torch_device)

    def test_single_file_loading_with_device_map(self):
        single_file_kwargs = {"device_map": torch_device}

        if self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        model = self.model_class.from_single_file(self.ckpt_path, **single_file_kwargs)

        assert model is not None, "Failed to load model with device_map"
        assert hasattr(model, "hf_device_map"), "Model should have hf_device_map attribute when loaded with device_map"
        assert model.hf_device_map is not None, "hf_device_map should not be None when loaded with device_map"
        check_device_map_is_respected(model, model.hf_device_map)
