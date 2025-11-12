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
import tempfile

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from diffusers.loaders.single_file_utils import _extract_repo_id_and_weights_name

from ...testing_utils import (
    backend_empty_cache,
    nightly,
    require_torch_accelerator,
    torch_device,
)


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

    Expected class attributes:
        - model_class: The model class to test
        - pretrained_model_name_or_path: Hub repository ID for the pretrained model
        - ckpt_path: Path or Hub path to the single file checkpoint
        - subfolder: (Optional) Subfolder within the repo
        - torch_dtype: (Optional) torch dtype to use for testing
    """

    pretrained_model_name_or_path = None
    ckpt_path = None

    def setup_method(self):
        """Setup before each test method."""
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        """Cleanup after each test method."""
        gc.collect()
        backend_empty_cache(torch_device)

    def test_single_file_model_config(self):
        """Test that config matches between pretrained and single file loading."""
        pretrained_kwargs = {}
        single_file_kwargs = {}

        pretrained_kwargs["device"] = torch_device
        single_file_kwargs["device"] = torch_device

        if hasattr(self, "subfolder") and self.subfolder:
            pretrained_kwargs["subfolder"] = self.subfolder

        if hasattr(self, "torch_dtype") and self.torch_dtype:
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
        """Test that parameters match between pretrained and single file loading."""
        pretrained_kwargs = {}
        single_file_kwargs = {}

        pretrained_kwargs["device"] = torch_device
        single_file_kwargs["device"] = torch_device

        if hasattr(self, "subfolder") and self.subfolder:
            pretrained_kwargs["subfolder"] = self.subfolder

        if hasattr(self, "torch_dtype") and self.torch_dtype:
            pretrained_kwargs["torch_dtype"] = self.torch_dtype
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, **pretrained_kwargs)
        model_single_file = self.model_class.from_single_file(self.ckpt_path, **single_file_kwargs)

        state_dict = model.state_dict()
        state_dict_single_file = model_single_file.state_dict()

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

            assert torch.allclose(param, param_single_file, rtol=1e-5, atol=1e-5), (
                f"Parameter values differ for {key}: "
                f"max difference {torch.max(torch.abs(param - param_single_file)).item()}"
            )

    def test_single_file_loading_local_files_only(self):
        """Test single file loading with local_files_only=True."""
        single_file_kwargs = {}

        if hasattr(self, "torch_dtype") and self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_model_name_or_path, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(pretrained_model_name_or_path, weight_name, tmpdir)

            model_single_file = self.model_class.from_single_file(
                local_ckpt_path, local_files_only=True, **single_file_kwargs
            )

            assert model_single_file is not None, "Failed to load model with local_files_only=True"

    def test_single_file_loading_with_diffusers_config(self):
        """Test single file loading with diffusers config."""
        single_file_kwargs = {}

        if hasattr(self, "torch_dtype") and self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        # Load with config parameter
        model_single_file = self.model_class.from_single_file(
            self.ckpt_path, config=self.pretrained_model_name_or_path, **single_file_kwargs
        )

        # Load pretrained for comparison
        pretrained_kwargs = {}
        if hasattr(self, "subfolder") and self.subfolder:
            pretrained_kwargs["subfolder"] = self.subfolder
        if hasattr(self, "torch_dtype") and self.torch_dtype:
            pretrained_kwargs["torch_dtype"] = self.torch_dtype

        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path, **pretrained_kwargs)

        # Compare configs
        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs: pretrained={model.config[param_name]}, single_file={param_value}"

    def test_single_file_loading_with_diffusers_config_local_files_only(self):
        """Test single file loading with diffusers config and local_files_only=True."""
        single_file_kwargs = {}

        if hasattr(self, "torch_dtype") and self.torch_dtype:
            single_file_kwargs["torch_dtype"] = self.torch_dtype

        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_model_name_or_path, weight_name = _extract_repo_id_and_weights_name(self.ckpt_path)
            local_ckpt_path = download_single_file_checkpoint(pretrained_model_name_or_path, weight_name, tmpdir)
            local_diffusers_config = download_diffusers_config(self.pretrained_model_name_or_path, tmpdir)

            model_single_file = self.model_class.from_single_file(
                local_ckpt_path, config=local_diffusers_config, local_files_only=True, **single_file_kwargs
            )

            assert model_single_file is not None, "Failed to load model with config and local_files_only=True"

    def test_single_file_loading_dtype(self):
        """Test single file loading with different dtypes."""
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
        """Test loading checkpoints with alternate keys/variants if provided."""
        if not hasattr(self, "alternate_ckpt_paths") or not self.alternate_ckpt_paths:
            return

        for ckpt_path in self.alternate_ckpt_paths:
            backend_empty_cache(torch_device)

            single_file_kwargs = {}
            if hasattr(self, "torch_dtype") and self.torch_dtype:
                single_file_kwargs["torch_dtype"] = self.torch_dtype

            model = self.model_class.from_single_file(ckpt_path, **single_file_kwargs)

            assert model is not None, f"Failed to load checkpoint from {ckpt_path}"

            del model
            gc.collect()
            backend_empty_cache(torch_device)
