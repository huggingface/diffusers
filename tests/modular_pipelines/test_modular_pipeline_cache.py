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

from pathlib import Path

import pytest
import torch
from huggingface_hub import constants, hf_hub_download, try_to_load_from_cache
from huggingface_hub.file_download import are_symlinks_supported, is_xet_available

from diffusers import ModularPipeline


SOURCE_REPO = "hf-internal-testing/tiny-modular-cache-source"
SOURCE_REVISION = "907560ec007d1f62a5924d3fbc6fae7be49d30c4"
REFERENCE_REPO = "hf-internal-testing/tiny-modular-cache-reference"
REFERENCE_REVISION = "36c88dbb663b35b7f52d79bbd2fcef430ca7acbf"
WEIGHTS_NAME = "vae/diffusion_pytorch_model.safetensors"


class TestModularPipelineCache:
    def test_cross_repository_component_reference(self, tmp_path):
        source = ModularPipeline.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION, cache_dir=tmp_path)
        source.load_components(names="vae", revision=SOURCE_REVISION, cache_dir=tmp_path)
        assert source.vae is not None

        reference = ModularPipeline.from_pretrained(REFERENCE_REPO, revision=REFERENCE_REVISION, cache_dir=tmp_path)
        reference.load_components(names="vae", cache_dir=tmp_path, local_files_only=True)
        assert reference.vae is not None
        assert reference.get_component_spec("vae").pretrained_model_name_or_path == SOURCE_REPO
        assert reference.vae._diffusers_load_id == source.vae._diffusers_load_id
        for reference_param, source_param in zip(reference.vae.parameters(), source.vae.parameters(), strict=True):
            assert torch.equal(reference_param, source_param)
        assert (
            try_to_load_from_cache(REFERENCE_REPO, WEIGHTS_NAME, revision=REFERENCE_REVISION, cache_dir=tmp_path)
            is None
        )

    @pytest.mark.parametrize("disable_shared_blobs", [False, True], ids=["shared", "repo-local"])
    def test_load_component_from_both_repositories(self, tmp_path, monkeypatch, disable_shared_blobs):
        monkeypatch.setattr(constants, "HF_HUB_DISABLE_SHARED_BLOBS", disable_shared_blobs)
        if not disable_shared_blobs and not is_xet_available():
            pytest.skip("Shared blobs require Xet downloads")
        if not are_symlinks_supported(tmp_path):
            pytest.skip("Symlinks are required")

        source = ModularPipeline.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION, cache_dir=tmp_path)
        source.load_components(names="vae", revision=SOURCE_REVISION, cache_dir=tmp_path)
        assert source.vae is not None

        reference = ModularPipeline.from_pretrained(REFERENCE_REPO, revision=REFERENCE_REVISION, cache_dir=tmp_path)
        reference.load_components(
            names="vae",
            pretrained_model_name_or_path=REFERENCE_REPO,
            revision=REFERENCE_REVISION,
            cache_dir=tmp_path,
        )
        assert reference.vae is not None
        assert reference.get_component_spec("vae").pretrained_model_name_or_path == REFERENCE_REPO
        for reference_param, source_param in zip(reference.vae.parameters(), source.vae.parameters(), strict=True):
            assert torch.equal(reference_param, source_param)

        paths = [
            Path(hf_hub_download(repo, WEIGHTS_NAME, revision=revision, cache_dir=tmp_path, local_files_only=True))
            for repo, revision in ((SOURCE_REPO, SOURCE_REVISION), (REFERENCE_REPO, REFERENCE_REVISION))
        ]
        assert paths[0] != paths[1]
        assert paths[0].samefile(paths[1]) == (not disable_shared_blobs)
        assert paths[0].read_bytes() == paths[1].read_bytes()

        reference.unload_components("vae")
        reference.load_components(names="vae", cache_dir=tmp_path, local_files_only=True)
        assert reference.vae is not None
        for reference_param, source_param in zip(reference.vae.parameters(), source.vae.parameters(), strict=True):
            assert torch.equal(reference_param, source_param)
