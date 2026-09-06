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

import pytest

from diffusers import AutoPipelineForText2Image
from diffusers.models.auto_model import AutoModel

from ..testing_utils import is_torch_available, require_flashpack, require_torch_gpu


if is_torch_available():
    import torch


class TestFlashPack:
    model_id: str = "hf-internal-testing/tiny-flux-pipe"

    # `AutoModel.from_pretrained` builds `_diffusers_load_id` by string-joining the path it is given,
    # so the `tmp_path` fixture has to be passed to it as a `str`.

    @require_flashpack
    def test_save_load_model(self, tmp_path):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        model.save_pretrained(tmp_path, use_flashpack=True)
        assert (tmp_path / "model.flashpack").exists()
        model = AutoModel.from_pretrained(str(tmp_path), use_flashpack=True)

    @require_flashpack
    def test_save_load_pipeline(self, tmp_path):
        pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id)
        pipeline.save_pretrained(tmp_path, use_flashpack=True)
        assert (tmp_path / "transformer" / "model.flashpack").exists()
        assert (tmp_path / "vae" / "model.flashpack").exists()
        pipeline = AutoPipelineForText2Image.from_pretrained(tmp_path, use_flashpack=True)

    @require_torch_gpu
    @require_flashpack
    def test_load_model_device_str(self, tmp_path):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        model.save_pretrained(tmp_path, use_flashpack=True)
        model = AutoModel.from_pretrained(str(tmp_path), use_flashpack=True, device_map={"": "cuda"})
        assert model.device.type == "cuda"

    @require_torch_gpu
    @require_flashpack
    def test_load_model_device(self, tmp_path):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        model.save_pretrained(tmp_path, use_flashpack=True)
        model = AutoModel.from_pretrained(str(tmp_path), use_flashpack=True, device_map={"": torch.device("cuda")})
        assert model.device.type == "cuda"

    @require_flashpack
    def test_load_model_device_auto(self, tmp_path):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        model.save_pretrained(tmp_path, use_flashpack=True)
        with pytest.raises(ValueError):
            model = AutoModel.from_pretrained(str(tmp_path), use_flashpack=True, device_map={"": "auto"})
