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

import pytest
import torch
from safetensors.torch import save_file

from diffusers import BiRefNetModel
from diffusers.utils import is_torchvision_available

from ..pipelines.triposplat.testing_utils import get_triposplat_dummy_components


class TestTripoSplatSingleFile:
    @pytest.mark.parametrize(
        "component_name",
        [
            "transformer",
            "decoder",
            pytest.param(
                "background_remover",
                marks=pytest.mark.skipif(not is_torchvision_available(), reason="BiRefNet requires torchvision"),
            ),
        ],
    )
    @pytest.mark.parametrize("original_format", [False, True])
    @pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
    def test_single_file_loading(self, tmp_path, component_name, original_format, low_cpu_mem_usage):
        if component_name == "background_remover":
            torch.manual_seed(0)
            model = BiRefNetModel(
                embed_dim=8, depths=(1, 1, 1, 1), num_heads=(1, 1, 2, 4), window_size=2, sample_size=64
            )
        else:
            model = get_triposplat_dummy_components()[component_name]
        model.eval()
        model.save_config(tmp_path)
        checkpoint = {}
        for key, value in model.state_dict().items():
            if original_format:
                if component_name == "transformer":
                    key = key.replace(".mlp.net.0.proj.", ".mlp.mlp.0.").replace(".mlp.net.2.", ".mlp.mlp.2.")
                    key = key.replace("cam_refiner.", "cam_refiner.mlp.")
                    key = key.replace(".attn.to_qkv.", ".attn.qkv.").replace(".attn.to_out.", ".attn.out.")
                elif component_name == "decoder":
                    key = key.replace(".q_norm.", ".q_rms_norm.").replace(".k_norm.", ".k_rms_norm.")
                    key = key.replace(".mlp.net.0.proj.", ".mlp.mlp.0.").replace(".mlp.net.2.", ".mlp.mlp.2.")
                elif key.endswith(".atrous_conv.weight"):
                    key = key.replace(".atrous_conv.weight", ".atrous_conv.regular_conv.weight")
            checkpoint[key] = value
        if original_format and component_name == "background_remover":
            checkpoint["decoder.conv_ms_spvn_2.weight"] = torch.ones(1, 8, 1, 1)
            checkpoint["decoder.gdt_convs_pred_2.0.weight"] = torch.ones(1, 8, 1, 1)
        path = tmp_path / "model.safetensors"
        save_file(checkpoint, path)
        loaded = type(model).from_single_file(
            str(path), config=str(tmp_path), dtype=torch.float32, low_cpu_mem_usage=low_cpu_mem_usage
        )
        assert not loaded.training
        assert set(loaded.state_dict()) == set(model.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
