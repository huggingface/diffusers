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
import torch

from diffusers import MagiTextConditioningModel
from diffusers.utils.torch_utils import randn_tensor

from ..testing_utils import enable_full_determinism, torch_device
from .testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class MagiTextConditioningTesterConfig(BaseModelTesterConfig):
    main_input_name = "hidden_states"

    @property
    def model_class(self):
        return MagiTextConditioningModel

    @property
    def pretrained_model_name_or_path(self):
        return None

    @property
    def pretrained_model_kwargs(self):
        return {}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {"caption_channels": 16, "caption_max_length": 8, "null_token_length": 4}

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((2, 8, 16), generator=self.generator, device=torch_device),
            "attention_mask": torch.ones(2, 8, device=torch_device, dtype=torch.bool),
            "num_chunks": 2,
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (8, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (2, 8, 16)


class TestMagiTextConditioningModel(MagiTextConditioningTesterConfig, ModelTesterMixin):
    @torch.no_grad()
    def test_special_token_layout_and_duration_clamping(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        model.special_embedding.weight.copy_(
            torch.arange(9, device=torch_device, dtype=torch.float32)[:, None].expand(-1, 16)
        )
        model.null_embedding.weight.copy_(
            torch.arange(8, device=torch_device, dtype=torch.float32)[:, None].expand(-1, 16)
        )
        inputs = self.get_dummy_inputs()
        inputs["attention_mask"][:, -3:] = False
        inputs["num_chunks"] = 10
        output = model(**inputs)

        duration_indices = torch.tensor([8, 8, 8, 7, 6, 5, 4, 3, 2, 1], device=torch_device)
        expected_duration = model.special_embedding.weight[duration_indices][None].expand(2, -1, -1)
        expected_hq = model.special_embedding.weight[0][None, None].expand(2, 10, -1)
        expected_text = inputs["hidden_states"][:, None, :-2].expand(-1, 10, -1, -1)
        expected_mask = inputs["attention_mask"][:, None, :-2].expand(-1, 10, -1)
        torch.testing.assert_close(output.sample[:, :, 0], expected_duration, atol=0, rtol=0)
        torch.testing.assert_close(output.sample[:, :, 1], expected_hq, atol=0, rtol=0)
        torch.testing.assert_close(output.sample[:, :, 2:], expected_text, atol=0, rtol=0)
        torch.testing.assert_close(output.attention_mask[:, :, 2:], expected_mask, atol=0, rtol=0)
        assert output.attention_mask[:, :, :2].all()
        torch.testing.assert_close(
            output.negative_prompt_embeds,
            model.null_embedding.weight[None].expand(2, -1, -1),
            atol=0,
            rtol=0,
        )
        expected_null_mask = (torch.arange(8, device=torch_device) < 4)[None].expand(2, -1)
        torch.testing.assert_close(output.negative_prompt_attention_mask, expected_null_mask, atol=0, rtol=0)

    @torch.no_grad()
    def test_batch_outputs_do_not_alias(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        output = model(**self.get_dummy_inputs())
        second_null = output.negative_prompt_embeds[1].clone()
        second_mask = output.negative_prompt_attention_mask[1].clone()
        output.negative_prompt_embeds[0].zero_()
        output.negative_prompt_attention_mask[0].zero_()
        torch.testing.assert_close(output.negative_prompt_embeds[1], second_null, atol=0, rtol=0)
        torch.testing.assert_close(output.negative_prompt_attention_mask[1], second_mask, atol=0, rtol=0)

    def test_invalid_inputs(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        with pytest.raises(ValueError, match="floating-point tensor shaped"):
            model(inputs["hidden_states"][0], inputs["attention_mask"])
        with pytest.raises(ValueError, match="floating-point tensor shaped"):
            model(inputs["hidden_states"].long(), inputs["attention_mask"])
        with pytest.raises(ValueError, match="attention_mask must match"):
            model(inputs["hidden_states"], inputs["attention_mask"][:1])
        for num_chunks in (0, 1.5, True):
            with pytest.raises(ValueError, match="positive integer"):
                model(inputs["hidden_states"], inputs["attention_mask"], num_chunks=num_chunks)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_from_save_pretrained_dtype(self, tmp_path, dtype):
        self.check_conditioning_dtype(tmp_path, torch_dtype=dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_from_pretrained_dtype_alias(self, tmp_path, dtype):
        self.check_conditioning_dtype(tmp_path, dtype=dtype)

    @pytest.mark.skip(reason="The small conditioning tables are kept together, not sharded across devices.")
    def test_model_parallelism(self):
        pass

    def check_conditioning_dtype(self, tmp_path, **kwargs):
        model = self.model_class(**self.get_init_dict())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_()
        model.save_pretrained(tmp_path)
        restored = self.model_class.from_pretrained(tmp_path, **kwargs)
        assert all(parameter.dtype == torch.float32 for parameter in restored.parameters())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


class TestMagiTextConditioningMemory(MagiTextConditioningTesterConfig, MemoryTesterMixin):
    @pytest.mark.skip(reason="Embedding tables are excluded from layerwise casting.")
    def test_layerwise_casting_memory(self):
        pass

    @pytest.mark.skip(reason="The small conditioning tables are offloaded as one component, not split.")
    def test_cpu_offload(self):
        pass

    @pytest.mark.skip(reason="The small conditioning tables are offloaded as one component, not split.")
    def test_disk_offload_without_safetensors(self):
        pass

    @pytest.mark.skip(reason="The small conditioning tables are offloaded as one component, not split.")
    def test_disk_offload_with_safetensors(self):
        pass


class TestMagiTextConditioningTorchCompile(MagiTextConditioningTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(1, 2), (2, 2), (3, 2)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((height, 8, 16), generator=self.generator, device=torch_device),
            "attention_mask": torch.ones(height, 8, device=torch_device, dtype=torch.bool),
            "num_chunks": width,
        }
