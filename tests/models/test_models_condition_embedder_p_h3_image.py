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

import torch

from diffusers import PH3ImageTextConnector
from diffusers.utils.torch_utils import randn_tensor

from ..testing_utils import enable_full_determinism, torch_device
from .testing_utils import BaseModelTesterConfig


enable_full_determinism()


NUM_TOKENS = 5
TEXT_HIDDEN_SIZE = 8
# `output_hidden_states` returns the embedding output plus one entry per decoder layer.
NUM_HIDDEN_STATES = 5


class PH3ImageTextConnectorTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return PH3ImageTextConnector

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (NUM_TOKENS, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {"hidden_state_indices": (1, 3, 4), "text_hidden_size": TEXT_HIDDEN_SIZE, "out_channels": 16}

    def get_dummy_inputs(self) -> dict:
        return {
            "hidden_states": tuple(
                randn_tensor((2, NUM_TOKENS, TEXT_HIDDEN_SIZE), generator=self.generator, device=torch_device)
                for _ in range(NUM_HIDDEN_STATES)
            )
        }


# Like the other text connectors, the connector returns a tensor (no `return_dict`), so `ModelTesterMixin` does not
# apply.
class TestPH3ImageTextConnector(PH3ImageTextConnectorTesterConfig):
    """Model tests for the P-H3-Image text connector."""

    def test_output_shape(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            output = model(**self.get_dummy_inputs())
        assert output.shape == (2, *self.output_shape)

    def test_from_save_pretrained(self, tmp_path):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=self.generator))
        model.save_pretrained(tmp_path)
        new_model = self.model_class.from_pretrained(tmp_path).to(torch_device).eval()
        with torch.no_grad():
            torch.testing.assert_close(new_model(**self.get_dummy_inputs()), model(**self.get_dummy_inputs()))

    def test_reads_the_configured_hidden_states(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        hidden_states = list(self.get_dummy_inputs()["hidden_states"])
        with torch.no_grad():
            output = model(tuple(hidden_states))
            # Entries outside `hidden_state_indices` are not read.
            hidden_states[0] = torch.zeros_like(hidden_states[0])
            hidden_states[2] = torch.zeros_like(hidden_states[2])
            unchanged = model(tuple(hidden_states))
            hidden_states[3] = torch.zeros_like(hidden_states[3])
            changed = model(tuple(hidden_states))
        torch.testing.assert_close(output, unchanged)
        assert not torch.allclose(output[:, 1:], changed[:, 1:])

    def test_first_token_is_the_sink_embedding(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            model.sink_embedding.copy_(torch.randn(16, generator=self.generator))
            output = model(**self.get_dummy_inputs())
        torch.testing.assert_close(output[:, 0], model.sink_embedding.expand(2, -1))

    def test_stays_float32(self, tmp_path):
        self.model_class(**self.get_init_dict()).save_pretrained(tmp_path)
        model = self.model_class.from_pretrained(tmp_path, dtype=torch.bfloat16)
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
