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

from unittest import mock

import torch
import torch.nn.functional as F

from diffusers import LuminaNextDiT2DModel
from diffusers.models import attention_processor
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class LuminaNextDiTTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LuminaNextDiT2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (4, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 16,
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 24,
            "num_layers": 2,
            "num_attention_heads": 3,
            "num_kv_heads": 1,
            "multiple_of": 16,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "learn_sigma": False,
            "qk_norm": True,
            "cross_attention_dim": 32,
            "scaling_factor": 1.0,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = 16
        embedding_dim = 32
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.rand(size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_mask": randn_tensor((batch_size, sequence_length), generator=self.generator, device=torch_device),
            "image_rotary_emb": randn_tensor((384, 384, 4), generator=self.generator, device=torch_device),
            "cross_attention_kwargs": {},
        }


class TestLuminaNextDiT(LuminaNextDiTTesterConfig, ModelTesterMixin):
    pass


class TestLuminaNextDiTTraining(LuminaNextDiTTesterConfig, TrainingTesterMixin):
    pass


class TestLuminaNextDiTBehavior(LuminaNextDiTTesterConfig):
    def _model(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        with torch.no_grad():
            for layer in model.layers:
                layer.gate.uniform_(0.5, 1.5)  # zero-initialised, would hide the cross-attention (and its mask)
        return model

    def _inputs(self):
        inputs = self.get_dummy_inputs()
        # padded prompt: the second sample has only 5 valid text tokens
        mask = torch.ones(2, 16, device=torch_device)
        mask[1, 5:] = 0
        inputs["encoder_mask"] = mask
        return inputs

    def test_attention_mask_broadcast_matches_expanded_mask(self):
        """Passing the (batch, 1, 1, kv_len) mask to SDPA equals expanding it to (batch, heads, q_len, kv_len)."""
        model = self._model()
        inputs = self._inputs()
        sdpa = F.scaled_dot_product_attention

        def sdpa_with_expanded_mask(query, key, value, attn_mask=None, **kwargs):
            if attn_mask is not None:
                attn_mask = attn_mask.expand(-1, query.shape[1], query.shape[2], -1)
            return sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

        perturbed = {**inputs, "encoder_hidden_states": inputs["encoder_hidden_states"].clone()}
        perturbed["encoder_hidden_states"][1, 5:] += 10

        with torch.no_grad():
            out = model(**inputs).sample
            out_perturbed = model(**perturbed).sample
            with mock.patch.object(attention_processor.F, "scaled_dot_product_attention", sdpa_with_expanded_mask):
                ref = model(**inputs).sample
        assert torch.equal(out, ref)
        # the mask reaches SDPA: the padded text tokens of the second sample do not change its output
        assert torch.equal(out_perturbed[1], out[1])

    def test_rotary_table_on_token_grid_matches_384_table(self):
        """The RoPE table built on the latent token grid gives the same output as the sliced 384 x 384 table."""
        model = self._model()
        inputs = self._inputs()
        tokens = inputs["hidden_states"].shape[-1] // self.get_init_dict()["patch_size"]
        for linear_factor, ntk_factor in ((2.0, 1.0), (1.0, 2.0), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)):
            kwargs = {"linear_factor": linear_factor, "ntk_factor": ntk_factor}
            full = get_2d_rotary_pos_embed_lumina(model.head_dim, 384, 384, **kwargs)
            grid = get_2d_rotary_pos_embed_lumina(model.head_dim, tokens, tokens, **kwargs)
            assert torch.equal(grid, full[:tokens, :tokens])
            # non-square grid: the pipeline passes height_tokens and width_tokens separately
            rect = get_2d_rotary_pos_embed_lumina(model.head_dim, tokens - 2, tokens, **kwargs)
            assert torch.equal(rect, full[: tokens - 2, :tokens])
            with torch.no_grad():
                ref = model(**{**inputs, "image_rotary_emb": full}).sample
                out = model(**{**inputs, "image_rotary_emb": grid.to(torch_device)}).sample
            assert torch.equal(out, ref)
