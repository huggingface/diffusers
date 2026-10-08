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

from diffusers import BriaFibo2Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class BriaFibo2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return BriaFibo2Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 8, 8)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 8, 8)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | tuple[int, ...]]:
        return {
            "in_channels": 4,
            "patch_size": 2,
            "dim": 32,
            "n_layers": 2,
            "n_refiner_layers": 1,
            "n_heads": 2,
            "cap_feat_dim": 24,
            "injection_layer_ids": (1,),
            "perceiver_num_layers": 1,
            "min_num_gist_tokens": 4,
            "max_num_gist_tokens": 8,
            "gist_step": 4,
            "gist_min_text_len": 4,
            "gist_max_text_len": 16,
            "axes_dims": (4, 6, 6),
            "axes_lens": (16, 16, 16),
        }

    def get_dummy_inputs(self, height: int = 8, width: int = 8) -> dict[str, torch.Tensor]:
        batch_size = 1
        text_len = 6
        # one text bundle for the Perceiver, plus one per injection block
        num_bundles = 1 + len(self.get_init_dict()["injection_layer_ids"])

        return {
            "hidden_states": randn_tensor(
                (batch_size, 4, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([0.5], device=torch_device).expand(batch_size),
            "encoder_hidden_states": randn_tensor(
                (batch_size, num_bundles, text_len, 24), generator=self.generator, device=torch_device
            ),
        }


class TestBriaFibo2TransformerModel(BriaFibo2TransformerTesterConfig, ModelTesterMixin):
    def test_context_latents(self):
        # Images to edit, each of its own size, condition the output without being predicted
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        context_latents = [
            randn_tensor((1, 4, 6, 10), generator=self.generator, device=torch_device),
            randn_tensor((1, 4, 8, 8), generator=self.generator, device=torch_device),
        ]
        with torch.no_grad():
            output = model(**inputs).sample
            output_with_context = model(**inputs, context_latents=context_latents).sample

        assert output_with_context.shape == output.shape
        assert not torch.allclose(output_with_context, output)

    def test_padded_batch_matches_single_prompts(self):
        # Prompts of 4 and 16 text tokens get 4 and 8 gist tokens, so the batch pads the first prompt's gist. The
        # padding has to end every row of the attention mask, which varlen attention backends assume
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        hidden_states = randn_tensor((2, 4, 8, 8), generator=self.generator, device=torch_device)
        timestep = torch.tensor([0.5, 0.5], device=torch_device)
        encoder_hidden_states = randn_tensor((2, 2, 16, 24), generator=self.generator, device=torch_device)
        encoder_attention_mask = torch.ones(2, 16, dtype=torch.bool, device=torch_device)
        encoder_attention_mask[0, 4:] = False
        text_lengths = (4, 16)

        for context_latents in (None, [randn_tensor((2, 4, 6, 10), generator=self.generator, device=torch_device)]):
            attention_masks = []
            hook = model.layers[0].register_forward_pre_hook(lambda module, args: attention_masks.append(args[1]))
            with torch.no_grad():
                batch = model(
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    context_latents=context_latents,
                ).sample
            hook.remove()
            with torch.no_grad():
                single = [
                    model(
                        hidden_states[i : i + 1],
                        timestep[i : i + 1],
                        encoder_hidden_states[i : i + 1, :, : text_lengths[i]],
                        context_latents=None if context_latents is None else [c[i : i + 1] for c in context_latents],
                    ).sample
                    for i in range(2)
                ]

            attention_mask = attention_masks[0]
            assert not attention_mask.all()
            assert torch.equal(attention_mask, attention_mask.long().cumprod(dim=1).bool())
            assert torch.allclose(batch, torch.cat(single), atol=1e-5, rtol=1e-5)


class TestBriaFibo2TransformerMemory(BriaFibo2TransformerTesterConfig, MemoryTesterMixin):
    pass


class TestBriaFibo2TransformerTorchCompile(BriaFibo2TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    @pytest.mark.skip(
        "RopeEmbedder, copied from Z-Image, builds its frequency table under a `torch.device` context on first use, "
        "which Dynamo cannot trace in a full graph. Z-Image skips this test too."
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        pass

    def test_torch_compile_repeated_blocks(self):
        # BriaFibo2TransformerBlock runs three ways: in the noise refiner (one timestep for every token), in the main
        # blocks (per-token timesteps) and in the injection blocks (plus cross-attention), so it compiles three times
        super().test_torch_compile_repeated_blocks(recompile_limit=3)

    @pytest.mark.skip(
        "RopeEmbedder, copied from Z-Image, cannot be traced in a full graph (see above). Z-Image skips this test too."
    )
    def test_compile_on_different_shapes(self):
        pass


class TestBriaFibo2TransformerTraining(BriaFibo2TransformerTesterConfig, TrainingTesterMixin):
    pass


class TestBriaFibo2TransformerAttention(BriaFibo2TransformerTesterConfig, AttentionTesterMixin):
    pass
