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

from diffusers import RFD4ProteinaAutoencoder
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class RFD4ProteinaAutoencoderTesterConfig(BaseModelTesterConfig):
    main_input_name = "z_latent"

    @property
    def model_class(self):
        return RFD4ProteinaAutoencoder

    def get_init_dict(self):
        return {
            "atom37_mask": [[True] * 37 for _ in range(33)],
            "allowed_residue_types": [[True] * 33 for _ in range(5)],
            "latent_dim": 4,
            "n_residue_types": 33,
            "num_token_types": 5,
            "nlayers": 1,
            "token_dim": 8,
            "nheads": 2,
            "dim_cond": 8,
            "pair_repr_dim": 4,
            "encoder_feats_seq": [],
            "decoder_feats_seq": ["z_latent_seq"],
        }

    def get_dummy_inputs(self):
        generator = torch.Generator("cpu").manual_seed(0)
        padding_mask = torch.tensor([[True, True, False]], device=torch_device)
        return {
            "z_latent": randn_tensor((1, 3, 4), generator=generator, device=torch_device),
            "ca_coors_nm": randn_tensor((1, 3, 3), generator=generator, device=torch_device),
            "padding_mask": padding_mask,
            "atom37_mask": torch.ones(1, 3, 37, dtype=torch.bool, device=torch_device),
            "conditioning": {
                "is_in_context_conditioning": torch.tensor([[False, True, False]], device=torch_device),
                "supports_local_latent": padding_mask,
                "condition_sequence_1_residue": torch.zeros(1, 3, dtype=torch.long, device=torch_device),
            },
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 3, 4)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 37, 3)


class TestRFD4ProteinaAutoencoderModel(RFD4ProteinaAutoencoderTesterConfig, ModelTesterMixin):
    def test_decode_keeps_the_conditioned_residue_and_its_coordinate_group_atoms(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        conditioning = inputs["conditioning"]
        conditioning["is_in_context_conditioning"] = torch.zeros(1, 3, dtype=torch.bool, device=torch_device)
        conditioning["condition_sequence_1_residue"] = torch.tensor([[5, 0, 0]], device=torch_device)
        groups = torch.full((1, 3, 37), -1, device=torch_device)
        groups[0, 0, 10] = 0
        conditioning["condition_CoordinateGroup_1_atom"] = groups
        with torch.no_grad():
            output = model.decode(**inputs)
        assert output.residue_type[0, 0] == 5
        assert output.atom_mask[0, 0, 10]


class TestRFD4ProteinaAutoencoderMemory(RFD4ProteinaAutoencoderTesterConfig, MemoryTesterMixin):
    def get_init_dict(self):
        # The layerwise-casting memory test needs layers large enough for the compute dtype to change peak memory.
        return {
            **super().get_init_dict(),
            "nlayers": 2,
            "token_dim": 512,
            "nheads": 8,
            "pair_repr_dim": 64,
            "dim_cond": 64,
        }


class TestRFD4ProteinaAutoencoderAttention(RFD4ProteinaAutoencoderTesterConfig, AttentionTesterMixin):
    pass


class TestRFD4ProteinaAutoencoderCompile(RFD4ProteinaAutoencoderTesterConfig, TorchCompileTesterMixin):
    pass
