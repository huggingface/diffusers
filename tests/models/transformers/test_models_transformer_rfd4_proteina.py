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

from diffusers import RFD4ProteinaTransformer
from diffusers.models.transformers.transformer_rfd4_proteina import CONDITION_KEYS, FEATURE_FUNCTIONS
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class RFD4ProteinaTransformerTesterConfig(BaseModelTesterConfig):
    # The shared tests read the batch size from this input, so it has to be a tensor, not a mapping.
    main_input_name = "padding_mask"

    @property
    def model_class(self):
        return RFD4ProteinaTransformer

    def get_init_dict(self):
        return {
            "nlayers": 1,
            "token_dim": 8,
            "pair_repr_dim": 4,
            "nheads": 2,
            "feats_seq": ["xt_bb_ca", "xt_local_latents"],
            "feats_cond_seq": [],
            "feats_pair_repr": [],
            "feats_pair_cond": [],
            "dim_cond": 8,
            "latent_dim": 4,
            "update_pair_repr": False,
            "use_tri_mult": False,
            "recycle": False,
        }

    def get_dummy_inputs(self):
        generator = torch.Generator("cpu").manual_seed(0)
        padding_mask = torch.tensor([[True, True, False]], device=torch_device)
        return {
            "bb_ca": randn_tensor((1, 3, 3), generator=generator, device=torch_device),
            "local_latents": randn_tensor((1, 3, 4), generator=generator, device=torch_device),
            "bb_ca_time": torch.tensor([0.5], device=torch_device),
            "local_latents_time": torch.tensor([0.5], device=torch_device),
            "padding_mask": padding_mask,
            "conditioning": {
                "is_in_context_conditioning": torch.tensor([[False, True, False]], device=torch_device),
                "supports_local_latent": padding_mask,
            },
            "recycle_depth": 1,
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 3, 3)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 3)


class TestRFD4ProteinaTransformerModel(RFD4ProteinaTransformerTesterConfig, ModelTesterMixin):
    def test_transformer_executes_requested_recycle_depth(self):
        init_dict = {**self.get_init_dict(), "recycle": True, "feats_cond_seq": ["recycle_remaining"], "t_emb_dim": 8}
        transformer = RFD4ProteinaTransformer(**init_dict).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        cycles = []
        handle = transformer.transformer_layers[0].register_forward_hook(lambda *_: cycles.append(None))
        for depth, expected_cycles in ((0, 1), (1, 1), (3, 3)):
            cycles.clear()
            transformer(**{**inputs, "recycle_depth": depth})
            assert len(cycles) == expected_cycles
        handle.remove()

    def test_condition_features_follow_the_annotation_semantics(self):
        atom_mask = torch.ones(1, 3, 37, dtype=torch.bool)
        atom_mask[:, :, 3] = False
        features = {
            "padding_mask": torch.tensor([[True, True, False]]),
            "atom37_mask": atom_mask,
            CONDITION_KEYS["internal_coordinates"]: torch.full((1, 3, 37, 3), 10.0),
            CONDITION_KEYS["coordinate_group"]: torch.tensor(
                [[[-1, 0, 7] + [-1] * 34, [-1, 0, 9] + [-1] * 34, [-1] * 37]]
            ),
            CONDITION_KEYS["rasa"]: torch.full((1, 3, 37), 3),
            CONDITION_KEYS["centroid"]: torch.tensor([[[10.0, 20.0, 30.0], [float("nan")] * 3, [float("inf")] * 3]]),
        }

        internal = FEATURE_FUNCTIONS["cond_a37coors_internal_nm"](features, 148)
        assert internal[..., 111:].sum().item() == 2
        assert internal[..., :111].sum().item() == 6
        rasa = FEATURE_FUNCTIONS["rasa_binned"](features, 37)
        assert torch.equal(rasa[~atom_mask], torch.zeros_like(rasa[~atom_mask]))
        centroid = FEATURE_FUNCTIONS["chain_centroid_xyz"](features, 3)
        assert torch.equal(centroid, torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))


class TestRFD4ProteinaTransformerMemory(RFD4ProteinaTransformerTesterConfig, MemoryTesterMixin):
    def get_init_dict(self):
        # The layerwise-casting memory test needs layers large enough for the compute dtype to change peak memory. Three
        # layers with the pair update after every second one keep a layer without an update, as the released model has.
        return {
            **super().get_init_dict(),
            "nlayers": 3,
            "update_pair_repr": True,
            "token_dim": 512,
            "nheads": 8,
            "pair_repr_dim": 64,
            "dim_cond": 64,
        }


class TestRFD4ProteinaTransformerTraining(RFD4ProteinaTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        # The shared default expects a `*Model` class name; this transformer is named `RFD4ProteinaTransformer`.
        super().test_gradient_checkpointing_is_applied(expected_set={"RFD4ProteinaTransformer"})


class TestRFD4ProteinaTransformerLoRA(RFD4ProteinaTransformerTesterConfig, LoraTesterMixin):
    pass
