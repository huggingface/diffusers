# coding=utf-8

import torch

from diffusers import JoyAIImageTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import AttentionTesterMixin, BaseModelTesterConfig, ModelTesterMixin


enable_full_determinism()


class JoyAIImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return JoyAIImageTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (4, 2, 4, 4)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | float | tuple[int, int, int] | str]:
        return {
            "patch_size": (1, 2, 2),
            "in_channels": 4,
            "out_channels": 4,
            "hidden_size": 32,
            "heads_num": 4,
            "text_states_dim": 24,
            "mlp_width_ratio": 2.0,
            "mm_double_blocks_depth": 2,
            "rope_dim_list": (2, 2, 4),
            "rope_type": "rope",
            "attn_backend": "torch_spda",
            "theta": 1000,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        hidden_states = randn_tensor((batch_size, 4, 2, 4, 4), generator=self.generator, device=torch_device)
        timestep = torch.tensor([1.0] * batch_size, device=torch_device)
        encoder_hidden_states = randn_tensor((batch_size, 6, 24), generator=self.generator, device=torch_device)
        encoder_hidden_states_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], device=torch_device, dtype=torch.long
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
        }


class TestJoyAIImageTransformer(JoyAIImageTransformerTesterConfig, ModelTesterMixin):
    pass


class TestJoyAIImageTransformerAttention(JoyAIImageTransformerTesterConfig, AttentionTesterMixin):
    def test_exposes_attention_processors(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device)

        assert hasattr(model, "attn_processors")
        assert len(model.attn_processors) == len(model.double_blocks)
