import pytest
import torch

from diffusers import ControlNetModel, UNet2DConditionModel


# config from ControlNet_SD1.5
unet_config = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
    "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 768,
    "attention_head_dim": 8,
    "use_linear_projection": False,
    "upcast_attention": False,
}

ctrlnet_config = {
    "sample_size": 64,
    "in_channels": 4,
    "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 768,
    "attention_head_dim": 8,
    "use_linear_projection": False,
    "hint_channels": 3,
    "upcast_attention": False,
}

################################################################################
# Scaffold for WIP
# ##############################################################################


@pytest.mark.skip
def test_unet_inference_without_exception():
    sample = torch.randn((1, 4, 64, 64)).cuda()
    timestep = 0
    encoder_hidden_states = torch.randn((1, 77, 768)).cuda()
    model = UNet2DConditionModel(**unet_config).cuda()
    print(model(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states))


def test_inference_without_exception():
    sample = torch.randn((1, 4, 64, 64)).cuda()
    hint = torch.randn((1, 3, 512, 512)).cuda()
    timestep = 0
    encoder_hidden_states = torch.randn((1, 77, 768)).cuda()
    model = ControlNetModel(**ctrlnet_config).cuda()
    outputs = model(sample=sample, hint=hint, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
    assert len(outputs) == 12 + 1  # 12layer down and one middle
    print(outputs)
