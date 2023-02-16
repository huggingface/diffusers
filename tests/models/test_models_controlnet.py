import torch

from diffusers import UNet2DConditionModel


################################################################################
# PoC version
################################################################################


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
    "controlnet_hint_channels": 3,
    "upcast_attention": False,
}
model_id_sd15_canny = "takuma104/control_sd15_canny"  # currntry this is private model


## utils #######################################################################


def controlnet_inference(model_id=None):
    sample = torch.randn((1, 4, 64, 64)).cuda()
    hint = torch.randn((1, 3, 512, 512)).cuda()
    timestep = 0
    encoder_hidden_states = torch.randn((1, 77, 768)).cuda()
    if model_id is None:
        model = UNet2DConditionModel(**ctrlnet_config).cuda()
    else:
        model = UNet2DConditionModel.from_pretrained(model_id, subfolder="controlnet").cuda()
    model.eval()
    with torch.no_grad():
        outputs = model(
            sample=sample, controlnet_hint=hint, timestep=timestep, encoder_hidden_states=encoder_hidden_states
        )
    return outputs


def controlled_unet_inference(control, model_id=None):
    sample = torch.randn((1, 4, 64, 64)).cuda()
    timestep = 0
    encoder_hidden_states = torch.randn((1, 77, 768)).cuda()
    if model_id is None:
        model = UNet2DConditionModel(**unet_config).cuda()
    else:
        model = UNet2DConditionModel.from_pretrained(model_id_sd15_canny, subfolder="unet").cuda()
    model.eval()
    with torch.no_grad():
        out = model(sample=sample, control=control, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
    return out


## tests #######################################################################


def test_unet_inference():
    sample = torch.randn((1, 4, 64, 64)).cuda()
    timestep = 0
    encoder_hidden_states = torch.randn((1, 77, 768)).cuda()
    model = UNet2DConditionModel(**unet_config).cuda()
    model.eval()
    with torch.no_grad():
        out = model(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
    assert out.sample.shape == (1, 4, 64, 64)
    print(out.sample)


def test_controlnet_inference():
    outputs = controlnet_inference()
    assert len(outputs) == 12 + 1  # 12 layer down and one middle
    print(outputs)


def test_controlled_unet_inference():
    control = controlnet_inference()
    out = controlled_unet_inference(control=control)
    assert out.sample.shape == (1, 4, 64, 64)
    print(out.sample)


def test_controlnet_from_pretrained_and_inference():
    outputs = controlnet_inference(model_id=model_id_sd15_canny)
    assert len(outputs) == 12 + 1  # 12 layer down and one middle
    print(outputs)


def test_controlled_unet_from_pretrained_and_inference():
    control = controlnet_inference()
    out = controlled_unet_inference(control=control, model_id=model_id_sd15_canny)
    assert out.sample.shape == (1, 4, 64, 64)
    print(out.sample)
