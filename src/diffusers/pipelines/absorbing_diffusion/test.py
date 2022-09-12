from diffusers import VQModel


# create appropriate VQ-VAE model for Absorbing Diffusion
model = VQModel(
    in_channels=3,
    out_channels=3,
    down_block_types=(
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "AttnDownEncoderBlock2D",
    ),
    up_block_types=(
        "AttnUpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ),
    block_out_channels=(128, 128, 256, 256, 512),
    layers_per_block=2,
    act_fn="swish",
    final_encoder_activation=False,
    latent_channels=256,
)

for name, param in model.named_parameters():
    print(name, param.shape)
