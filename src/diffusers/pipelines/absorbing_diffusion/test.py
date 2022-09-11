from diffusers.models.vae import Encoder


# create appropriate VQ-VAE encoder for Absorbing Diffusion
encoder = Encoder(
    in_channels=3,
    out_channels=256,
    down_block_types=(
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "AttnDownEncoderBlock2D",
    ),
    block_out_channels=(128, 128, 256, 256, 512),
    layers_per_block=2,
    act_fn="swish",
    double_z=False,
)

for name, param in encoder.named_parameters():
    print(name, param.shape)
