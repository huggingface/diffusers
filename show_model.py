import torch
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from sgm.models.autoencoder import AutoencodingEngine

# (1) create vae_sat
# AutoencodingEngine initialization arguments:
encoder_config={'target': 'sgm.modules.diffusionmodules.model.Encoder', 'params': {'attn_type': 'vanilla', 'double_z': True, 'z_channels': 16, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 4, 8, 8], 'num_res_blocks': 3, 'attn_resolutions': [], 'mid_attn': False, 'dropout': 0.0}}
decoder_config={'target': 'sgm.modules.diffusionmodules.model.Decoder', 'params': {'attn_type': 'vanilla', 'double_z': True, 'z_channels': 16, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 4, 8, 8], 'num_res_blocks': 3, 'attn_resolutions': [], 'mid_attn': False, 'dropout': 0.0}}
loss_config={'target': 'torch.nn.Identity'}
regularizer_config={'target': 'sgm.modules.autoencoding.regularizers.DiagonalGaussianRegularizer'}
optimizer_config=None
lr_g_factor=1.0
ckpt_path="/raid/.cache/huggingface/models--ZP2HF--CogView3-SAT/snapshots/ca86ce9ba94f9a7f2dd109e7a59e4c8ad04121be/3plus_ae/imagekl_ch16.pt"
ignore_keys= []
kwargs = {"monitor": "val/rec_loss"}
vae_sat = AutoencodingEngine(
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    loss_config=loss_config,
    regularizer_config=regularizer_config,
    optimizer_config=optimizer_config,
    lr_g_factor=lr_g_factor,
    ckpt_path=ckpt_path,
    ignore_keys=ignore_keys,
    **kwargs)



# (2) create vae (diffusers)
ckpt_path_vae_cogview3 = hf_hub_download(repo_id="ZP2HF/CogView3-SAT", subfolder="3plus_ae", filename="imagekl_ch16.pt")
cogview3_ckpt = torch.load(ckpt_path_vae_cogview3, map_location='cpu')["state_dict"]

in_channels = 3  # Inferred from encoder.conv_in.weight shape
out_channels = 3  # Inferred from decoder.conv_out.weight shape
down_block_types = ("DownEncoderBlock2D",) * 4  # Inferred from the presence of 4 encoder.down blocks
up_block_types = ("UpDecoderBlock2D",) * 4  # Inferred from the presence of 4 decoder.up blocks
block_out_channels = (128, 512, 1024, 1024)  # Inferred from the channel sizes in encoder.down blocks
layers_per_block = 3  # Inferred from the number of blocks in each encoder.down and decoder.up
act_fn = "silu" # This is the default, cannot be inferred from state_dict
latent_channels = 16  # Inferred from decoder.conv_in.weight shape
norm_num_groups = 32  # This is the default, cannot be inferred from state_dict
sample_size = 1024  # This is the default, cannot be inferred from state_dict
scaling_factor = 0.18215  # This is the default, cannot be inferred from state_dict
force_upcast = True  # This is the default, cannot be inferred from state_dict
use_quant_conv = False  # Inferred from the presence of encoder.conv_out
use_post_quant_conv = False  # Inferred from the presence of decoder.conv_in
mid_block_add_attention = False  # Inferred from the absence of attention layers in mid blocks

vae = AutoencoderKL(
    in_channels=in_channels,
    out_channels=out_channels,
    down_block_types=down_block_types,
    up_block_types=up_block_types,
    block_out_channels=block_out_channels,
    layers_per_block=layers_per_block,
    act_fn=act_fn,
    latent_channels=latent_channels,
    norm_num_groups=norm_num_groups,
    sample_size=sample_size,
    scaling_factor=scaling_factor,
    force_upcast=force_upcast,
    use_quant_conv=use_quant_conv,
    use_post_quant_conv=use_post_quant_conv,
    mid_block_add_attention=mid_block_add_attention,
)

vae.eval()
vae_sat.eval()

converted_vae_state_dict = convert_ldm_vae_checkpoint(cogview3_ckpt, vae.config)
vae.load_state_dict(converted_vae_state_dict, strict=False)

# (3) run forward pass for both models

# [2, 16, 128, 128] -> [2, 3, 1024, 1024
z = torch.load("z.pt").float().to("cpu")

with torch.no_grad():
    print(" ")
    print(f" running forward pass for diffusers vae")
    out = vae.decode(z).sample
    print(f" ")
    print(f" running forward pass for sgm vae")
    out_sat = vae_sat.decode(z)

print(f" output shape: {out.shape}")
print(f" expected output shape: {out_sat.shape}")
assert out.shape == out_sat.shape
assert (out - out_sat).abs().max() < 1e-4, f"max diff: {(out - out_sat).abs().max()}"