from diffusers.models.transformers.smol_dit_transformer_2d import SmolDiT2DModel 
from diffusers.models.embeddings import get_2d_rotary_pos_embed
import torch

# taken from Hunyuan
def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

init_dict = {
    "sample_size": 16,
    "num_layers": 2,
    "patch_size": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "in_channels": 4,
    "cross_attention_dim": 32,
    "out_channels": 4,
    "activation_fn": "gelu-approximate",
}
model = SmolDiT2DModel(**init_dict)
assert model

height = width = 16

hidden_states = torch.randn((1, 4, height, width))
timesteps = torch.randint(0, 1000, size=(1,))
encoder_hidden_states = torch.randn((1, 8, 32))

grid_height = height // 8 // model.config.patch_size
grid_width = width // 8 // model.config.patch_size
base_size = 512 // 8 // model.config.patch_size
grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)

inputs = {
    "hidden_states": hidden_states,
    "timestep": timesteps,
    "encoder_hidden_states": encoder_hidden_states,
    "image_rotary_emb": get_2d_rotary_pos_embed(
        model.inner_dim // model.config.num_attention_heads, grid_crops_coords, (grid_height, grid_width)
    )
}
print(inputs["image_rotary_emb"][0].shape)

with torch.no_grad():
    out = model(**inputs).sample
    print(out.shape)