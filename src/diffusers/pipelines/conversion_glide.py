import torch
from torch import nn

from diffusers import ClassifierFreeGuidanceScheduler, DDIMScheduler, GLIDESuperResUNetModel, GLIDETextToImageUNetModel
from diffusers.pipelines.pipeline_glide import GLIDE, CLIPTextModel
from transformers import CLIPTextConfig, GPT2Tokenizer


# wget https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt
state_dict = torch.load("base.pt", map_location="cpu")
state_dict = {k: nn.Parameter(v) for k, v in state_dict.items()}

### Convert the text encoder

config = CLIPTextConfig(
    vocab_size=50257,
    max_position_embeddings=128,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=16,
    num_attention_heads=8,
    use_padding_embeddings=True,
)
model = CLIPTextModel(config).eval()
tokenizer = GPT2Tokenizer(
    "./glide-base/tokenizer/vocab.json", "./glide-base/tokenizer/merges.txt", pad_token="<|endoftext|>"
)

hf_encoder = model.text_model

hf_encoder.embeddings.token_embedding.weight = state_dict["token_embedding.weight"]
hf_encoder.embeddings.position_embedding.weight.data = state_dict["positional_embedding"]
hf_encoder.embeddings.padding_embedding.weight.data = state_dict["padding_embedding"]

hf_encoder.final_layer_norm.weight = state_dict["final_ln.weight"]
hf_encoder.final_layer_norm.bias = state_dict["final_ln.bias"]

for layer_idx in range(config.num_hidden_layers):
    hf_layer = hf_encoder.encoder.layers[layer_idx]
    hf_layer.self_attn.qkv_proj.weight = state_dict[f"transformer.resblocks.{layer_idx}.attn.c_qkv.weight"]
    hf_layer.self_attn.qkv_proj.bias = state_dict[f"transformer.resblocks.{layer_idx}.attn.c_qkv.bias"]

    hf_layer.self_attn.out_proj.weight = state_dict[f"transformer.resblocks.{layer_idx}.attn.c_proj.weight"]
    hf_layer.self_attn.out_proj.bias = state_dict[f"transformer.resblocks.{layer_idx}.attn.c_proj.bias"]

    hf_layer.layer_norm1.weight = state_dict[f"transformer.resblocks.{layer_idx}.ln_1.weight"]
    hf_layer.layer_norm1.bias = state_dict[f"transformer.resblocks.{layer_idx}.ln_1.bias"]
    hf_layer.layer_norm2.weight = state_dict[f"transformer.resblocks.{layer_idx}.ln_2.weight"]
    hf_layer.layer_norm2.bias = state_dict[f"transformer.resblocks.{layer_idx}.ln_2.bias"]

    hf_layer.mlp.fc1.weight = state_dict[f"transformer.resblocks.{layer_idx}.mlp.c_fc.weight"]
    hf_layer.mlp.fc1.bias = state_dict[f"transformer.resblocks.{layer_idx}.mlp.c_fc.bias"]
    hf_layer.mlp.fc2.weight = state_dict[f"transformer.resblocks.{layer_idx}.mlp.c_proj.weight"]
    hf_layer.mlp.fc2.bias = state_dict[f"transformer.resblocks.{layer_idx}.mlp.c_proj.bias"]

### Convert the Text-to-Image UNet

text2im_model = GLIDETextToImageUNetModel(
    in_channels=3,
    model_channels=192,
    out_channels=6,
    num_res_blocks=3,
    attention_resolutions=(2, 4, 8),
    dropout=0.1,
    channel_mult=(1, 2, 3, 4),
    num_heads=1,
    num_head_channels=64,
    num_heads_upsample=1,
    use_scale_shift_norm=True,
    resblock_updown=True,
    transformer_dim=512,
)

text2im_model.load_state_dict(state_dict, strict=False)

text_scheduler = ClassifierFreeGuidanceScheduler(timesteps=1000, beta_schedule="squaredcos_cap_v2")

### Convert the Super-Resolution UNet

# wget https://openaipublic.blob.core.windows.net/diffusion/dec-2021/upsample.pt
ups_state_dict = torch.load("upsample.pt", map_location="cpu")

superres_model = GLIDESuperResUNetModel(
    in_channels=6,
    model_channels=192,
    out_channels=6,
    num_res_blocks=2,
    attention_resolutions=(8, 16, 32),
    dropout=0.1,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_heads=1,
    num_head_channels=64,
    num_heads_upsample=1,
    use_scale_shift_norm=True,
    resblock_updown=True,
)

superres_model.load_state_dict(ups_state_dict, strict=False)

upscale_scheduler = DDIMScheduler(timesteps=1000, beta_schedule="linear", beta_start=0.0001, beta_end=0.02, tensor_format="pt")

glide = GLIDE(
    text_unet=text2im_model,
    text_noise_scheduler=text_scheduler,
    text_encoder=model,
    tokenizer=tokenizer,
    upscale_unet=superres_model,
    upscale_noise_scheduler=upscale_scheduler,
)

glide.save_pretrained("./glide-base")
