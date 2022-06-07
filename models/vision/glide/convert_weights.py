import argparse

import torch
from torch import nn

from transformers import CLIPTextConfig, CLIPTextModel, GPT2Tokenizer

# wget https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt
state_dict = torch.load("base.pt", map_location="cpu")
state_dict = {k: nn.Parameter(v) for k, v in state_dict.items()}
config = CLIPTextConfig(
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=16,
    num_attention_heads=8,
    max_position_embeddings=128
)
model = CLIPTextModel(config).eval()
tokenizer = GPT2Tokenizer("./glide-base/vocab.json", "./glide-base/merges.txt", pad_token="<|endoftext|>")
tokenizer.save_pretrained("./glide-base")

hf_encoder = model.text_model

hf_encoder.embeddings.token_embedding.weight = state_dict["token_embedding.weight"]
hf_encoder.embeddings.position_embedding.weight.data = state_dict["positional_embedding"]
hf_encoder.embeddings.padding_embedding.weight.data = state_dict["padding_embedding"]

hf_encoder.final_layer_norm.weight = state_dict["final_ln.weight"]
hf_encoder.final_layer_norm.bias = state_dict["final_ln.bias"]

for layer_idx in range(config.num_hidden_layers):
    hf_layer = hf_encoder.encoder.layers[layer_idx]
    q_proj, k_proj, v_proj = state_dict[f"transformer.resblocks.{layer_idx}.attn.c_qkv.weight"].chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = state_dict[f"transformer.resblocks.{layer_idx}.attn.c_qkv.bias"].chunk(3, dim=0)

    hf_layer.self_attn.q_proj.weight.data = q_proj
    hf_layer.self_attn.q_proj.bias.data = q_proj_bias
    hf_layer.self_attn.k_proj.weight.data = k_proj
    hf_layer.self_attn.k_proj.bias.data = k_proj_bias
    hf_layer.self_attn.v_proj.weight.data = v_proj
    hf_layer.self_attn.v_proj.bias.data = v_proj_bias

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

inputs = tokenizer(["an oil painting of a corgi", ""], padding="max_length", max_length=128, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

model.save_pretrained("./glide-base")