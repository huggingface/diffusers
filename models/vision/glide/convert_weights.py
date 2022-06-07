import argparse

import torch
from torch import nn

from transformers import CLIPTextConfig, GPT2Tokenizer
from modelling_text_encoder import CLIPTextModel

# wget https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt
state_dict = torch.load("base.pt", map_location="cpu")
state_dict = {k: nn.Parameter(v) for k, v in state_dict.items()}
config = CLIPTextConfig(
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=16,
    num_attention_heads=8,
    max_position_embeddings=128,
    use_padding_embeddings=True,
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

inputs = tokenizer(["an oil painting of a corgi", ""], padding="max_length", max_length=128, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

model.save_pretrained("./glide-base")