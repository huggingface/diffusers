# integration test (hunyuan dit)
import torch
from diffusers import HunyuanDiTPipeline

import torch
from huggingface_hub import hf_hub_download
from diffusers import HunyuanDiT2DModel
import safetensors.torch

device = "cuda"
model_config = HunyuanDiT2DModel.load_config("XCLiu/HunyuanDiT-0523", subfolder="transformer")
# input_size -> sample_size, text_dim -> cross_attention_dim
model_config["sample_size"] = model_config.pop("input_size")[0]
model_config["cross_attention_dim"] = model_config.pop("text_dim")

model = HunyuanDiT2DModel.from_config(model_config).to(device)

ckpt_path = hf_hub_download(
    "XCLiu/HunyuanDiT-0523",
    filename ="diffusion_pytorch_model.safetensors",
    subfolder="transformer",
)
state_dict = safetensors.torch.load_file(ckpt_path)

num_layers = 40
for i in range(num_layers):

    # attn1
    # Wkqv -> to_q, to_k, to_v
    q, k, v = torch.chunk(state_dict[f"blocks.{i}.attn1.Wqkv.weight"], 3, dim=0)
    q_bias, k_bias, v_bias = torch.chunk(state_dict[f"blocks.{i}.attn1.Wqkv.bias"], 3, dim=0)
    state_dict[f"blocks.{i}.attn1.to_q.weight"] = q
    state_dict[f"blocks.{i}.attn1.to_q.bias"] = q_bias
    state_dict[f"blocks.{i}.attn1.to_k.weight"] = k
    state_dict[f"blocks.{i}.attn1.to_k.bias"] = k_bias
    state_dict[f"blocks.{i}.attn1.to_v.weight"] = v
    state_dict[f"blocks.{i}.attn1.to_v.bias"] = v_bias
    state_dict.pop(f"blocks.{i}.attn1.Wqkv.weight")
    state_dict.pop(f"blocks.{i}.attn1.Wqkv.bias")

    # q_norm, k_norm -> norm_q, norm_k
    state_dict[f"blocks.{i}.attn1.norm_q.weight"] = state_dict[f"blocks.{i}.attn1.q_norm.weight"]
    state_dict[f"blocks.{i}.attn1.norm_q.bias"] = state_dict[f"blocks.{i}.attn1.q_norm.bias"]
    state_dict[f"blocks.{i}.attn1.norm_k.weight"] = state_dict[f"blocks.{i}.attn1.k_norm.weight"]
    state_dict[f"blocks.{i}.attn1.norm_k.bias"] = state_dict[f"blocks.{i}.attn1.k_norm.bias"]

    state_dict.pop(f"blocks.{i}.attn1.q_norm.weight")
    state_dict.pop(f"blocks.{i}.attn1.q_norm.bias")
    state_dict.pop(f"blocks.{i}.attn1.k_norm.weight")
    state_dict.pop(f"blocks.{i}.attn1.k_norm.bias")

    # out_proj -> to_out
    state_dict[f"blocks.{i}.attn1.to_out.0.weight"] = state_dict[f"blocks.{i}.attn1.out_proj.weight"]
    state_dict[f"blocks.{i}.attn1.to_out.0.bias"] = state_dict[f"blocks.{i}.attn1.out_proj.bias"]
    state_dict.pop(f"blocks.{i}.attn1.out_proj.weight")
    state_dict.pop(f"blocks.{i}.attn1.out_proj.bias")

    # attn2
    # kq_proj -> to_k, to_v
    k, v = torch.chunk(state_dict[f"blocks.{i}.attn2.kv_proj.weight"], 2, dim=0)
    k_bias, v_bias = torch.chunk(state_dict[f"blocks.{i}.attn2.kv_proj.bias"], 2, dim=0)
    state_dict[f"blocks.{i}.attn2.to_k.weight"] = k
    state_dict[f"blocks.{i}.attn2.to_k.bias"] = k_bias
    state_dict[f"blocks.{i}.attn2.to_v.weight"] = v
    state_dict[f"blocks.{i}.attn2.to_v.bias"] = v_bias
    state_dict.pop(f"blocks.{i}.attn2.kv_proj.weight")
    state_dict.pop(f"blocks.{i}.attn2.kv_proj.bias")

    # q_proj -> to_q
    state_dict[f"blocks.{i}.attn2.to_q.weight"] = state_dict[f"blocks.{i}.attn2.q_proj.weight"]
    state_dict[f"blocks.{i}.attn2.to_q.bias"] = state_dict[f"blocks.{i}.attn2.q_proj.bias"]
    state_dict.pop(f"blocks.{i}.attn2.q_proj.weight")
    state_dict.pop(f"blocks.{i}.attn2.q_proj.bias")

    # q_norm, k_norm -> norm_q, norm_k
    state_dict[f"blocks.{i}.attn2.norm_q.weight"] = state_dict[f"blocks.{i}.attn2.q_norm.weight"]
    state_dict[f"blocks.{i}.attn2.norm_q.bias"] = state_dict[f"blocks.{i}.attn2.q_norm.bias"]
    state_dict[f"blocks.{i}.attn2.norm_k.weight"] = state_dict[f"blocks.{i}.attn2.k_norm.weight"]
    state_dict[f"blocks.{i}.attn2.norm_k.bias"] = state_dict[f"blocks.{i}.attn2.k_norm.bias"]

    state_dict.pop(f"blocks.{i}.attn2.q_norm.weight")
    state_dict.pop(f"blocks.{i}.attn2.q_norm.bias")
    state_dict.pop(f"blocks.{i}.attn2.k_norm.weight")
    state_dict.pop(f"blocks.{i}.attn2.k_norm.bias")

    # out_proj -> to_out
    state_dict[f"blocks.{i}.attn2.to_out.0.weight"] = state_dict[f"blocks.{i}.attn2.out_proj.weight"]
    state_dict[f"blocks.{i}.attn2.to_out.0.bias"] = state_dict[f"blocks.{i}.attn2.out_proj.bias"]
    state_dict.pop(f"blocks.{i}.attn2.out_proj.weight")
    state_dict.pop(f"blocks.{i}.attn2.out_proj.bias")

    # switch norm 2 and norm 3
    norm2_weight = state_dict[f"blocks.{i}.norm2.weight"]
    norm2_bias = state_dict[f"blocks.{i}.norm2.bias"]
    state_dict[f"blocks.{i}.norm2.weight"] = state_dict[f"blocks.{i}.norm3.weight"]
    state_dict[f"blocks.{i}.norm2.bias"] = state_dict[f"blocks.{i}.norm3.bias"]
    state_dict[f"blocks.{i}.norm3.weight"] = norm2_weight
    state_dict[f"blocks.{i}.norm3.bias"] = norm2_bias

    # norm1 -> norm1.norm
    # default_modulation.1 -> norm1.linear 
    state_dict[f"blocks.{i}.norm1.norm.weight"] = state_dict[f"blocks.{i}.norm1.weight"]
    state_dict[f"blocks.{i}.norm1.norm.bias"] = state_dict[f"blocks.{i}.norm1.bias"]
    state_dict[f"blocks.{i}.norm1.linear.weight"] = state_dict[f"blocks.{i}.default_modulation.1.weight"]
    state_dict[f"blocks.{i}.norm1.linear.bias"] = state_dict[f"blocks.{i}.default_modulation.1.bias"]
    state_dict.pop(f"blocks.{i}.norm1.weight")
    state_dict.pop(f"blocks.{i}.norm1.bias")
    state_dict.pop(f"blocks.{i}.default_modulation.1.weight")
    state_dict.pop(f"blocks.{i}.default_modulation.1.bias")

# t_embedder -> time_embedding (`TimestepEmbedding`)
state_dict["time_embedding.linear_1.bias"] = state_dict["t_embedder.mlp.0.bias"]
state_dict["time_embedding.linear_1.weight"] = state_dict["t_embedder.mlp.0.weight"]
state_dict["time_embedding.linear_2.bias"] = state_dict["t_embedder.mlp.2.bias"]
state_dict["time_embedding.linear_2.weight"] = state_dict["t_embedder.mlp.2.weight"]

state_dict.pop("t_embedder.mlp.0.bias")
state_dict.pop("t_embedder.mlp.0.weight")
state_dict.pop("t_embedder.mlp.2.bias")
state_dict.pop("t_embedder.mlp.2.weight")

# x_embedder -> pos_embd (`PatchEmbed`)
state_dict["pos_embed.proj.weight"] = state_dict["x_embedder.proj.weight"]
state_dict["pos_embed.proj.bias"] = state_dict["x_embedder.proj.bias"]
state_dict.pop("x_embedder.proj.weight")
state_dict.pop("x_embedder.proj.bias")

# mlp_t5 -> text_embedder
state_dict["text_embedder.linear_1.bias"] = state_dict["mlp_t5.0.bias"]
state_dict["text_embedder.linear_1.weight"] = state_dict["mlp_t5.0.weight"]
state_dict["text_embedder.linear_2.bias"] = state_dict["mlp_t5.2.bias"]
state_dict["text_embedder.linear_2.weight"] = state_dict["mlp_t5.2.weight"]
state_dict.pop("mlp_t5.0.bias")
state_dict.pop("mlp_t5.0.weight")
state_dict.pop("mlp_t5.2.bias")
state_dict.pop("mlp_t5.2.weight")

# extra_embedder -> extra_embedder
state_dict["extra_embedder.linear_1.bias"] = state_dict["extra_embedder.0.bias"]
state_dict["extra_embedder.linear_1.weight"] = state_dict["extra_embedder.0.weight"]
state_dict["extra_embedder.linear_2.bias"] = state_dict["extra_embedder.2.bias"]
state_dict["extra_embedder.linear_2.weight"] = state_dict["extra_embedder.2.weight"]
state_dict.pop("extra_embedder.0.bias")
state_dict.pop("extra_embedder.0.weight")
state_dict.pop("extra_embedder.2.bias")
state_dict.pop("extra_embedder.2.weight")

# model.final_adaLN_modulation.1 -> norm_out.linear
def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight
state_dict["norm_out.linear.weight"] = swap_scale_shift(state_dict["final_adaLN_modulation.1.weight"])
state_dict["norm_out.linear.bias"] = swap_scale_shift(state_dict["final_adaLN_modulation.1.bias"])
state_dict.pop("final_adaLN_modulation.1.weight")
state_dict.pop("final_adaLN_modulation.1.bias")

# final_linear -> proj_out
state_dict["proj_out.weight"] = state_dict["final_linear.weight"]
state_dict["proj_out.bias"] = state_dict["final_linear.bias"]
state_dict.pop("final_linear.weight")
state_dict.pop("final_linear.bias")

model.load_state_dict(state_dict)

from transformers import BertModel
bert_model = BertModel.from_pretrained("XCLiu/HunyuanDiT-0523", add_pooling_layer=True, subfolder="text_encoder")

pipe = HunyuanDiTPipeline.from_pretrained("XCLiu/HunyuanDiT-0523", text_encoder=bert_model, transformer=model, torch_dtype=torch.float32)
pipe.to('cuda')
pipe.to(dtype=torch.float16)

### NOTE: HunyuanDiT supports both Chinese and English inputs
prompt = "一个宇航员在骑马"
#prompt = "An astronaut riding a horse"
generator=torch.Generator(device="cuda").manual_seed(0)
image = pipe(height=1024, width=1024, prompt=prompt, generator=generator).images[0]

image.save("img.png")