# integration test (hunyuan dit)
import torch
from diffusers import HunyuanDiTPipeline

import torch
from huggingface_hub import hf_hub_download
from diffusers import HunyuanDiT2DModel
import safetensors.torch


device = "cuda"
model_config = HunyuanDiT2DModel.load_config("XCLiu/HunyuanDiT-0523", subfolder="transformer")
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

#model.load_state_dict(state_dict)

#from transformers import BertModel
#bert_model = BertModel.from_pretrained("XCLiu/HunyuanDiT-0523", add_pooling_layer=False, subfolder="text_encoder")

#pipe = HunyuanDiTPipeline.from_pretrained("XCLiu/HunyuanDiT-0523", text_encoder=bert_model, transformer=model, torch_dtype=torch.float32)
pipe = HunyuanDiTPipeline.from_pretrained("../HunyuanDiT-ckpt")
pipe.to('cuda')
#pipe.to(dtype=torch.float16)

#pipe.save_pretrained("../HunyuanDiT-ckpt")

### NOTE: HunyuanDiT supports both Chinese and English inputs
prompt = "一个宇航员在骑马"
#prompt = "An astronaut riding a horse"
generator=torch.Generator(device="cuda").manual_seed(0)
image = pipe(height=1024, width=1024, prompt=prompt, generator=generator).images[0]

image.save("img.png")