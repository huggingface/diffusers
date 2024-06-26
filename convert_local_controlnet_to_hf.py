import torch
from huggingface_hub import hf_hub_download
from diffusers import HunyuanDiT2DControlNetModel
from diffusers import HunyuanDiTPipeline
import safetensors.torch

model_path = '/apdcephfs_cq8/share_1367250/xingchaoliu/ckpts/t2i/controlnet/pytorch_model_pose_distill.pt'

state_dict = torch.load(model_path, map_location='cpu')

device = "cuda"

'''
transformer = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers").transformer
HunyuanDiT2DControlNetModel.from_transformer(transformer)
'''

model_config = HunyuanDiT2DControlNetModel.load_config("Tencent-Hunyuan/HunyuanDiT-Diffusers", subfolder="transformer")
# input_size -> sample_size, text_dim -> cross_attention_dim

'''
i=0
for key in state_dict:
    if 'blocks.' not in key:
        print(i, 'local:', key)
        i+=1
'''
        
model = HunyuanDiT2DControlNetModel.from_config(model_config).to(device)

for key in model.state_dict():
    print('diffusers:', key)

num_layers = 19
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

    # mlp.fc1 -> ff.net.0, mlp.fc2 -> ff.net.2
    state_dict[f"blocks.{i}.ff.net.0.proj.weight"] = state_dict[f"blocks.{i}.mlp.fc1.weight"]
    state_dict[f"blocks.{i}.ff.net.0.proj.bias"] = state_dict[f"blocks.{i}.mlp.fc1.bias"]
    state_dict[f"blocks.{i}.ff.net.2.weight"] = state_dict[f"blocks.{i}.mlp.fc2.weight"]
    state_dict[f"blocks.{i}.ff.net.2.bias"] = state_dict[f"blocks.{i}.mlp.fc2.bias"]
    state_dict.pop(f"blocks.{i}.mlp.fc1.weight")
    state_dict.pop(f"blocks.{i}.mlp.fc1.bias")
    state_dict.pop(f"blocks.{i}.mlp.fc2.weight")
    state_dict.pop(f"blocks.{i}.mlp.fc2.bias")

    # after_proj_list -> controlnet_blocks
    state_dict[f"controlnet_blocks.{i}.weight"] = state_dict[f"after_proj_list.{i}.weight"]
    state_dict[f"controlnet_blocks.{i}.bias"] = state_dict[f"after_proj_list.{i}.bias"]
    state_dict.pop(f"after_proj_list.{i}.weight")
    state_dict.pop(f"after_proj_list.{i}.bias")

# before_proj -> input_block
state_dict["input_block.weight"] = state_dict["before_proj.weight"]
state_dict["input_block.bias"] = state_dict["before_proj.bias"]
state_dict.pop("before_proj.weight")
state_dict.pop("before_proj.bias")

# pooler -> time_extra_emb
state_dict["time_extra_emb.pooler.positional_embedding"] = state_dict["pooler.positional_embedding"]
state_dict["time_extra_emb.pooler.k_proj.weight"] = state_dict["pooler.k_proj.weight"]
state_dict["time_extra_emb.pooler.k_proj.bias"] = state_dict["pooler.k_proj.bias"]
state_dict["time_extra_emb.pooler.q_proj.weight"] = state_dict["pooler.q_proj.weight"]
state_dict["time_extra_emb.pooler.q_proj.bias"] = state_dict["pooler.q_proj.bias"]
state_dict["time_extra_emb.pooler.v_proj.weight"] = state_dict["pooler.v_proj.weight"]
state_dict["time_extra_emb.pooler.v_proj.bias"] = state_dict["pooler.v_proj.bias"]
state_dict["time_extra_emb.pooler.c_proj.weight"] = state_dict["pooler.c_proj.weight"]
state_dict["time_extra_emb.pooler.c_proj.bias"] = state_dict["pooler.c_proj.bias"]
state_dict.pop("pooler.k_proj.weight")
state_dict.pop("pooler.k_proj.bias")
state_dict.pop("pooler.q_proj.weight")
state_dict.pop("pooler.q_proj.bias")
state_dict.pop("pooler.v_proj.weight")
state_dict.pop("pooler.v_proj.bias")
state_dict.pop("pooler.c_proj.weight")
state_dict.pop("pooler.c_proj.bias")
state_dict.pop("pooler.positional_embedding")


# t_embedder -> time_embedding (`TimestepEmbedding`)
state_dict["time_extra_emb.timestep_embedder.linear_1.bias"] = state_dict["t_embedder.mlp.0.bias"]
state_dict["time_extra_emb.timestep_embedder.linear_1.weight"] = state_dict["t_embedder.mlp.0.weight"]
state_dict["time_extra_emb.timestep_embedder.linear_2.bias"] = state_dict["t_embedder.mlp.2.bias"]
state_dict["time_extra_emb.timestep_embedder.linear_2.weight"] = state_dict["t_embedder.mlp.2.weight"]

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
state_dict["time_extra_emb.extra_embedder.linear_1.bias"] = state_dict["extra_embedder.0.bias"]
state_dict["time_extra_emb.extra_embedder.linear_1.weight"] = state_dict["extra_embedder.0.weight"]
state_dict["time_extra_emb.extra_embedder.linear_2.bias"] = state_dict["extra_embedder.2.bias"]
state_dict["time_extra_emb.extra_embedder.linear_2.weight"] = state_dict["extra_embedder.2.weight"]
state_dict.pop("extra_embedder.0.bias")
state_dict.pop("extra_embedder.0.weight")
state_dict.pop("extra_embedder.2.bias")
state_dict.pop("extra_embedder.2.weight")

# style_embedder
print(state_dict["style_embedder.weight"])
print(state_dict["style_embedder.weight"].shape)
state_dict["time_extra_emb.style_embedder.weight"] = state_dict["style_embedder.weight"][0:1]
state_dict.pop("style_embedder.weight")

model.load_state_dict(state_dict)

model.save_pretrained('./HunyuanDiT-ControlNet-Pose')
# from diffusers import HunyuanDiTPipeline
# pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", torch_dtype=torch.float16)
# pipe.to('cuda')
# pipe.to(dtype=torch.float32)

# pipe.save_pretrained('../HunyuanDiT-ckpt-v1-1-Distilled')

# ### NOTE: HunyuanDiT supports both Chinese and English inputs
# prompt = "一个宇航员在骑马"
# #prompt = "An astronaut riding a horse"
# generator=torch.Generator(device="cuda").manual_seed(0)
# image = pipe(height=1024, width=1024, prompt=prompt, generator=generator, num_inference_steps=25, guidance_scale=5.0).images[0]

# image.save("img.png")