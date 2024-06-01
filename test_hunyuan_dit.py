# integration test (hunyuan dit)
import torch
from huggingface_hub import hf_hub_download

from diffusers import HunyuanDiTPipeline, HunyuanDiT2DModel
from transformers import T5EncoderModel, MT5Tokenizer

import safetensors.torch

device = "cuda"
dtype = torch.float16

repo = "XCLiu/HunyuanDiT-0523"
tokenizer_2 = MT5Tokenizer.from_pretrained(repo, subfolder = "tokenizer_t5", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(repo, subfolder = "embedder_t5", torch_dtype=dtype)

model_config = HunyuanDiT2DModel.load_config("XCLiu/HunyuanDiT-0523", subfolder="transformer")
model = HunyuanDiT2DModel.from_config(model_config).to(device)

ckpt_path = hf_hub_download(
    "XCLiu/HunyuanDiT-0523",
    filename ="diffusion_pytorch_model.safetensors",
    subfolder="transformer",
)

state_dict = safetensors.torch.load_file(ckpt_path)

prefix = "time_extra_emb."

# time_embedding.linear_1 -> timestep_embedder.linear_1 
state_dict[f"{prefix}timestep_embedder.linear_1.weight"] = state_dict["time_embedding.linear_1.weight"]
state_dict[f"{prefix}timestep_embedder.linear_1.bias"] = state_dict["time_embedding.linear_1.bias"]
state_dict.pop("time_embedding.linear_1.weight")
state_dict.pop("time_embedding.linear_1.bias")

# time_embedding.linear_2 -> timestep_embedder.linear_2
state_dict[f"{prefix}timestep_embedder.linear_2.weight"] = state_dict["time_embedding.linear_2.weight"]
state_dict[f"{prefix}timestep_embedder.linear_2.bias"] = state_dict["time_embedding.linear_2.bias"]
state_dict.pop("time_embedding.linear_2.weight")
state_dict.pop("time_embedding.linear_2.bias")

# pooler.positional_embedding
state_dict[f"{prefix}pooler.positional_embedding"] = state_dict["pooler.positional_embedding"]
state_dict.pop("pooler.positional_embedding")

# pooler.k_proj
state_dict[f"{prefix}pooler.k_proj.weight"] = state_dict["pooler.k_proj.weight"]
state_dict[f"{prefix}pooler.k_proj.bias"] = state_dict["pooler.k_proj.bias"]
state_dict.pop("pooler.k_proj.weight")
state_dict.pop("pooler.k_proj.bias")

#pooler.q_proj
state_dict[f"{prefix}pooler.q_proj.weight"] = state_dict["pooler.q_proj.weight"]
state_dict[f"{prefix}pooler.q_proj.bias"] = state_dict["pooler.q_proj.bias"]
state_dict.pop("pooler.q_proj.weight")
state_dict.pop("pooler.q_proj.bias")

#  pooler.v_proj
state_dict[f"{prefix}pooler.v_proj.weight"] = state_dict["pooler.v_proj.weight"]
state_dict[f"{prefix}pooler.v_proj.bias"] = state_dict["pooler.v_proj.bias"]
state_dict.pop("pooler.v_proj.weight")
state_dict.pop("pooler.v_proj.bias")

# pooler.c_proj
state_dict[f"{prefix}pooler.c_proj.weight"] = state_dict["pooler.c_proj.weight"]
state_dict[f"{prefix}pooler.c_proj.bias"] = state_dict["pooler.c_proj.bias"]
state_dict.pop("pooler.c_proj.weight")
state_dict.pop("pooler.c_proj.bias")

# style_embedder.weight
state_dict[f"{prefix}style_embedder.weight"] = state_dict["style_embedder.weight"]
state_dict.pop("style_embedder.weight")

# extra_embedder.linear_1
state_dict[f"{prefix}extra_embedder.linear_1.weight"] = state_dict["extra_embedder.linear_1.weight"]
state_dict[f"{prefix}extra_embedder.linear_1.bias"] = state_dict["extra_embedder.linear_1.bias"]
state_dict.pop("extra_embedder.linear_1.weight")
state_dict.pop("extra_embedder.linear_1.bias")

# extra_embedder.linear_2
state_dict[f"{prefix}extra_embedder.linear_2.weight"] = state_dict["extra_embedder.linear_2.weight"]
state_dict[f"{prefix}extra_embedder.linear_2.bias"] = state_dict["extra_embedder.linear_2.bias"]
state_dict.pop("extra_embedder.linear_2.weight")
state_dict.pop("extra_embedder.linear_2.bias")

model.load_state_dict(state_dict)
model.to(device, dtype)

pipe = HunyuanDiTPipeline.from_pretrained(
    repo, 
    tokenizer_2 = tokenizer_2,
    text_encoder_2 = text_encoder_2,
    transformer = model,
    torch_dtype=dtype)

pipe.enable_model_cpu_offload()

### NOTE: HunyuanDiT supports both Chinese and English inputs
prompt = "一个宇航员在骑马"
#prompt = "An astronaut riding a horse"
generator=torch.Generator(device="cuda").manual_seed(0)
image = pipe(height=1024, width=1024, prompt=prompt, generator=generator).images[0]

image.save("img.png")