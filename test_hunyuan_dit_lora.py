import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

pipe.load_lora_weights("YOUR_LORA_PATH", weight_name="lora_weights.pt", adapter_name="yushi")

prompt = "玉石绘画风格，一个人在雨中跳舞"
image = pipe(
    prompt, num_inference_steps=50, generator=torch.manual_seed(0)
).images[0]
image.save('img.png')