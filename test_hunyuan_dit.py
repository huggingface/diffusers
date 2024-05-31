# integration test (hunyuan dit)
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("XCLiu/HunyuanDiT-0523", torch_dtype=torch.float16)
pipe.to('cuda')

### NOTE: HunyuanDiT supports both Chinese and English inputs
prompt = "一个宇航员在骑马"
#prompt = "An astronaut riding a horse"
generator=torch.Generator(device="cuda").manual_seed(0)
image = pipe(height=1024, width=1024, prompt=prompt, generator=generator).images[0]

image.save("img.png")