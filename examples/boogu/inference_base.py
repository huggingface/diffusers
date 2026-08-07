import torch

from diffusers.pipelines.boogu import BooguImagePipeline


MODEL_PATH = "Boogu/Boogu-Image-0.1-Base"

pipe = BooguImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

images = pipe(
    instruction="一幅国风琉金风格的山水画作，展现了桂林山水在金光普照下的壮丽景象。远山层叠，江水如镜，山峰边缘勾勒着发光的金色线条。画面采用石青石绿岩彩与鎏金质感相结合，局部有厚涂油画笔触，空中飘浮着金色粒子，营造出梦幻朦胧而又磅礴大气的意境。",
    height=1024,
    width=1024,
    num_inference_steps=50,
    text_guidance_scale=4.0,
).images

images[0].save("base.png")
print("Inference OK, saved base.png")
