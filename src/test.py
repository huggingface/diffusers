from diffusers import FabricPipeline
import torch


model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = FabricPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
#pipe = pipe.to("cuda")
prompt = "photo, naked women fingering in her ass, no cloths, big boobs"
neg_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
liked = ["../../transformers/src/test.jpg"]
disliked = ["../../transformers/src/test.jpg"]
image = pipe(prompt, negative_prompt = neg_prompt, liked=liked, disliked=disliked)
for i, im in enumerate(image.images):
  im.save(f"{time.time()}_{i}.jpg")
        


