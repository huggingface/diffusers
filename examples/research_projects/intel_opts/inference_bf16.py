import intel_extension_for_pytorch as ipex
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


prompt = ["a lovely <dicoo> in red dress and hat, in the snowly and brightly night, with many brighly buildings"]
batch_size = 8
prompt = prompt * batch_size

device = "cpu"
model_id = "path-to-your-trained-model"
model = StableDiffusionPipeline.from_pretrained(model_id)
model = model.to(device)

# to channels last
model.unet = model.unet.to(memory_format=torch.channels_last)
model.vae = model.vae.to(memory_format=torch.channels_last)
model.text_encoder = model.text_encoder.to(memory_format=torch.channels_last)
model.safety_checker = model.safety_checker.to(memory_format=torch.channels_last)

# optimize with ipex
model.unet = ipex.optimize(model.unet.eval(), dtype=torch.bfloat16, inplace=True)
model.vae = ipex.optimize(model.vae.eval(), dtype=torch.bfloat16, inplace=True)
model.text_encoder = ipex.optimize(model.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
model.safety_checker = ipex.optimize(model.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)

# compute
seed = 666
generator = torch.Generator(device).manual_seed(seed)
with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    images = model(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images

    # save image
    grid = image_grid(images, rows=2, cols=4)
    grid.save(model_id + ".png")
