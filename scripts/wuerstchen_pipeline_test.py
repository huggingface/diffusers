import os
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import WuerstchenPriorPipeline, WuerstchenGeneratorPipeline

transformers.utils.logging.set_verbosity_error()


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# effnet_preprocess = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.Resize(
#             768, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True
#         ),
#         torchvision.transforms.CenterCrop(768),
#         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ]
# )

# transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Resize(1024),
#         torchvision.transforms.RandomCrop(1024),
#     ]
# )
device = "cuda"
dtype = torch.float16
batch_size = 4

# generator_pipeline = WuerstchenGeneratorPipeline.from_pretrained("C:\\Users\\d6582\\Documents\\ml\\diffusers\\scripts\\warp-diffusion\\WuerstchenGeneratorPipeline", torch_dtype=dtype)
# generator_pipeline = generator_pipeline.to("cuda")
# text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# image = Image.open("C:\\Users\\d6582\\Documents\\ml\\wand\\finetuning\\images\\fernando\\IMG_0352.JPG")
# image = effnet_preprocess(transforms(image).unsqueeze(0).expand(batch_size, -1, -1, -1)).to("cuda").to(dtype)
# print(image.shape)

# caption = "princess | centered| key visual| intricate| highly detailed| breathtaking beauty| precise lineart| vibrant| comprehensive cinematic| Carne Griffiths| Conrad Roset"
# negative_prompt = "low resolution, low detail, bad quality, blurry"

# clip_tokens = tokenizer([caption] * image.size(0), truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to("cuda")
# clip_text_embeddings = text_encoder(**clip_tokens).last_hidden_state.to(dtype)
# clip_tokens_uncond = tokenizer([negative_prompt] * image.size(0), truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to("cuda")
# clip_text_embeddings_uncond = text_encoder(**clip_tokens_uncond).last_hidden_state.to(dtype)

# image_embeds = generator_pipeline.encode_image(image)
# generator_output = generator_pipeline(image_embeds, clip_text_embeddings, guidance_scale=0.0, output_type="np").images
# images = numpy_to_pil(generator_output)
# os.makedirs("samples", exist_ok=True)
# for i, image in enumerate(images):
#     image.save(os.path.join("samples", caption.replace(" ", "_").replace("|", "") + f"_{i}.png"))


prior_pipeline = WuerstchenPriorPipeline.from_pretrained("warp-diffusion/WuerstchenPriorPipeline", torch_dtype=dtype)
generator_pipeline = WuerstchenGeneratorPipeline.from_pretrained(
    "warp-diffusion/WuerstchenGeneratorPipeline", torch_dtype=dtype
)
prior_pipeline = prior_pipeline.to("cuda")
generator_pipeline = generator_pipeline.to("cuda")
# generator_pipeline.vqgan.to(torch.float16)
# text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to("cpu")
# tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# negative_prompt = "low resolution, low detail, bad quality, blurry"
negative_prompt = "bad anatomy, blurry, fuzzy, extra arms, extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated, drawing, helmet"
# negative_prompt = ""
caption = (
    "Bee flying out of a glass jar in a green and red leafy basket, glass and lens flare, diffuse lighting elegant"
)
# caption = "princess | centered| key visual| intricate| highly detailed| breathtaking beauty| precise lineart| vibrant| comprehensive cinematic| Carne Griffiths| Conrad Roset"
# caption = "An armchair in the shape of an avocado"
# clip_tokens = tokenizer(
#     [caption] * batch_size,
#     truncation=True,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     return_tensors="pt",
# )
# clip_text_embeddings = text_encoder(**clip_tokens).last_hidden_state.to(dtype).to(device)
# clip_tokens_uncond = tokenizer(
#     [negative_prompt] * batch_size,
#     truncation=True,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     return_tensors="pt",
# )
# clip_text_embeddings_uncond = text_encoder(**clip_tokens_uncond).last_hidden_state.to(dtype).to(device)

prior_output = prior_pipeline(
    caption,
    guidance_scale=8.0,
    num_images_per_prompt=batch_size,
    negative_prompt=negative_prompt,
)
generator_output = generator_pipeline(
    predicted_image_embeddings=prior_output.image_embeds,
    prompt=caption,
    negative_prompt=negative_prompt,
    guidance_scale=8.0,
    output_type="np",
).images
images = numpy_to_pil(generator_output)
os.makedirs("samples", exist_ok=True)
for i, image in enumerate(images):
    image.save(os.path.join("samples", caption.replace(" ", "_").replace("|", "") + f"_{i}.png"))


# caption = input("Prompt please: ")
# while caption != "q":
#     prior_output = prior_pipeline(caption, num_images_per_prompt=4, negative_prompt=negative_prompt)
#     generator_output = generator_pipeline(prior_output.image_embeds, prior_output.text_embeds, output_type="np").images
#     images = numpy_to_pil(generator_output)

#     os.makedirs("samples", exist_ok=True)
#     for i, image in enumerate(images):
#         image.save(os.path.join("samples", caption.replace(" ", "_").replace("|", "") + f"_{i}.png"))

#     caption = input("Prompt please: ")
