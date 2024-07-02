import glob
import hashlib

import pandas as pd
import torch
from transformers import T5EncoderModel

from diffusers import StableDiffusion3Pipeline


PROMPT = "a photo of sks dog"
MAX_SEQ_LENGTH = 77


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


def generate_image_hash(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    return hashlib.sha256(img_data).hexdigest()


id = "stabilityai/stable-diffusion-3-medium-diffusers"
text_encoder = T5EncoderModel.from_pretrained(id, subfolder="text_encoder_3", load_in_8bit=True, device_map="auto")
pipeline = StableDiffusion3Pipeline.from_pretrained(
    id, text_encoder_3=text_encoder, transformer=None, vae=None, device_map="balanced"
).to("cuda")
with torch.no_grad():
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(prompt=PROMPT, prompt_2=None, prompt_3=None, max_sequence_length=MAX_SEQ_LENGTH)
    print(
        f"{prompt_embeds.shape=}, {negative_prompt_embeds.shape=}, {pooled_prompt_embeds.shape=}, {negative_pooled_prompt_embeds.shape}"
    )

print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")

local_dir = "dog"
image_paths = glob.glob(f"{local_dir}/*.jpeg")

data = []
for image_path in image_paths:
    img_hash = generate_image_hash(image_path)
    data.append((img_hash, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds))

# Create a DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "image_hash",
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    ],
)

# Convert embedding lists to arrays (for proper storage in parquet)
for col in ["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]:
    df[col] = df[col].apply(lambda x: x.cpu().numpy().flatten().tolist())


# Save the table to a parquet file
output_path = "sample_embeddings.parquet"
df.to_parquet(output_path)

print(f"Data successfully serialized to {output_path}")
