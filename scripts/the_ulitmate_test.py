# import torch
# from diffuzz import Diffuzz
# from diffusers import DDPMWuerstchenScheduler

# torch.manual_seed(42)
# scheduler = DDPMWuerstchenScheduler()
# scheduler.set_timesteps({0.0: 30})
# diffuzz = Diffuzz()

# shape = (1, 16, 24, 24)
# x = torch.randn(shape)
# noise = torch.randn(shape)
# t = torch.rand(1)
# t_prev = t - 0.1

# output_diffuzz = diffuzz.undiffuse(x, t, t_prev, noise)
# output_scheduler = scheduler.step(noise, timestep=t, prev_t=t_prev, sample=x).prediction
# # scheduler.step(noise, timestep=t, sample=x)

# print(output_diffuzz.mean())
# print(output_scheduler.mean())
# print(output_diffuzz.shape)
# print(output_scheduler.shape)

from transformers import AutoTokenizer, CLIPTextModel

device = "cuda"

def embed_clip(caption, negative_caption="", batch_size=4, device="cuda"):
    clip_tokens = clip_tokenizer([caption] * batch_size, truncation=True, padding="max_length", max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
    clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state
    return clip_text_embeddings

clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device).eval().requires_grad_(False)
clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

caption = "An armchair in the shape of an avocado"

emb = embed_clip(caption)

print(emb)