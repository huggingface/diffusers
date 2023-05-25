import torch

from diffusers import KandinskyPipeline


prompt = "red cat, 4k photo"
device = "cuda"
# batch_size=1
# guidance_scale=4
# prior_cf_scale=4,
# prior_steps="5"

pipe_prior = KandinskyPipeline.from_pretrained("/home/yiyi_huggingface_co/test-kandinsky")
pipe_prior.to(device)

# step1. testing prior
# set_seed(0)
# hidden_states = torch.randn(2,768, device=device)
# print(f"hidden_states:{hidden_states.sum()}")
# timestep = torch.tensor([4,4], device=device)
# out = pipe_prior.prior(
#     hidden_states,
#     timestep,
#     proj_embedding,
#     encoder_hidden_states,
#     attention_mask

# )
# print(out)
# print(f"predicted_image_embedding: {out['predicted_image_embedding'].shape}, {out['predicted_image_embedding'].sum()}")

generator = torch.Generator(device="cuda").manual_seed(0)
out = pipe_prior(
    prompt,
    generator=generator,
)

print(f"image_embeddings:{out.shape},{out.sum()}")

print(out)
