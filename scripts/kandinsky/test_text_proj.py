import torch

from diffusers.pipelines.kandinsky.text_proj import KandinskyTextProjModel


model_dtype = torch.float32
# repo = "/home/yiyi_huggingface_co/test-kandinsky"
# unet = UNet2DConditionModel.from_pretrained(repo, subfolder='unet')

text_proj = KandinskyTextProjModel(
    clip_extra_context_tokens=10,  # num_image_embs= 10
    clip_text_encoder_hidden_states_dim=1024,  # text_encoder_in_dim1
    clip_embeddings_dim=768,  # text_encoder_in_dim2
    time_embed_dim=1536,  # model_channels * 4
    cross_attention_dim=768,  # model_dim
).to("cuda")

print("text proj checkpoint:")
for k, w in text_proj.state_dict().items():
    print(f"{k}:{w.shape}")

x = torch.randn(2, 4, 96, 96, device="cuda")
timesteps = torch.tensor([979.0, 979], device="cuda")
full_emb = torch.randn(2, 77, 1024, device="cuda").to(model_dtype)
pooled_emb = torch.randn(2, 768, device="cuda").to(model_dtype)
image_emb = torch.randn(2, 768, device="cuda").to(model_dtype)

text_encoder_hidden_states, additive_clip_time_embeddings = text_proj(
    image_embeddings=image_emb,
    prompt_embeds=image_emb,
    text_encoder_hidden_states=full_emb,
)

print(f"text_encoder_hidden_states:{text_encoder_hidden_states.shape},{text_encoder_hidden_states.mean()}")
# 2, 87, 768
print(f"additive_clip_time_embeddings:{additive_clip_time_embeddings.shape},{additive_clip_time_embeddings.mean()}")
# 2, 1536
