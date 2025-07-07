from diffusers.models.embeddings import FluxPosEmbed
import torch 

batch_size = 4 
seq_length = 16
img_seq_length = 32
txt_ids = torch.randn(batch_size, seq_length, 3)
img_ids = torch.randn(batch_size, img_seq_length, 3)

pos_embed = FluxPosEmbed(theta=10000, axes_dim=[4, 4, 8])
ids = torch.cat((txt_ids, img_ids), dim=1)
image_rotary_emb = pos_embed(ids)
# image_rotary_emb[0].shape=torch.Size([4, 48, 16]), image_rotary_emb[1].shape=torch.Size([4, 48, 16])
print(f"{image_rotary_emb[0].shape=}, {image_rotary_emb[1].shape=}")
