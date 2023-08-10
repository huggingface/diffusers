import torch
from tortoise.models.arch_util import AttentionBlock
from diffusers.pipelines.tortoise_tts.modeling_common import TortoiseTTSSelfAttention

# input
ipt = torch.rand([5, 1024, 10])
print(ipt)

attn = AttentionBlock(channels=1024,
                      num_heads=16,
                      num_head_channels=-1,
                      relative_pos_embeddings=True
                      )
weights = torch.load("./tmp_weights/attn_block.bin")
attn.load_state_dict(weights, strict=True)
print(f"Tortoise-tts attn params - {sum([param.numel() for param in attn.parameters()])}")
attn.eval()


attn_hf = TortoiseTTSSelfAttention(query_dim=1024,
                                   n_heads=16,
                                   dim_head=64,
                                   dropout=0.0,
                                   bias=True,
                                   out_bias=True,
                                   scale_qk=True,
                                   norm_num_groups=32,
                                   has_relative_attention_bias=True,
                                   relative_attention_num_buckets=32,
                                   relative_attention_max_distance=128,
                                   )

# WEIGHTS
w1, w2, w3 = dict(attn.named_parameters())["qkv.weight"].squeeze(-1).split(split_size=1024, dim=0)
b1, b2, b3 = dict(attn.named_parameters())["qkv.bias"].split(split_size=1024, dim=0)
dim = 1024
n_heads = 16

# init weights
attn_hf.state_dict()["SelfAttention.q.weight"].copy_(torch.concatenate([w1[0*(dim//n_heads):1*(dim//n_heads), :],
                                                   w1[3*(dim//n_heads):4*(dim//n_heads), :],
                                                   w1[6*(dim//n_heads):7*(dim//n_heads), :],
                                                   w1[9*(dim//n_heads):10*(dim//n_heads), :],
                                                   w1[12*(dim//n_heads):13*(dim//n_heads), :],
                                                   w1[15*(dim//n_heads):16*(dim//n_heads), :],

                                                   w2[2*(dim//n_heads):3*(dim//n_heads), :],
                                                   w2[5*(dim//n_heads):6*(dim//n_heads), :],
                                                   w2[8*(dim//n_heads):9*(dim//n_heads), :],
                                                   w2[11*(dim//n_heads):12*(dim//n_heads), :],
                                                   w2[14*(dim//n_heads):15*(dim//n_heads), :],

                                                   w3[1*(dim//n_heads):2*(dim//n_heads), :],
                                                   w3[4*(dim//n_heads):5*(dim//n_heads), :],
                                                   w3[7*(dim//n_heads):8*(dim//n_heads), :],
                                                   w3[10*(dim//n_heads):11*(dim//n_heads), :],
                                                   w3[13*(dim//n_heads):14*(dim//n_heads), :],
                                                    ], axis=0))
attn_hf.state_dict()["SelfAttention.q.bias"].copy_(torch.concatenate([b1[0*(dim//n_heads):1*(dim//n_heads)],
                                                   b1[3*(dim//n_heads):4*(dim//n_heads)],
                                                   b1[6*(dim//n_heads):7*(dim//n_heads)],
                                                   b1[9*(dim//n_heads):10*(dim//n_heads)],
                                                   b1[12*(dim//n_heads):13*(dim//n_heads)],
                                                   b1[15*(dim//n_heads):16*(dim//n_heads)],

                                                   b2[2*(dim//n_heads):3*(dim//n_heads)],
                                                   b2[5*(dim//n_heads):6*(dim//n_heads)],
                                                   b2[8*(dim//n_heads):9*(dim//n_heads)],
                                                   b2[11*(dim//n_heads):12*(dim//n_heads)],
                                                   b2[14*(dim//n_heads):15*(dim//n_heads)],

                                                   b3[1*(dim//n_heads):2*(dim//n_heads)],
                                                   b3[4*(dim//n_heads):5*(dim//n_heads)],
                                                   b3[7*(dim//n_heads):8*(dim//n_heads)],
                                                   b3[10*(dim//n_heads):11*(dim//n_heads)],
                                                   b3[13*(dim//n_heads):14*(dim//n_heads)],]))

attn_hf.state_dict()["SelfAttention.k.weight"].copy_(torch.concatenate([w1[1*(dim//n_heads):2*(dim//n_heads), :],
                   w1[4*(dim//n_heads):5*(dim//n_heads), :],
                   w1[7*(dim//n_heads):8*(dim//n_heads), :],
                   w1[10*(dim//n_heads):11*(dim//n_heads), :],
                   w1[13*(dim//n_heads):14*(dim//n_heads), :],

                   w2[0*(dim//n_heads):1*(dim//n_heads), :],
                   w2[3*(dim//n_heads):4*(dim//n_heads), :],
                   w2[6*(dim//n_heads):7*(dim//n_heads), :],
                   w2[9*(dim//n_heads):10*(dim//n_heads), :],
                   w2[12*(dim//n_heads):13*(dim//n_heads), :],
                   w2[15*(dim//n_heads):16*(dim//n_heads), :],

                   w3[2*(dim//n_heads):3*(dim//n_heads), :],
                   w3[5*(dim//n_heads):6*(dim//n_heads), :],
                   w3[8*(dim//n_heads):9*(dim//n_heads), :],
                   w3[11*(dim//n_heads):12*(dim//n_heads), :],
                   w3[14*(dim//n_heads):15*(dim//n_heads), :],
                    ], axis=0))
attn_hf.state_dict()["SelfAttention.k.bias"].copy_(torch.concatenate([b1[1*(dim//n_heads):2*(dim//n_heads)],
                   b1[4*(dim//n_heads):5*(dim//n_heads)],
                   b1[7*(dim//n_heads):8*(dim//n_heads)],
                   b1[10*(dim//n_heads):11*(dim//n_heads)],
                   b1[13*(dim//n_heads):14*(dim//n_heads)],

                   b2[0*(dim//n_heads):1*(dim//n_heads)],
                   b2[3*(dim//n_heads):4*(dim//n_heads)],
                   b2[6*(dim//n_heads):7*(dim//n_heads)],
                   b2[9*(dim//n_heads):10*(dim//n_heads)],
                   b2[12*(dim//n_heads):13*(dim//n_heads)],
                   b2[15*(dim//n_heads):16*(dim//n_heads)],

                   b3[2*(dim//n_heads):3*(dim//n_heads)],
                   b3[5*(dim//n_heads):6*(dim//n_heads)],
                   b3[8*(dim//n_heads):9*(dim//n_heads)],
                   b3[11*(dim//n_heads):12*(dim//n_heads)],
                   b3[14*(dim//n_heads):15*(dim//n_heads)],]))

attn_hf.state_dict()["SelfAttention.v.weight"].copy_(torch.concatenate([w1[2*(dim//n_heads):3*(dim//n_heads), :],
                   w1[5*(dim//n_heads):6*(dim//n_heads), :],
                   w1[8*(dim//n_heads):9*(dim//n_heads), :],
                   w1[11*(dim//n_heads):12*(dim//n_heads), :],
                   w1[14*(dim//n_heads):15*(dim//n_heads), :],

                   w2[1*(dim//n_heads):2*(dim//n_heads), :],
                   w2[4*(dim//n_heads):5*(dim//n_heads), :],
                   w2[7*(dim//n_heads):8*(dim//n_heads), :],
                   w2[10*(dim//n_heads):11*(dim//n_heads), :],
                   w2[13*(dim//n_heads):14*(dim//n_heads), :],

                   w3[0*(dim//n_heads):1*(dim//n_heads), :],
                   w3[3*(dim//n_heads):4*(dim//n_heads), :],
                   w3[6*(dim//n_heads):7*(dim//n_heads), :],
                   w3[9*(dim//n_heads):10*(dim//n_heads), :],
                   w3[12*(dim//n_heads):13*(dim//n_heads), :],
                   w3[15*(dim//n_heads):16*(dim//n_heads), :],
                    ], axis=0))
attn_hf.state_dict()["SelfAttention.v.bias"].copy_(torch.concatenate([b1[2*(dim//n_heads):3*(dim//n_heads)],
                   b1[5*(dim//n_heads):6*(dim//n_heads)],
                   b1[8*(dim//n_heads):9*(dim//n_heads)],
                   b1[11*(dim//n_heads):12*(dim//n_heads)],
                   b1[14*(dim//n_heads):15*(dim//n_heads)],

                   b2[1*(dim//n_heads):2*(dim//n_heads)],
                   b2[4*(dim//n_heads):5*(dim//n_heads)],
                   b2[7*(dim//n_heads):8*(dim//n_heads)],
                   b2[10*(dim//n_heads):11*(dim//n_heads)],
                   b2[13*(dim//n_heads):14*(dim//n_heads)],

                   b3[0*(dim//n_heads):1*(dim//n_heads)],
                   b3[3*(dim//n_heads):4*(dim//n_heads)],
                   b3[6*(dim//n_heads):7*(dim//n_heads)],
                   b3[9*(dim//n_heads):10*(dim//n_heads)],
                   b3[12*(dim//n_heads):13*(dim//n_heads)],
                   b3[15*(dim//n_heads):16*(dim//n_heads)],]))

attn_hf.state_dict()["SelfAttention.o.weight"].copy_(dict(attn.named_parameters())["proj_out.weight"].squeeze(-1))
attn_hf.state_dict()["SelfAttention.o.bias"].copy_(dict(attn.named_parameters())["proj_out.bias"].squeeze(-1))

attn_hf.state_dict()["layer_norm.weight"].copy_(dict(attn.named_parameters())["norm.weight"])
attn_hf.state_dict()["layer_norm.bias"].copy_(dict(attn.named_parameters())["norm.bias"])

attn_hf.state_dict()["SelfAttention.relative_attention_bias.weight"].copy_(dict(attn.named_parameters())["relative_pos_embeddings.relative_attention_bias.weight"])

attn_hf.eval()

print(f"Tortoise-tts attn params - {sum([param.numel() for param in attn_hf.parameters()])}")

print("TORTOISE TTS Attn OPTS:\n", attn(ipt).permute(0, 2, 1))
print("\n\nHF Attn OPTS:\n", attn_hf(ipt)[0])

print(f"\n\nOutputs same or not : {torch.allclose(attn(ipt).permute(0, 2, 1), attn_hf(ipt)[0], atol=1e-4, rtol=1e-4)}")