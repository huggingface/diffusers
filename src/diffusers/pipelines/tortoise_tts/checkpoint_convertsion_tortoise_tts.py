


# WIP (having issues with dealing with the n_heads>2) #

from tortoise.models.arch_util import AttentionBlock
attn = AttentionBlock(channels=1024,
                      num_heads=16,
                      num_head_channels=-1,
                      relative_pos_embeddings=True
                      )
weights = torch.load("./tmp_weights/attn_block.bin")
attn.load_state_dict(weights, strict=True)

cfg = T5Config()
cfg.d_model = 1024
cfg.num_heads = 16
cfg.dropout_rate = 0.0
cfg.is_encoder_decoder = False
cfg.use_cache = False
attn_hf = TortoiseTTSDiffusionModelSelfAttention(cfg, has_relative_attention_bias=True)

print(attn_hf.state_dict().keys())


# WEIGHTS
q_weight, k_weight, v_weight = dict(attn.named_parameters())["qkv.weight"].squeeze(-1).split(split_size=1024, dim=0)
q_bias, k_bias, v_bias = dict(attn.named_parameters())["qkv.bias"].split(split_size=1024, dim=0)

# init weights
attn_hf.state_dict()["SelfAttention.q.weight"].copy_(q_weight)
attn_hf.state_dict()["SelfAttention.q.bias"].copy_(q_bias)

attn_hf.state_dict()["SelfAttention.k.weight"].copy_(k_weight)
attn_hf.state_dict()["SelfAttention.k.bias"].copy_(k_bias)

attn_hf.state_dict()["SelfAttention.v.weight"].copy_(v_weight)
attn_hf.state_dict()["SelfAttention.v.bias"].copy_(v_bias)

attn_hf.state_dict()["SelfAttention.o.weight"].copy_(dict(attn.named_parameters())["proj_out.weight"].squeeze(-1))
attn_hf.state_dict()["SelfAttention.o.weight"].copy_(dict(attn.named_parameters())["proj_out.bias"].squeeze(-1))

attn_hf.state_dict()["layer_norm.weight"].copy_(dict(attn.named_parameters())["norm.weight"])
attn_hf.state_dict()["layer_norm.bias"].copy_(dict(attn.named_parameters())["norm.bias"])

attn_hf.state_dict()["SelfAttention.relative_attention_bias.weight"].copy_(dict(attn.named_parameters())["relative_pos_embeddings.relative_attention_bias.weight"])


print(f"Tortoise-tts attn params - {sum([param.numel() for param in attn_hf.parameters()])}")
ipt = torch.load("./tmp_files/ipt_attn.bin")

attn_hf.eval()
print(attn_hf(ipt))