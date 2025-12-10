import torch 
import torch.nn.functional as F
from typing import Optional, Tuple
from .transformer_wan import WanAttention
from .wan_mako_kernels import triton_matmul, triton_rms_norm2, fused_matmul_residual


# TODO: incorporate I2V support
class WanMakoAttnProcessor:
    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, S, D = hidden_states.shape
        H = attn.heads
        head_dim = D // H  # or assert against attn.inner_dim if needed

        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            # if you don't need this, drop it to avoid the slice
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # --- QKV / KV projections -------------------------------------------------
        if hasattr(attn, "w_qkv_self") and hasattr(attn, "b_qkv_self"):
            # Fused QKV via single matmul (self-attention)
            qkv = triton_matmul(hidden_states, attn.w_qkv_self, attn.b_qkv_self)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Q Projection (self from hidden_states)
            q = triton_matmul(hidden_states, attn.to_q.weight, attn.to_q.bias)
            # Fused KV Projection (cross from encoder_hidden_states)
            kv = triton_matmul(encoder_hidden_states, attn.w_kv_cross, attn.b_kv_cross)
            k, v = kv.chunk(2, dim=-1)

        # --- Fused RMS Norm for Q and K to reduce 1 launch ----------------------
        q, k = triton_rms_norm2(
            q, attn.norm_q.weight,
            k, attn.norm_k.weight,
            attn.norm_q.eps,
        )

        # --- Reshape ------------------------------------
        q, k, v = (a.unflatten(2, (attn.heads, -1)) for a in (q, k, v))

        # --- Rotary embedding -----------------------------------------------------
        if rotary_emb is not None:
            
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            q = apply_rotary_emb(q, *rotary_emb)
            k = apply_rotary_emb(k, *rotary_emb)

        # --- Scaled dot-product attention ----------------------------------------
        q, k, v = (x.permute(0, 2, 1, 3) for x in (q, k, v))
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False,
        )

        # (B, H, S, head_dim) -> (B, S, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)

        return attn_out
