"""Shared test helpers for transformer model tests.

Diffusers imports inside helpers are deliberately deferred to function bodies so the test-fetcher's
module-level import graph doesn't propagate edits to specific transformer source files through this
file into every pipeline test that imports it.
"""

from typing import Any


def create_flux_ip_adapter_state_dict(model) -> dict[str, dict[str, Any]]:
    """Create a dummy IP Adapter state dict for Flux transformer testing."""
    from diffusers.models.embeddings import ImageProjection
    from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor

    ip_cross_attn_state_dict = {}
    key_id = 0

    for name in model.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            continue

        joint_attention_dim = model.config["joint_attention_dim"]
        hidden_size = model.config["num_attention_heads"] * model.config["attention_head_dim"]
        sd = FluxIPAdapterAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=joint_attention_dim, scale=1.0
        ).state_dict()
        ip_cross_attn_state_dict.update(
            {
                f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                f"{key_id}.to_k_ip.bias": sd["to_k_ip.0.bias"],
                f"{key_id}.to_v_ip.bias": sd["to_v_ip.0.bias"],
            }
        )
        key_id += 1

    image_projection = ImageProjection(
        cross_attention_dim=model.config["joint_attention_dim"],
        image_embed_dim=(
            model.config["pooled_projection_dim"] if "pooled_projection_dim" in model.config.keys() else 768
        ),
        num_image_text_embeds=4,
    )

    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.weight": sd["image_embeds.weight"],
            "proj.bias": sd["image_embeds.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    return {"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict}
