# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .core import Conversion, Rule
from .transforms import Chain, Permute, Reshape, Split
from .unet_2d import unet_2d_conversion


def unclip_unet_conversion(config):
    base = unet_2d_conversion(
        {**config, "original_format": "ldm", "downsample_type": "resnet", "upsample_type": "resnet"}
    )
    mapping, rules = dict(base.mapping), list(base.rules)
    modules = []
    if config.get("addition_embed_type") == "text_image":
        modules.extend(
            [
                ("ln_model_n", "add_embedding.text_norm"),
                ("proj_n", "add_embedding.text_proj"),
                ("img_layer", "add_embedding.image_proj"),
            ]
        )
    if config.get("encoder_hid_dim_type") == "text_image_proj":
        modules.extend(
            [("clip_to_seq", "encoder_hid_proj.image_embeds"), ("to_model_dim_n", "encoder_hid_proj.text_proj")]
        )
    for rule in base.rules:
        source = rule.original[0]
        if not source.endswith(".qkv.weight"):
            continue
        old = source.removesuffix(".qkv.weight")
        new = rule.diffusers[0].removesuffix(".to_q.weight")
        parts = new.split(".")
        if parts[0] == "mid_block":
            index = len(config["block_out_channels"]) - 1
        elif parts[0] == "up_blocks":
            index = len(config["block_out_channels"]) - 1 - int(parts[1])
        else:
            index = int(parts[1])
        channel = config["block_out_channels"][index]
        dim = config["attention_head_dim"]
        dim = dim[index] if isinstance(dim, (tuple, list)) else dim
        heads = channel // dim
        inner = heads * dim
        context = config["cross_attention_dim"]
        context = context[index] if isinstance(context, (tuple, list)) else context
        for p in ("weight", "bias"):
            trailing = (context,) if p == "weight" else ()
            original_shape = (2 * inner, context, 1) if p == "weight" else (2 * inner,)
            transform = Chain(
                (
                    Reshape(original_shape, (heads, 2, dim) + trailing),
                    Permute((1, 0, 2, 3) if trailing else (1, 0, 2)),
                    Reshape((2, heads, dim) + trailing, (2 * inner,) + trailing),
                    Split((inner,) * 2),
                )
            )
            rules.append(
                Rule((f"{old}.encoder_kv.{p}",), (f"{new}.add_k_proj.{p}", f"{new}.add_v_proj.{p}"), transform)
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
