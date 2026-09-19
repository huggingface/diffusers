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
from .transforms import Reshape
from .unclip_unet import unclip_unet_conversion


def if_unet_conversion(config):
    base = unclip_unet_conversion(config)
    mapping, rules = dict(base.mapping), []
    modules = []
    if config.get("encoder_hid_dim") is not None:
        modules.append(("encoder_proj", "encoder_hid_proj"))
    if config.get("addition_embed_type") == "text":
        modules.extend(
            [
                ("encoder_pooling.0", "add_embedding.norm1"),
                ("encoder_pooling.2", "add_embedding.proj"),
                ("encoder_pooling.3", "add_embedding.norm2"),
            ]
        )
        modules.extend(
            (f"encoder_pooling.1.{part}_proj", f"add_embedding.pool.{part}_proj") for part in ("q", "k", "v")
        )
        mapping["encoder_pooling.1.positional_embedding"] = "add_embedding.pool.positional_embedding"
    if config.get("class_embed_type") in ("timestep", "projection"):
        modules.extend([("label_emb.0.0", "class_embedding.linear_1"), ("label_emb.0.2", "class_embedding.linear_2")])
    for rule in base.rules:
        source = rule.original[0]
        if ".qkv." not in source:
            rules.append(rule)
            continue
        old = source.rsplit(".qkv.", 1)[0]
        new = rule.diffusers[0].rsplit(".to_q.", 1)[0]
        p = source.rsplit(".", 1)[1]
        parts = new.split(".")
        count = len(config["block_out_channels"])
        index = (
            count - 1
            if parts[0] == "mid_block"
            else (count - 1 - int(parts[1]) if parts[0] == "up_blocks" else int(parts[1]))
        )
        only_cross = config.get("only_cross_attention", False)
        only_cross = only_cross[index] if isinstance(only_cross, (tuple, list)) else only_cross
        if parts[0] == "mid_block":
            only_cross = (
                config.get("mid_block_only_cross_attention")
                if config.get("mid_block_only_cross_attention") is not None
                else only_cross
            )
        if only_cross:
            channel = config["block_out_channels"][index]
            dim = config["attention_head_dim"]
            dim = dim[index] if isinstance(dim, (tuple, list)) else dim
            inner = channel // dim * dim
            if p == "weight":
                rules.append(Rule((source,), (f"{new}.to_q.weight",), Reshape((inner, channel, 1), (inner, channel))))
            else:
                mapping[source] = f"{new}.to_q.bias"
        else:
            rules.append(rule)
        if p == "weight" and config.get("cross_attention_norm") is not None:
            modules.append((f"{old}.norm_encoder", f"{new}.norm_cross"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
