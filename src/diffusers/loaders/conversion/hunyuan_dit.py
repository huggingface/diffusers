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
from .transforms import ReorderChunks, Split


def hunyuan_dit_conversion(config):
    mapping, rules = _hunyuan_dit_rules(config, controlnet=False)
    return Conversion(mapping=mapping, rules=rules)


def _hunyuan_dit_rules(config, controlnet):
    modules = [("x_embedder.proj", "pos_embed.proj")]
    mapping = {
        "text_embedding_padding": "text_embedding_padding",
        "pooler.positional_embedding": "time_extra_emb.pooler.positional_embedding",
    }
    modules.extend(
        (f"pooler.{name}", f"time_extra_emb.pooler.{name}") for name in ("q_proj", "k_proj", "v_proj", "c_proj")
    )
    modules.extend(
        (f"{source}.{i}", f"{target}.linear_{j}")
        for source, target in (
            ("t_embedder.mlp", "time_extra_emb.timestep_embedder"),
            ("mlp_t5", "text_embedder"),
            ("extra_embedder", "time_extra_emb.extra_embedder"),
        )
        for i, j in ((0, 1), (2, 2))
    )
    if config["use_style_cond_and_image_meta_size"]:
        mapping["style_embedder.weight"] = "time_extra_emb.style_embedder.weight"
    count = config["transformer_num_layers"] // 2 - 1 if controlnet else config["num_layers"]
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    rules = []
    for i in range(count):
        prefix = f"blocks.{i}"
        modules.extend(
            (f"{prefix}.{old}", f"{prefix}.{new}")
            for old, new in (
                ("norm1", "norm1.norm"),
                ("default_modulation.1", "norm1.linear"),
                ("norm2", "norm3"),
                ("norm3", "norm2"),
                ("mlp.fc1", "ff.net.0.proj"),
                ("mlp.fc2", "ff.net.2"),
                ("attn2.q_proj", "attn2.to_q"),
            )
        )
        for attn in ("attn1", "attn2"):
            modules.extend(
                (f"{prefix}.{attn}.{old}", f"{prefix}.{attn}.{new}")
                for old, new in (("q_norm", "norm_q"), ("k_norm", "norm_k"), ("out_proj", "to_out.0"))
            )
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{prefix}.attn1.Wqkv.{p}",),
                    tuple(f"{prefix}.attn1.to_{part}.{p}" for part in ("q", "k", "v")),
                    Split((hidden,) * 3),
                )
            )
            rules.append(
                Rule(
                    (f"{prefix}.attn2.kv_proj.{p}",),
                    tuple(f"{prefix}.attn2.to_{part}.{p}" for part in ("k", "v")),
                    Split((hidden,) * 2),
                )
            )
        if not controlnet and i > count // 2:
            modules.extend((f"{prefix}.{name}", f"{prefix}.{name}") for name in ("skip_norm", "skip_linear"))
    if controlnet:
        modules.append(("input_block", "input_block"))
        modules.extend((f"controlnet_blocks.{i}", f"controlnet_blocks.{i}") for i in range(count))
    else:
        modules.append(("final_layer.linear", "proj_out"))
        for p in ("weight", "bias"):
            rules.append(
                Rule((f"final_layer.adaLN_modulation.1.{p}",), (f"norm_out.linear.{p}",), ReorderChunks((1, 0)))
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return mapping, tuple(rules)
