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
from .hunyuan_video import hunyuan_video_conversion
from .transforms import Split


def hunyuan_image_conversion(config):
    base = hunyuan_video_conversion(config)
    mapping = {}
    for old, new in base.mapping.items():
        if old.startswith("vector_in."):
            continue
        if old.startswith(("time_in.", "guidance_in.")):
            new = new.replace("time_text_embed.", "time_guidance_embed.")
        if old.startswith("single_blocks."):
            old = old.replace(".linear2.", ".linear2.fc.")
        mapping[old] = new
    rules = [rule for rule in base.rules if not rule.original[0].startswith(("double_blocks.", "single_blocks."))]
    modules = []
    if config["text_embed_2_dim"] is not None:
        modules.append(("byt5_in.layernorm", "context_embedder_2.norm"))
        modules.extend((f"byt5_in.fc{i}", f"context_embedder_2.linear_{i}") for i in (1, 2, 3))
    if config["use_meanflow"]:
        modules.extend(
            (f"time_r_in.mlp.{i}", f"time_guidance_embed.timestep_embedder_r.linear_{j}") for i, j in ((0, 1), (2, 2))
        )
    for i in range(config["num_layers"]):
        for source, targets in (
            ("img", ("to_q", "to_k", "to_v")),
            ("txt", ("add_q_proj", "add_k_proj", "add_v_proj")),
        ):
            modules.extend(
                (f"double_blocks.{i}.{source}_attn_{part}", f"transformer_blocks.{i}.attn.{target}")
                for part, target in zip(("q", "k", "v"), targets)
            )
    for i in range(config["num_single_layers"]):
        modules.extend(
            (f"single_blocks.{i}.linear1_{part}", f"single_transformer_blocks.{i}.{target}")
            for part, target in (("q", "attn.to_q"), ("k", "attn.to_k"), ("v", "attn.to_v"), ("mlp", "proj_mlp"))
        )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    original_format = config.get(
        "original_format", "hunyuan_image_split" if config["use_meanflow"] else "hunyuan_image_fused"
    )
    if original_format == "hunyuan_image_fused":
        width = config["num_attention_heads"] * config["attention_head_dim"]
        for i in range(config["num_layers"]):
            for modality in ("img", "txt"):
                for p in ("weight", "bias"):
                    targets = tuple(
                        mapping.pop(f"double_blocks.{i}.{modality}_attn_{part}.{p}") for part in ("q", "k", "v")
                    )
                    rules.append(Rule((f"double_blocks.{i}.{modality}_attn_qkv.{p}",), targets, Split((width,) * 3)))
        for i in range(config["num_single_layers"]):
            for p in ("weight", "bias"):
                targets = tuple(
                    mapping.pop(f"single_blocks.{i}.linear1_{part}.{p}") for part in ("q", "k", "v", "mlp")
                )
                rules.append(
                    Rule(
                        (f"single_blocks.{i}.linear1.{p}",),
                        targets,
                        Split((width, width, width, int(width * config["mlp_ratio"]))),
                    )
                )
                mapping[f"single_blocks.{i}.linear2.{p}"] = mapping.pop(f"single_blocks.{i}.linear2.fc.{p}")
    elif original_format != "hunyuan_image_split":
        raise ValueError("Hunyuan Image original_format must be hunyuan_image_fused or hunyuan_image_split.")
    return Conversion(mapping=mapping, rules=rules)
