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
from .flux import flux_conversion
from .transforms import Reverse, Split


def ovis_image_conversion(config):
    base = flux_conversion({**config, "guidance_embeds": False})
    mapping = {}
    for old, new in base.mapping.items():
        if old.startswith("vector_in.") or "_mlp.0." in old:
            continue
        old = (
            old.replace("txt_in.", "semantic_txt_in.")
            .replace(".scale", ".weight")
            .replace("_mlp.2.", "_mlp.down_proj.")
        )
        new = new.replace("time_text_embed.timestep_embedder.", "timestep_embedder.")
        mapping[old] = new
    mapping["semantic_txt_norm.weight"] = "context_embedder_norm.weight"
    hidden = config["num_attention_heads"] * config["attention_head_dim"]
    mlp = 4 * hidden
    rules = []
    for rule in base.rules:
        if rule.original[0].startswith("single_blocks.") and ".linear1." in rule.original[0]:
            rule = Rule(rule.original, rule.diffusers, Split((hidden,) * 3 + (2 * mlp,)))
        rules.append(rule)
    for i in range(config["num_layers"]):
        for source, target in (("img", "ff"), ("txt", "ff_context")):
            for p in ("weight", "bias"):
                rules.append(
                    Rule(
                        (
                            f"double_blocks.{i}.{source}_mlp.up_proj.{p}",
                            f"double_blocks.{i}.{source}_mlp.gate_proj.{p}",
                        ),
                        (f"transformer_blocks.{i}.{target}.net.0.proj.{p}",),
                        Reverse(Split((mlp,) * 2)),
                    )
                )
    return Conversion(mapping=mapping, rules=tuple(rules))
