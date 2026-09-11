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
from .sd3 import sd3_conversion
from .transforms import Split


def sd3_controlnet_conversion(config):
    modules = [("pos_embed_input.proj", "pos_embed_input.proj")]
    modules.extend(
        (f"time_text_embed.{kind}.linear_{i}", f"time_text_embed.{kind}.linear_{i}")
        for kind in ("timestep_embedder", "text_embedder")
        for i in (1, 2)
    )
    modules.extend((f"controlnet_blocks.{i}", f"controlnet_blocks.{i}") for i in range(config["num_layers"]))
    mapping, rules = {}, []
    if config["joint_attention_dim"] is not None:
        # Joint-stream ControlNets were trained directly with Diffusers and already use its parameter names.
        base = sd3_conversion({**config, "num_layers": config["num_layers"] + 1})
        for key in sorted(base.diffusers_keys):
            if key.startswith("transformer_blocks.") and int(key.split(".")[1]) < config["num_layers"]:
                mapping[key] = key
        modules.append(("context_embedder", "context_embedder"))
    else:
        hidden = config["num_attention_heads"] * config["attention_head_dim"]
        for i in range(config["num_layers"]):
            prefix = f"transformer_blocks.{i}"
            for p in ("weight", "bias"):
                rules.append(
                    Rule(
                        (f"{prefix}.attn.qkv.{p}",),
                        tuple(f"{prefix}.attn.to_{part}.{p}" for part in ("q", "k", "v")),
                        Split((hidden,) * 3),
                    )
                )
            modules.extend(
                (f"{prefix}.{a}", f"{prefix}.{b}")
                for a, b in (
                    ("attn.proj", "attn.to_out.0"),
                    ("mlp.fc1", "ff.net.0.proj"),
                    ("mlp.fc2", "ff.net.2"),
                    ("adaLN_modulation.1", "norm1.linear"),
                )
            )
    if config["use_pos_embed"]:
        modules.append(("pos_embed.proj", "pos_embed.proj"))
        if config["pos_embed_max_size"] is not None and config["pos_embed_type"] is not None:
            mapping["pos_embed.pos_embed"] = "pos_embed.pos_embed"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
