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


from .core import Conversion
from .hunyuan_video import hunyuan_video_conversion


def hunyuan_video15_conversion(config):
    base = hunyuan_video_conversion({**config, "guidance_embeds": False, "num_single_layers": 0})
    mapping = {
        old: new.replace("time_text_embed.timestep_embedder.", "time_embed.timestep_embedder.")
        if old.startswith("time_in.")
        else new
        for old, new in base.mapping.items()
        if not old.startswith("vector_in.")
    }
    rules = tuple(rule for rule in base.rules if not rule.original[0].startswith("double_blocks."))
    mapping["cond_type_embedding.weight"] = "cond_type_embed.weight"
    modules = [("byt5_in.layernorm", "context_embedder_2.norm")]
    modules.extend((f"byt5_in.fc{i}", f"context_embedder_2.linear_{i}") for i in (1, 2, 3))
    modules.extend(
        (f"vision_in.proj.{i}", f"image_embedder.{name}")
        for i, name in ((0, "norm_in"), (1, "linear_1"), (3, "linear_2"), (4, "norm_out"))
    )
    if config["use_meanflow"]:
        modules.extend(
            (f"time_r_in.mlp.{i}", f"time_embed.timestep_embedder_r.linear_{j}") for i, j in ((0, 1), (2, 2))
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
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
