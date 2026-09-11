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
from .transforms import MergeEqual, Reverse


def lumina_conversion(config):
    mapping = {"pad_token": "pad_token"}
    modules = [
        ("x_embedder", "patch_embedder.proj"),
        ("cap_embedder.0", "time_caption_embed.caption_embedder.0"),
        ("cap_embedder.1", "time_caption_embed.caption_embedder.1"),
        ("final_layer.linear", "norm_out.linear_2"),
        ("final_layer.adaLN_modulation.1", "norm_out.linear_1"),
    ]
    modules.extend(
        (f"t_embedder.mlp.{i}", f"time_caption_embed.timestep_embedder.linear_{j}") for i, j in ((0, 1), (2, 2))
    )
    rules = []
    for i in range(config["num_layers"]):
        prefix = f"layers.{i}"
        mapping[prefix + ".attention.gate"] = prefix + ".gate"
        modules.append((prefix + ".adaLN_modulation.1", prefix + ".norm1.linear"))
        rules.append(
            Rule(
                (prefix + ".attention.wq.weight",),
                (prefix + ".attn1.to_q.weight", prefix + ".attn2.to_q.weight"),
                Reverse(MergeEqual(2)),
            )
        )
        for old, new in (
            ("attention.wk", "attn1.to_k"),
            ("attention.wv", "attn1.to_v"),
            ("attention.wk_y", "attn2.to_k"),
            ("attention.wv_y", "attn2.to_v"),
            ("attention.wo", "attn2.to_out.0"),
            ("attention_norm1", "norm1.norm"),
            ("attention_norm2", "norm2"),
            ("attention_y_norm", "norm1_context"),
            ("feed_forward.w1", "feed_forward.linear_1"),
            ("feed_forward.w2", "feed_forward.linear_2"),
            ("feed_forward.w3", "feed_forward.linear_3"),
            ("ffn_norm1", "ffn_norm1"),
            ("ffn_norm2", "ffn_norm2"),
        ):
            mapping[f"{prefix}.{old}.weight"] = f"{prefix}.{new}.weight"
        if config["qk_norm"]:
            for p in ("weight", "bias"):
                rules.append(
                    Rule(
                        (f"{prefix}.attention.q_norm.{p}",),
                        (f"{prefix}.attn1.norm_q.{p}", f"{prefix}.attn2.norm_q.{p}"),
                        Reverse(MergeEqual(2)),
                    )
                )
            modules.extend(
                [
                    (prefix + ".attention.k_norm", prefix + ".attn1.norm_k"),
                    (prefix + ".attention.ky_norm", prefix + ".attn2.norm_k"),
                ]
            )
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=tuple(rules))
