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


def vq_diffusion_conversion(config):
    prefix = "transformer.transformer"
    mapping = {
        f"{prefix}.content_emb.{name}.weight": f"latent_image_embedding.{name}.weight"
        for name in ("emb", "height_emb", "width_emb")
    }
    modules = [(f"{prefix}.to_logits.0", "norm_out"), (f"{prefix}.to_logits.1", "out")]
    for i in range(config["num_layers"]):
        old, new = f"{prefix}.blocks.{i}", f"transformer_blocks.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("ln1.linear", "norm1.linear"),
                ("ln1_1.linear", "norm2.linear"),
                ("ln2", "norm3"),
                ("mlp.0", "ff.net.0.proj"),
                ("mlp.2", "ff.net.2"),
            )
        )
        for a, b in (("ln1", "norm1"), ("ln1_1", "norm2")):
            mapping[f"{old}.{a}.emb.weight"] = f"{new}.{b}.emb.weight"
        for attn in ("attn1", "attn2"):
            modules.append((f"{old}.{attn}.proj", f"{new}.{attn}.to_out.0"))
            for a, b in (("query", "to_q"), ("key", "to_k"), ("value", "to_v")):
                for p in ("weight", "bias") if config["attention_bias"] else ("weight",):
                    mapping[f"{old}.{attn}.{a}.{p}"] = f"{new}.{attn}.{b}.{p}"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
