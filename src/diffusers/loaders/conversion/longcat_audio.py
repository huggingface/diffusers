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


def longcat_audio_conversion(config):
    modules = [("proj_out", True), ("norm_out.linear", config["bias"])]
    modules.extend((f"time_embed.time_mlp.{i}", True) for i in (0, 2))
    embeddings = ["input_embed", "text_embed"]
    if config["use_latent_condition"]:
        embeddings.extend(["latent_embed", "latent_cond_embedder"])
    modules.extend((f"{name}.proj.{i}", True) for name in embeddings for i in (0, 2))
    if config["adaln_type"] == "global":
        modules.append(("adaln_global_mlp.mlp.1", True))
    keys = []
    if config["text_conv"]:
        for i in range(4):
            prefix = f"text_conv_layer.{i}"
            modules.append((prefix + ".norm", True))
            modules.extend((f"{prefix}.{name}", config["bias"]) for name in ("dwconv", "pwconv1", "pwconv2"))
            keys.extend(f"{prefix}.grn.{name}" for name in ("gamma", "beta"))
    for i in range(config["dit_depth"]):
        prefix = f"blocks.{i}"
        if config["adaln_type"] == "global":
            keys.append(prefix + ".adaln_scale_shift")
        elif config["adaln_type"] == "local":
            modules.append((prefix + ".adaln_mlp.mlp.1", True))
        attentions = ["self_attn"]
        if config["cross_attn"]:
            attentions.append("cross_attn")
            if config["cross_attn_norm"]:
                modules.extend((f"{prefix}.{name}", True) for name in ("cross_attn_norm", "cross_attn_norm_c"))
        for attn in attentions:
            modules.extend(
                (f"{prefix}.{attn}.{name}", config["bias"]) for name in ("to_q", "to_k", "to_v", "to_out.0")
            )
            if config["qk_norm"]:
                keys.extend(f"{prefix}.{attn}.{part}_norm.weight" for part in ("q", "k"))
        modules.extend((f"{prefix}.ffn.ff.{j}", config["bias"]) for j in (0, 3))
    keys.extend(f"{name}.{p}" for name, bias in modules for p in (("weight", "bias") if bias else ("weight",)))
    return Conversion(mapping={key: key for key in keys})
