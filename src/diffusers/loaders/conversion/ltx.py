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


def ltx_conversion(config):
    mapping = {"scale_shift_table": "scale_shift_table"}
    modules = [("patchify_proj", "proj_in"), ("proj_out", "proj_out")]
    modules.extend(
        ("adaln_single." + name, "time_embed." + name)
        for name in ("emb.timestep_embedder.linear_1", "emb.timestep_embedder.linear_2", "linear")
    )
    modules.extend(("caption_projection." + name, "caption_projection." + name) for name in ("linear_1", "linear_2"))
    for i in range(config["num_layers"]):
        prefix = f"transformer_blocks.{i}"
        mapping[prefix + ".scale_shift_table"] = prefix + ".scale_shift_table"
        modules.extend((prefix + ".ff." + name, prefix + ".ff." + name) for name in ("net.0.proj", "net.2"))
        for attn in ("attn1", "attn2"):
            for name in ("to_q", "to_k", "to_v", "to_out.0"):
                bias = config["attention_out_bias"] if name == "to_out.0" else config["attention_bias"]
                for p in ("weight", "bias") if bias else ("weight",):
                    key = f"{prefix}.{attn}.{name}.{p}"
                    mapping[key] = key
            if config["qk_norm"] is not None:
                for part in ("q", "k"):
                    mapping[f"{prefix}.{attn}.{part}_norm.weight"] = f"{prefix}.{attn}.norm_{part}.weight"
        if config["norm_elementwise_affine"]:
            for name in ("norm1", "norm2"):
                mapping[f"{prefix}.{name}.weight"] = f"{prefix}.{name}.weight"
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
