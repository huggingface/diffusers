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
from .transforms import Reverse, Split


def uvit_conversion(config):
    mapping = {"embed.embeddings.weight": "embed.embeddings.weight"}
    modules = [
        (name, name)
        for name in (
            "encoder_proj",
            "embed.conv",
            "project_to_hidden",
            "project_from_hidden",
            "mlm_layer.conv1",
            "mlm_layer.conv2",
        )
    ]
    modules.extend([("cond_embed.0", "cond_embed.linear_1"), ("cond_embed.2", "cond_embed.linear_2")])
    norms = [
        (name, name)
        for name in (
            "encoder_proj_layer_norm",
            "embed.layer_norm",
            "project_to_hidden_norm",
            "project_from_hidden_norm",
        )
    ]
    norms.append(("mlm_layer.layer_norm.norm", "mlm_layer.layer_norm"))
    rules = []
    for i in range(config["num_hidden_layers"]):
        prefix = f"transformer_layers.{i}"
        for source, target in (
            ("attn_layer_norm", "norm1.norm"),
            ("crossattn_layer_norm", "norm2.norm"),
            ("ffn.pre_mlp_layer_norm", "norm3.norm"),
        ):
            norms.append((f"{prefix}.{source}", f"{prefix}.{target}"))
        modules.extend(
            (f"{prefix}.{source}.mapper", f"{prefix}.norm{j}.linear")
            for j, source in (
                (1, "self_attn_adaLN_modulation"),
                (2, "cross_attn_adaLN_modulation"),
                (3, "ffn.adaLN_modulation"),
            )
        )
        for old, new in (("attention", "attn1"), ("crossattention", "attn2")):
            modules.extend(
                (f"{prefix}.{old}.{a}", f"{prefix}.{new}.{b}")
                for a, b in (("query", "to_q"), ("key", "to_k"), ("value", "to_v"), ("out", "to_out.0"))
            )
        modules.append((prefix + ".ffn.wo", prefix + ".ff.net.2"))
        for p in ("weight", "bias") if config["use_bias"] else ("weight",):
            rules.append(
                Rule(
                    (f"{prefix}.ffn.wi_1.{p}", f"{prefix}.ffn.wi_0.{p}"),
                    (f"{prefix}.ff.net.0.proj.{p}",),
                    Reverse(Split((config["intermediate_size"],) * 2)),
                )
            )
    for direction in ("down", "up"):
        old, new = f"{direction}_blocks.0", f"{direction}_block"
        resampling = direction + "sample"
        if config[resampling]:
            modules.append((f"{old}.{resampling}.1", f"{new}.{resampling}.conv"))
            norms.append((f"{old}.{resampling}.0.norm", f"{new}.{resampling}.norm"))
        for i in range(config["num_res_blocks"]):
            a, b = f"{old}.res_blocks.{i}", f"{new}.res_blocks.{i}"
            modules.extend(
                (f"{a}.{source}", f"{b}.{target}")
                for source, target in (
                    ("depthwise", "depthwise"),
                    ("channelwise.0", "channelwise_linear_1"),
                    ("channelwise.4", "channelwise_linear_2"),
                    ("adaLN_modulation.mapper", "cond_embeds_mapper"),
                )
            )
            norms.append((a + ".norm.norm", b + ".norm"))
            mapping.update({f"{a}.channelwise.2.{p}": f"{b}.channelwise_norm.{p}" for p in ("gamma", "beta")})
            a, b = f"{old}.attention_blocks.{i}", f"{new}.attention_blocks.{i}"
            mapping.update(
                {
                    f"{a}.{source}.weight": f"{b}.{target}.weight"
                    for source, target in (("attn_layer_norm", "norm1"), ("crossattn_layer_norm", "norm2"))
                }
            )
            if config["hidden_size"] != config["block_out_channels"]:
                modules.append((a + ".kv_mapper", b + ".kv_mapper"))
            for source, target in (("attention", "attn1"), ("crossattention", "attn2")):
                modules.extend(
                    (f"{a}.{source}.{x}", f"{b}.{target}.{y}")
                    for x, y in (("query", "to_q"), ("key", "to_k"), ("value", "to_v"), ("out", "to_out.0"))
                )
    mapping.update(
        {
            f"{old}.{p}": f"{new}.{p}"
            for old, new in modules
            for p in (("weight", "bias") if config["use_bias"] else ("weight",))
        }
    )
    if config["ln_elementwise_affine"]:
        mapping.update({old + ".weight": new + ".weight" for old, new in norms})
    return Conversion(mapping=mapping, rules=tuple(rules))
