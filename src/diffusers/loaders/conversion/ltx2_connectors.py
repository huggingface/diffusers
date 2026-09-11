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


def ltx2_connectors_conversion(config):
    mapping = {}
    modalities = ("video", "audio") if config["per_modality_projections"] else ("",)
    for modality in modalities:
        prefix = f"{modality}_" if modality else ""
        for parameter in ("weight", "bias") if config["proj_bias"] else ("weight",):
            mapping[f"text_embedding_projection.{prefix}aggregate_embed.{parameter}"] = (
                f"{prefix}text_proj_in.{parameter}"
            )
    for modality in ("video", "audio"):
        old = f"{modality}_embeddings_connector"
        new = f"{modality}_connector"
        if config[f"{modality}_connector_num_learnable_registers"] is not None:
            mapping[f"{old}.learnable_registers"] = f"{new}.learnable_registers"
        for i in range(config[f"{modality}_connector_num_layers"]):
            source, target = f"{old}.transformer_1d_blocks.{i}", f"{new}.transformer_blocks.{i}"
            modules = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2"]
            if config[f"{modality}_gated_attn"]:
                modules.append("attn1.to_gate_logits")
            mapping.update(
                {f"{source}.{name}.{p}": f"{target}.{name}.{p}" for name in modules for p in ("weight", "bias")}
            )
            for kind in ("q", "k"):
                mapping[f"{source}.attn1.{kind}_norm.weight"] = f"{target}.attn1.norm_{kind}.weight"
    return Conversion(mapping=mapping)
