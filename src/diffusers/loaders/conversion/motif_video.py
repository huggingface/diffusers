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


def motif_video_conversion(config):
    """Validate the native Motif Video checkpoint layout used by its single-file loader."""
    modules = [
        "x_embedder.proj",
        "context_embedder.linear_1",
        "context_embedder.linear_2",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
        "norm_out.linear",
        "proj_out",
    ]
    keys = []
    if config["image_embed_dim"] is not None:
        modules.extend(f"image_embedder.{name}" for name in ("norm_in", "linear_1", "linear_2", "norm_out"))
    for group, count in (
        ("transformer_blocks", config["num_layers"]),
        ("single_transformer_blocks", config["num_single_layers"]),
    ):
        dual = group == "transformer_blocks"
        for i in range(count):
            prefix = f"{group}.{i}"
            names = (
                (
                    "norm1.linear",
                    "norm1_context.linear",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                )
                if dual
                else ("norm.linear", "proj_mlp", "proj_out")
            )
            modules.extend(f"{prefix}.{name}" for name in names)
            cross = (
                config["enable_text_cross_attention_dual"]
                if dual
                else (config["enable_text_cross_attention_single"] and i < count - config["num_decoder_layers"])
            )
            for attention in ("attn", "cross_attn") if cross else ("attn",):
                names = ["to_q", "to_k", "to_v"]
                if dual or attention == "cross_attn":
                    names.append("to_out.0")
                if dual and attention == "attn":
                    names.extend(("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"))
                    keys.extend(f"{prefix}.attn.norm_added_{part}.weight" for part in ("q", "k"))
                modules.extend(f"{prefix}.{attention}.{name}" for name in names)
                if config["qk_norm"] in ("rms_norm", "layer_norm"):
                    parameters = ("weight", "bias") if config["qk_norm"] == "layer_norm" else ("weight",)
                    keys.extend(f"{prefix}.{attention}.norm_{part}.{p}" for part in ("q", "k") for p in parameters)
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
