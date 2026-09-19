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
from .transforms import Permute, Split


def clip_vision_conversion(config):
    """Convert CLIP/HF or OpenCLIP visual tower weights, including optional projection."""
    projected = config["_class_name"] == "CLIPVisionModelWithProjection"
    prefix = (
        ""
        if not projected and int((config.get("transformers_version") or "4").split(".")[0]) >= 5
        else "vision_model."
    )
    original_format = config.get("original_format", "clip")
    if original_format not in ("clip", "openclip"):
        raise ValueError("CLIP vision original_format must be clip or openclip.")
    modules = [("ln_pre", "pre_layrnorm"), ("ln_post", "post_layernorm")]
    pairs = [
        ("class_embedding", "embeddings.class_embedding"),
        ("positional_embedding", "embeddings.position_embedding.weight"),
        ("conv1.weight", "embeddings.patch_embedding.weight"),
    ]
    rules = []
    for i in range(config["num_hidden_layers"]):
        old, new = f"transformer.resblocks.{i}", f"encoder.layers.{i}"
        modules.extend(
            (f"{old}.{a}", f"{new}.{b}")
            for a, b in (
                ("ln_1", "layer_norm1"),
                ("ln_2", "layer_norm2"),
                ("mlp.c_fc", "mlp.fc1"),
                ("mlp.c_proj", "mlp.fc2"),
                ("attn.out_proj", "self_attn.out_proj"),
            )
        )
        for p in ("weight", "bias"):
            if original_format == "openclip":
                rules.append(
                    Rule(
                        (f"{old}.attn.in_proj_{p}",),
                        tuple(f"{prefix}{new}.self_attn.{part}_proj.{p}" for part in ("q", "k", "v")),
                        Split((config["hidden_size"],) * 3),
                    )
                )
            else:
                pairs.extend(("", f"{new}.self_attn.{part}_proj.{p}") for part in ("q", "k", "v"))
    pairs.extend((f"{old}.{p}", f"{new}.{p}") for old, new in modules for p in ("weight", "bias"))
    mapping = {(old if original_format == "openclip" else "vision_model." + new): prefix + new for old, new in pairs}
    if projected:
        if original_format == "openclip":
            rules.append(Rule(("proj",), ("visual_projection.weight",), Permute((1, 0))))
        else:
            mapping["visual_projection.weight"] = "visual_projection.weight"
    return Conversion(mapping=mapping, rules=rules)
