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

from .clap_text import clap_text_conversion
from .core import Conversion, Rule
from .transforms import Split


def clap_audio_conversion(config):
    full = config["_class_name"] == "ClapModel"
    cfg = config["audio_config"] if full else config
    projected = config["_class_name"] != "ClapAudioModel"
    prefix = "audio_model.audio_encoder" if projected else "audio_encoder"
    mapping, modules, rules = (
        {},
        [
            ("audio_branch.patch_embed.proj", f"{prefix}.patch_embed.proj"),
            ("audio_branch.norm", f"{prefix}.norm"),
            ("audio_branch.bn0", f"{prefix}.batch_norm"),
        ],
        [],
    )
    if cfg["enable_patch_layer_norm"]:
        modules.append(("audio_branch.patch_embed.norm", f"{prefix}.patch_embed.norm"))
    for p in ("running_mean", "running_var", "num_batches_tracked"):
        mapping[f"audio_branch.bn0.{p}"] = f"{prefix}.batch_norm.{p}"
    if cfg["enable_fusion"]:
        modules.append(("audio_branch.patch_embed.mel_conv2d", f"{prefix}.patch_embed.mel_conv2d"))
        for kind, indices, bns in (("local_att", (0, 1, 3, 4), (1, 4)), ("global_att", (1, 2, 4, 5), (2, 5))):
            for i in indices:
                old, new = (
                    f"audio_branch.patch_embed.fusion_model.{kind}.{i}",
                    f"{prefix}.patch_embed.fusion_model.{kind}.{i}",
                )
                modules.append((old, new))
                if i in bns:
                    mapping.update(
                        {f"{old}.{p}": f"{new}.{p}" for p in ("running_mean", "running_var", "num_batches_tracked")}
                    )
    for i, depth in enumerate(cfg["depths"]):
        width = cfg["patch_embeds_hidden_size"] * 2**i
        for j in range(depth):
            old, new = f"audio_branch.layers.{i}.blocks.{j}", f"{prefix}.layers.{i}.blocks.{j}"
            modules.extend(
                (f"{old}.{a}", f"{new}.{b}")
                for a, b in (
                    ("norm1", "layernorm_before"),
                    ("norm2", "layernorm_after"),
                    ("attn.proj", "attention.output.dense"),
                    ("mlp.fc1", "intermediate.dense"),
                    ("mlp.fc2", "output.dense"),
                )
            )
            for name in ("relative_position_bias_table", "relative_position_index"):
                mapping[f"{old}.attn.{name}"] = f"{new}.attention.self.{name}"
            for p in ("weight", "bias") if cfg["qkv_bias"] else ("weight",):
                rules.append(
                    Rule(
                        (f"{old}.attn.qkv.{p}",),
                        tuple(f"{new}.attention.self.{part}.{p}" for part in ("query", "key", "value")),
                        Split((width,) * 3),
                    )
                )
        if i < len(cfg["depths"]) - 1:
            old, new = f"audio_branch.layers.{i}.downsample", f"{prefix}.layers.{i}.downsample"
            modules.append((f"{old}.norm", f"{new}.norm"))
            mapping[f"{old}.reduction.weight"] = f"{new}.reduction.weight"
    if projected:
        modules.extend((f"audio_projection.{i}", f"audio_projection.linear{j}") for i, j in ((0, 1), (2, 2)))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    if full:
        text = clap_text_conversion(config["text_config"])
        mapping.update(text.mapping)
        rules.extend(text.rules)
        mapping.update({name: name for name in ("logit_scale_a", "logit_scale_t")})
    return Conversion(mapping=mapping, rules=rules)
