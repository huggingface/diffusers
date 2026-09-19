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
from .transforms import FoldLinearGate, Split


def ltx2_diffusion_decoder_conversion(config):
    """Convert the selected decoder and statistics; gated originals export with canonical unit gates."""
    mapping = {
        "per_channel_statistics.mean-of-means": "latents_mean",
        "per_channel_statistics.std-of-means": "latents_std",
    }
    modules = [(name, name) for name in ("conv_in", "conv_in_x_t", "conv_out", "shared_adaln.proj")]
    modules.extend((f"t_embedder.mlp.{i}", f"t_embedder.timestep_embedder.linear_{j}") for i, j in ((0, 1), (2, 2)))
    modules.extend(
        (f"upsamples.{i}.proj", f"upsamples.{i}.proj") for i in range(len(config["decoder_upsample_strides"]))
    )
    mapping["decoder.norm_out.weight"] = "decoder.norm_out.weight"
    rules = []
    gated = config.get("original_format", "ltx2_diffusion_decoder") == "ltx2_diffusion_decoder_gated"
    blocks = [
        (f"det_stages.{i}.{j}", channels, False)
        for i, channels in enumerate(config["decoder_stage_channels"][:-1])
        for j in range(config["decoder_stage_depths"][i])
    ]
    blocks.extend(
        (f"diff_blocks.{i}", config["decoder_stage_channels"][-1], True)
        for i in range(config["decoder_stage_depths"][-1])
    )
    for name, width, diffusion in blocks:
        prefix = f"decoder.{name}"
        for norm in ("norm1", "norm2"):
            mapping[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
        for kind in ("q", "k"):
            mapping[f"{prefix}.attn.{kind}_norm.weight"] = f"{prefix}.attn.norm_{kind}.weight"
        for p in ("weight", "bias"):
            rules.append(
                Rule(
                    (f"{prefix}.attn.qkv.{p}",),
                    tuple(f"{prefix}.attn.to_{kind}.{p}" for kind in ("q", "k", "v")),
                    Split((width,) * 3),
                )
            )
        for name_mlp in ("w_up", "w_gate"):
            mapping[f"{prefix}.mlp.{name_mlp}.weight"] = f"{prefix}.mlp.{name_mlp}.weight"
        leaves = [("attn.proj", "attn.to_out.0", "gate_msa", True), ("mlp.w_down", "mlp.w_down", "gate_mlp", False)]
        if diffusion:
            mapping[f"{prefix}.scale_shift_table"] = f"{prefix}.scale_shift_table"
            leaves.append(("context_proj", "context_proj", "gate_ctx", True))
        for old, new, gate, bias in leaves:
            parameters = ("weight", "bias") if bias else ("weight",)
            if gated:
                rules.append(
                    Rule(
                        (f"{prefix}.{gate}", *(f"{prefix}.{old}.{p}" for p in parameters)),
                        tuple(f"{prefix}.{new}.{p}" for p in parameters),
                        FoldLinearGate(bias=bias),
                    )
                )
            else:
                mapping.update({f"{prefix}.{old}.{p}": f"{prefix}.{new}.{p}" for p in parameters})
    mapping.update({f"decoder.{old}.{p}": f"decoder.{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping, rules=rules)
