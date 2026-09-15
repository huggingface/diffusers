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
from .ldm_vae import ldm_vae_conversion


def svd_vae_conversion(config):
    original_format = config.get("original_format", "svd")
    if original_format not in ("svd", "temporal_vae"):
        raise ValueError("Temporal VAE original_format must be 'svd' or 'temporal_vae'.")
    encoder_prefix = "conditioner.embedders.3.encoder." if original_format == "svd" else ""
    decoder_prefix = "first_stage_model." if original_format == "svd" else ""
    base = ldm_vae_conversion(
        {
            **config,
            "up_block_types": ["UpDecoderBlock2D"] * len(config["block_out_channels"]),
            "mid_block_add_attention": True,
            "use_post_quant_conv": False,
        }
    )
    mapping, rules, resnets = {}, [], []
    for old, new in base.mapping.items():
        prefix = decoder_prefix if new.startswith("decoder.") else encoder_prefix
        if new.startswith("decoder.mid_block.resnets."):
            continue
        if new.startswith("decoder.") and ".resnets." in new:
            parent, suffix = new.rsplit(".resnets.", 1)
            index, suffix = suffix.split(".", 1)
            parent += ".resnets." + index
            mapping[prefix + old] = parent + ".spatial_res_block." + suffix
            if suffix == "norm1.weight":
                resnets.append((prefix + old.removesuffix(".norm1.weight"), parent))
        else:
            mapping[prefix + old] = new
    for i in range(config["layers_per_block"]):
        old, new = f"{decoder_prefix}decoder.mid.block_{i + 1}", f"decoder.mid_block.resnets.{i}"
        resnets.append((old, new))
        mapping.update(
            {
                f"{old}.{name}.{p}": f"{new}.spatial_res_block.{name}.{p}"
                for name in ("norm1", "norm2", "conv1", "conv2")
                for p in ("weight", "bias")
            }
        )
    for rule in base.rules:
        prefix = decoder_prefix if rule.diffusers[0].startswith("decoder.") else encoder_prefix
        rules.append(Rule(tuple(prefix + key for key in rule.original), rule.diffusers, rule.transform))
    for old, new in resnets:
        mapping[old + ".mix_factor"] = new + ".time_mixer.mix_factor"
        for a, b in (
            ("in_layers.0", "norm1"),
            ("in_layers.2", "conv1"),
            ("out_layers.0", "norm2"),
            ("out_layers.3", "conv2"),
        ):
            mapping.update(
                {f"{old}.time_stack.{a}.{p}": f"{new}.temporal_res_block.{b}.{p}" for p in ("weight", "bias")}
            )
    mapping.update(
        {
            f"{decoder_prefix}decoder.conv_out.time_mix_conv.{p}": f"decoder.time_conv_out.{p}"
            for p in ("weight", "bias")
        }
    )
    return Conversion(mapping=mapping, rules=tuple(rules))
