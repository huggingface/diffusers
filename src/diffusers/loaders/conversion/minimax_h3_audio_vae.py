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


def minimax_h3_audio_vae_conversion(config):
    modules = [
        "mean_proj",
        "logs_proj",
        "dec_in_proj",
        "pre_block.norm1",
        "pre_block.norm2",
        "pre_block.norm3",
        "pre_block.proj",
        "pre_block.attn.proj",
        "pre_block.mlp.norm",
        "pre_block.mlp.w0",
        "pre_block.mlp.w1",
        "pre_block.mlp.w2",
    ]
    keys = [
        "pre_block.attn.qkv.weight",
        "pre_block.attn.q_bias",
        "pre_block.attn.v_bias",
        "pre_block.attn.zero_k_bias",
    ]
    count = len(config["encoder_rates"])
    convolutions = [
        ("encoder.block.0", True),
        (f"encoder.block.{count + 2}", True),
        ("decoder.conv_pre", True),
        ("decoder.conv_post", False),
    ]
    keys.append(f"encoder.block.{count + 1}.alpha")
    for i in range(count):
        prefix = f"encoder.block.{i + 1}.block"
        keys.append(prefix + ".3.alpha")
        convolutions.append((prefix + ".4", True))
        for j in range(3):
            unit = f"{prefix}.{j}.block"
            keys.extend(unit + f".{k}.alpha" for k in (0, 2))
            convolutions.extend((unit + f".{k}", True) for k in (1, 3))
    activations = ["decoder.activation_post"]
    kernels = len(config["resblock_kernel_sizes"])
    for i in range(len(config["decoder_rates"])):
        convolutions.append((f"decoder.ups.{i}.0", True))
        for j in range(kernels):
            prefix = f"decoder.resblocks.{i * kernels + j}"
            depth = len(config["resblock_dilation_sizes"][j])
            convolutions.extend((f"{prefix}.convs{part}.{k}", True) for part in (1, 2) for k in range(depth))
            activations.extend(f"{prefix}.activations.{k}" for k in range(2 * depth))
    keys.extend(
        f"{prefix}.{name}"
        for prefix in activations
        for name in ("act.alpha", "act.beta", "upsample.filter", "downsample.lowpass.filter")
    )
    keys.extend(
        f"{name}.{p}"
        for name, bias in convolutions
        for p in (("weight_g", "weight_v", "bias") if bias else ("weight_g", "weight_v"))
    )
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
