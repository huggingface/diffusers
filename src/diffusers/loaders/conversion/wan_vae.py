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


def wan_vae_conversion(config):
    residual = config["is_residual"]
    mapping = {}
    modules = [("conv1", "quant_conv"), ("conv2", "post_quant_conv")]
    resnets, attentions = [], []
    for component in ("encoder", "decoder"):
        modules.extend(
            [(f"{component}.conv1", f"{component}.conv_in"), (f"{component}.head.2", f"{component}.conv_out")]
        )
        mapping[f"{component}.head.0.gamma"] = f"{component}.norm_out.gamma"
        resnets.extend((f"{component}.middle.{i * 2}", f"{component}.mid_block.resnets.{i}", False) for i in range(2))
        attentions.append((f"{component}.middle.1", f"{component}.mid_block.attentions.0"))
    dims = [config["base_dim"] * n for n in [1] + list(config["dim_mult"])]
    idx, scale = 0, 1.0
    for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
        for j in range(config["num_res_blocks"]):
            if residual:
                old, new = f"encoder.downsamples.{i}.downsamples.{j}", f"encoder.down_blocks.{i}.resnets.{j}"
            else:
                old, new = f"encoder.downsamples.{idx}", f"encoder.down_blocks.{idx}"
            resnets.append((old, new, input_dim != output_dim))
            input_dim = output_dim
            idx += 1
            if not residual and scale in config["attn_scales"]:
                attentions.append((f"encoder.downsamples.{idx}", f"encoder.down_blocks.{idx}"))
                idx += 1
        if i < len(dims) - 2:
            if residual:
                old = f"encoder.downsamples.{i}.downsamples.{config['num_res_blocks']}"
                new = f"encoder.down_blocks.{i}.downsampler"
            else:
                old, new = f"encoder.downsamples.{idx}", f"encoder.down_blocks.{idx}"
            modules.append((old + ".resample.1", new + ".resample.1"))
            if config["temperal_downsample"][i]:
                modules.append((old + ".time_conv", new + ".time_conv"))
            idx += 1
            scale /= 2
    decoder_dim = config["decoder_base_dim"] or config["base_dim"]
    dims = [decoder_dim * n for n in [config["dim_mult"][-1]] + list(reversed(config["dim_mult"]))]
    temporal = list(reversed(config["temperal_downsample"]))
    idx = 0
    for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
        if i > 0 and not residual:
            input_dim //= 2
        for j in range(config["num_res_blocks"] + 1):
            resnets.append(
                (
                    f"decoder.upsamples.{i}.upsamples.{j}" if residual else f"decoder.upsamples.{idx}",
                    f"decoder.up_blocks.{i}.resnets.{j}",
                    input_dim != output_dim,
                )
            )
            input_dim = output_dim
            idx += 1
        if i < len(dims) - 2:
            if residual:
                old = f"decoder.upsamples.{i}.upsamples.{config['num_res_blocks'] + 1}"
                new = f"decoder.up_blocks.{i}.upsampler"
            else:
                old, new = f"decoder.upsamples.{idx}", f"decoder.up_blocks.{i}.upsamplers.0"
            modules.append((old + ".resample.1", new + ".resample.1"))
            if temporal[i]:
                modules.append((old + ".time_conv", new + ".time_conv"))
            idx += 1
    for old, new, shortcut in resnets:
        mapping[old + ".residual.0.gamma"] = new + ".norm1.gamma"
        mapping[old + ".residual.3.gamma"] = new + ".norm2.gamma"
        modules.extend([(old + ".residual.2", new + ".conv1"), (old + ".residual.6", new + ".conv2")])
        if shortcut:
            modules.append((old + ".shortcut", new + ".conv_shortcut"))
    for old, new in attentions:
        mapping[old + ".norm.gamma"] = new + ".norm.gamma"
        modules.extend((old + "." + name, new + "." + name) for name in ("to_qkv", "proj"))
    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping=mapping)
