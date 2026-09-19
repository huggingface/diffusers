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


def ltx2_vocoder_conversion(config):
    mapping = {}
    bwe = config["_class_name"] == "LTX2VocoderWithBWE"
    for prefix in ("vocoder.", "bwe_generator.") if bwe else ("",):
        cfg = (
            {key.removeprefix("bwe_"): value for key, value in config.items() if key.startswith("bwe_")}
            if prefix == "bwe_generator."
            else config
        )
        modules = [("conv_pre", "conv_in"), ("conv_post", "conv_out")]
        for i in range(len(cfg["upsample_factors"])):
            modules.append((f"ups.{i}", f"upsamplers.{i}"))
            for j, dilations in enumerate(cfg["resnet_dilations"]):
                index = i * len(cfg["resnet_dilations"]) + j
                for n in range(len(dilations)):
                    for side in (1, 2):
                        modules.append((f"resblocks.{index}.convs{side}.{n}", f"resnets.{index}.convs{side}.{n}"))
                        old, new = f"resblocks.{index}.acts{side}.{n}", f"resnets.{index}.acts{side}.{n}"
                        if cfg["antialias"]:
                            mapping[f"{prefix}{old}.upsample.filter"] = f"{prefix}{new}.upsample.filter"
                            mapping[f"{prefix}{old}.downsample.lowpass.filter"] = f"{prefix}{new}.downsample.filter"
                            old, new = old + ".act", new + ".act"
                        if cfg["act_fn"] in ("snake", "snakebeta"):
                            for p in ("alpha", "beta") if cfg["act_fn"] == "snakebeta" else ("alpha",):
                                mapping[f"{prefix}{old}.{p}"] = f"{prefix}{new}.{p}"
        for old, new in modules:
            for p in ("weight",) if new == "conv_out" and not cfg["final_bias"] else ("weight", "bias"):
                mapping[f"{prefix}{old}.{p}"] = f"{prefix}{new}.{p}"
        if cfg["act_fn"] in ("snake", "snakebeta"):
            for p in ("alpha", "beta"):
                mapping[f"{prefix}act_post.act.{p}"] = f"{prefix}act_out.act.{p}"
            mapping[f"{prefix}act_post.upsample.filter"] = f"{prefix}act_out.upsample.filter"
            mapping[f"{prefix}act_post.downsample.lowpass.filter"] = f"{prefix}act_out.downsample.filter"
    if bwe:
        for key in ("mel_stft.mel_basis", "mel_stft.stft_fn.forward_basis", "mel_stft.stft_fn.inverse_basis"):
            mapping[key] = key
    return Conversion(mapping=mapping)
