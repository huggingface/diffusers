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


def z_image_controlnet_conversion(config):
    keys, modules = [], []
    modules.extend(
        f"control_all_x_embedder.{p}-{f}" for p, f in zip(config["all_patch_size"], config["all_f_patch_size"])
    )
    blocks = [(f"control_layers.{i}", True, index == 0) for i, index in enumerate(config["control_layers_places"])]
    mode = config["add_control_noise_refiner"]
    if mode != "control_layers":
        blocks.extend(
            (f"control_noise_refiner.{i}", mode == "control_noise_refiner", i == 0)
            for i in range(config["n_refiner_layers"])
        )
    for prefix, controlled, first in blocks:
        names = [
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "attention_norm1",
            "attention_norm2",
            "ffn_norm1",
            "ffn_norm2",
        ]
        if config["qk_norm"]:
            names.extend(["attention.norm_q", "attention.norm_k"])
        keys.extend(f"{prefix}.{name}.weight" for name in names)
        modules.append(prefix + ".adaLN_modulation.0")
        if controlled:
            modules.append(prefix + ".after_proj")
            if first:
                modules.append(prefix + ".before_proj")
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
