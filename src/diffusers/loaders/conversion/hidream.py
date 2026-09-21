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


def hidream_conversion(config):
    modules = ["x_embedder.proj", "final_layer.linear", "final_layer.adaLN_modulation.1"]
    modules.extend(
        f"{prefix}.linear_{i}"
        for prefix in ("t_embedder.timestep_embedder", "p_embedder.pooled_embedder")
        for i in (1, 2)
    )
    keys = [
        f"caption_projection.{i}.linear.weight" for i in range(config["num_layers"] + config["num_single_layers"] + 1)
    ]
    for group, count in (
        ("double_stream_blocks", config["num_layers"]),
        ("single_stream_blocks", config["num_single_layers"]),
    ):
        for i in range(count):
            prefix = f"{group}.{i}.block"
            modules.append(prefix + ".adaLN_modulation.1")
            for suffix in ("", "_t") if group == "double_stream_blocks" else ("",):
                modules.extend(f"{prefix}.attn1.{name}{suffix}" for name in ("to_q", "to_k", "to_v", "to_out"))
                keys.extend(f"{prefix}.attn1.{name}{suffix}.weight" for name in ("q_rms_norm", "k_rms_norm"))
            experts = ["ff_i.shared_experts"] + [f"ff_i.experts.{j}" for j in range(config["num_routed_experts"])]
            if group == "double_stream_blocks":
                experts.append("ff_t")
            keys.extend(f"{prefix}.{expert}.w{j}.weight" for expert in experts for j in (1, 2, 3))
            keys.append(prefix + ".ff_i.gate.weight")
    keys.extend(f"{name}.{p}" for name in modules for p in ("weight", "bias"))
    return Conversion(mapping={key: key for key in keys})
