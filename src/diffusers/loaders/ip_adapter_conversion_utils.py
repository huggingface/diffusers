# Copyright 2023 The HuggingFace Team. All rights reserved.
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


def _convert_ip_adapter_to_diffusers(state_dict):
    updated_state_dict = {}

    if "proj.weight" in state_dict:
        for key, value in state_dict.items():
            diffusers_name = key.replace("proj", "image_embeds")
            updated_state_dict[diffusers_name] = value

    elif "proj.3.weight" in state_dict:
        for key, value in state_dict.items():
            diffusers_name = key.replace("proj.0", "ff.net.0.proj")
            diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
            diffusers_name = diffusers_name.replace("proj.3", "norm")
            updated_state_dict[diffusers_name] = value

    else:
        for key, value in state_dict.items():
            diffusers_name = key.replace("0.to", "2.to")
            diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
            diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
            diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
            diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

            if "norm1" in diffusers_name:
                updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
            elif "norm2" in diffusers_name:
                updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
            elif "to_kv" in diffusers_name:
                v_chunk = value.chunk(2, dim=0)
                updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
            elif "to_out" in diffusers_name:
                updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
            else:
                updated_state_dict[diffusers_name] = value

    return updated_state_dict
