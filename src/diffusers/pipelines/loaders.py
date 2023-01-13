# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from ..models.cross_attention import LoRACrossAttnProcessor
from collections import defaultdict
import torch


class LoraUNetLoaderMixin:
    def load_lora(self, pretrained_model_name_or_path):
        state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value

        attn_processors = {}
        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict["to_k_lora.lora_down.weight"].shape[0]
            cross_attention_dim = value_dict["to_k_lora.lora_down.weight"].shape[1]
            hidden_size = value_dict["to_k_lora.lora_up.weight"].shape[0]

            attn_processors[key] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
            )
            attn_processors[key].load_state_dict(value_dict)

        self.unet.set_attn_processor(attn_processors)
