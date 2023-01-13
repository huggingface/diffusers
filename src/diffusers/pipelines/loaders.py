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
from ..models.modeling_utils import _get_model_file
from collections import defaultdict
import torch
from ..utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    logging,
)


logger = logging.get_logger(__name__)


LORA_WEIGHT_NAME = "pytorch_lora.bin"


class LoraUNetLoaderMixin:
    def load_lora(self, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)

        user_agent = {
            "file_type": "lora",
            "framework": "pytorch",
        }

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )

        model_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=LORA_WEIGHT_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )

        state_dict = torch.load(model_file, map_location="cpu")
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

            if torch_dtype is not None:
                attn_processors[key].to(torch_dtype)

        self.unet.set_attn_processor(attn_processors)
