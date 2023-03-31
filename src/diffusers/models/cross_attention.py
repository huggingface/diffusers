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
from typing import Union

import torch
import torch.nn.functional as F

from ..utils import deprecate
from .attention_processor import (  # noqa: F401
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRALinearLayer,
    LoRAXFormersAttnProcessor,
    SlicedAttnAddedKVProcessor,
    SlicedAttnProcessor,
    XFormersAttnProcessor,
    TuneAVideoAttnProcessor
)
from .attention_processor import AttnProcessor as AttnProcessorRename  # noqa: F401


deprecate(
    "cross_attention",
    "0.18.0",
    "Importing from cross_attention is deprecated. Please import from diffusers.models.attention_processor instead.",
    standard_warn=False,
)


AttnProcessor = AttentionProcessor


class CrossAttention(Attention):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class CrossAttnProcessor(AttnProcessorRename):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class LoRACrossAttnProcessor(LoRAAttnProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class CrossAttnAddedKVProcessor(AttnAddedKVProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class XFormersCrossAttnProcessor(XFormersAttnProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class LoRAXFormersCrossAttnProcessor(LoRAXFormersAttnProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class SlicedCrossAttnProcessor(SlicedAttnProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class TuneAVideoCrossAttnProcessor(TuneAVideoAttnProcessor):
    # def __call__(
    #     self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None
    # ):
    #     batch_size, sequence_length, _ = hidden_states.shape

    #     encoder_hidden_states = encoder_hidden_states

    #     if attn.group_norm is not None:
    #         hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    #     query = attn.to_q(hidden_states)
    #     query.shape[-1]
    #     query = attn.head_to_batch_dim(query)

    #     if attn.added_kv_proj_dim is not None:
    #         raise NotImplementedError

    #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    #     key = attn.to_k(encoder_hidden_states)
    #     value = attn.to_v(encoder_hidden_states)

    #     former_frame_index = torch.arange(video_length) - 1
    #     former_frame_index[0] = 0

    #     # key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    #     key = key.reshape([-1, video_length, *key.shape[1:]])
    #     key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
    #     # key = rearrange(key, "b f d c -> (b f) d c")
    #     key = key.flatten(0, 1)

    #     # value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    #     value = value.reshape([-1, video_length, *value.shape[1:]])
    #     value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
    #     # value = rearrange(value, "b f d c -> (b f) d c")
    #     value = value.flatten(0, 1)

    #     key = attn.head_to_batch_dim(key)
    #     value = attn.head_to_batch_dim(value)

    #     if attention_mask is not None:
    #         if attention_mask.shape[-1] != query.shape[1]:
    #             target_length = query.shape[1]
    #             attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
    #             attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

    #     # attention, what we cannot get enough of
    #     # if self._use_memory_efficient_attention_xformers:
    #     #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
    #     #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
    #     #     hidden_states = hidden_states.to(query.dtype)
    #     # else:
    #     #     if attn._slice_size is None or query.shape[0] // self._slice_size == 1:
    #     #         hidden_states = self._attention(query, key, value, attention_mask)
    #     #     else:
    #     #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

    #     attention_probs = attn.get_attention_scores(query, key, attention_mask)
    #     hidden_states = torch.bmm(attention_probs, value)
    #     hidden_states = attn.batch_to_head_dim(hidden_states)

    #     # linear proj
    #     hidden_states = attn.to_out[0](hidden_states)

    #     # dropout
    #     hidden_states = attn.to_out[1](hidden_states)
    #     return hidden_states
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


class SlicedCrossAttnAddedKVProcessor(SlicedAttnAddedKVProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = f"{self.__class__.__name__} is deprecated and will be removed in `0.18.0`. Please use `from diffusers.models.attention_processor import {''.join(self.__class__.__name__.split('Cross'))} instead."
        deprecate("cross_attention", "0.18.0", deprecation_message, standard_warn=False)
        super().__init__(*args, **kwargs)


AttnProcessor = Union[
    CrossAttnProcessor,
    TuneAVideoCrossAttnProcessor,
    XFormersCrossAttnProcessor,
    SlicedAttnProcessor,
    CrossAttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    LoRACrossAttnProcessor,
    LoRAXFormersCrossAttnProcessor,
]
