# Copyright 2024 AniMemory Team and The HuggingFace Team. All rights reserved.
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

import os

import torch
from safetensors.torch import load_file
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.configuration_t5 import T5Config
from transformers import CLIPTextModelWithProjection, CLIPTextConfig


class AniMemoryT5(torch.nn.Module):
    def __init__(self, config: T5Config, embed_tokens=None):
        super().__init__()
        self.encoder = T5Stack(config, embed_tokens)
        self.embed_tokens_encoder = torch.nn.Embedding(250002, 4096, padding_idx=1)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        subfolder="",
        embed_tokens=None,
        emb_name='weights.safetensors', 
        torch_dtype=torch.float16,
    ):

        cls.dtype = torch_dtype
        config = T5Stack.config_class.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder)
        model = cls(config=config, embed_tokens=embed_tokens)
        model.encoder = T5Stack.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        embed_tokens_encoder_path = load_file(os.path.join(pretrained_model_name_or_path, subfolder, emb_name))
        model.embed_tokens_encoder.load_state_dict(embed_tokens_encoder_path)
        model.encoder.to(torch_dtype)
        model.embed_tokens_encoder.to(torch_dtype)
        return model

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        super(AniMemoryT5, self).to(*args, **kwargs)
        self.dtype = dtype if dtype is not None else self.dtype
        self.device = device if device is not None else self.device
        return self

    def make_attn_mask(self, attn_mask):
        seq_len = attn_mask.shape[1]
        query = attn_mask.unsqueeze(1).float()
        attn_mask = query.repeat([1, seq_len, 1]).unsqueeze(1).repeat([1, self.num_head, 1, 1])
        attn_mask = attn_mask.view([-1, seq_len, seq_len])
        return attn_mask

    def forward(self, text, attention_mask):
        embeddings = self.embed_tokens_encoder(text)
        encoder_outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states[-2]
        hidden_states = self.encoder.final_layer_norm(hidden_states)
        return hidden_states, hidden_states


class AniMemoryAltCLip(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.model_hf = CLIPTextModelWithProjection(config)
        self.linear_proj = torch.nn.Linear(in_features=1280, out_features=1280)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        subfolder="",
        linear_proj_name="weights.safetensors",
        torch_dtype=torch.float16,
    ):
        cls.dtype = torch_dtype
        config = CLIPTextModelWithProjection.config_class.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        model = cls(config=config)
        model.model_hf = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        linear_proj_state = load_file(os.path.join(pretrained_model_name_or_path, subfolder, linear_proj_name))
        model.linear_proj.load_state_dict(linear_proj_state)
        return model

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        super(AniMemoryAltCLip, self).to(*args, **kwargs)
        self.dtype = dtype if dtype is not None else self.dtype
        self.device = device if device is not None else self.device
        return self

    def expand_mask(self, mask=None, dtype="", tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def make_attn_mask(self, attn_mask):
        seq_len = attn_mask.shape[1]
        query = attn_mask.unsqueeze(1).float()
        attn_mask = query.repeat([1, seq_len, 1]).unsqueeze(
            1).repeat([1, self.num_head, 1, 1])
        attn_mask = attn_mask.view([-1, seq_len, seq_len])
        return attn_mask

    def gradient_checkpointing_enable(self,):
        self.model_hf.gradient_checkpointing_enable()

    def forward(self, text, attention_mask):
        hidden_states = self.model_hf.text_model.embeddings(
             input_ids=text, position_ids=None)
        if attention_mask is None:
            print('Warning: attention_mask is None in altclip!')
        new_attn_mask = self.expand_mask(attention_mask, hidden_states.dtype) if not attention_mask is None else None
        encoder_outputs = self.model_hf.text_model.encoder(
           inputs_embeds=hidden_states,
           attention_mask=new_attn_mask,
           causal_attention_mask=None,
           output_attentions=False,
           output_hidden_states=True,
           return_dict=True,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.model_hf.text_model.final_layer_norm(last_hidden_state)
        last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), 0] @ self.model_hf.text_projection.weight
        pooled_output = self.linear_proj(last_hidden_state)

        extra_features = encoder_outputs.hidden_states[-2]
        extra_features = self.model_hf.text_model.final_layer_norm(extra_features)
        return extra_features, pooled_output
