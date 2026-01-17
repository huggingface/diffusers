# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch


def build_prompt_embeds_and_mask(split_hidden_states):
    seq_lens = [e.size(0) for e in split_hidden_states]
    max_seq_len = max(seq_lens)
    if all(seq_len == max_seq_len for seq_len in seq_lens):
        prompt_embeds = torch.stack(split_hidden_states)
        return prompt_embeds, None

    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    return prompt_embeds, encoder_attention_mask


def slice_prompt_embeds_and_mask(prompt_embeds, prompt_embeds_mask, max_sequence_length):
    prompt_embeds = prompt_embeds[:, :max_sequence_length]
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
    return prompt_embeds, prompt_embeds_mask


def repeat_prompt_embeds_and_mask(prompt_embeds, prompt_embeds_mask, num_images_per_prompt):
    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
    return prompt_embeds, prompt_embeds_mask


def concat_prompt_embeds_for_cfg(
    prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask
):
    pos_len = prompt_embeds.shape[1]
    neg_len = negative_prompt_embeds.shape[1]
    max_len = max(pos_len, neg_len)

    def _pad_prompt(embeds, mask):
        orig_len = embeds.shape[1]
        if orig_len != max_len:
            pad_len = max_len - orig_len
            embeds = torch.cat([embeds, embeds.new_zeros(embeds.shape[0], pad_len, embeds.shape[2])], dim=1)
        if mask is None and orig_len != max_len:
            mask = torch.ones((embeds.shape[0], orig_len), dtype=torch.long, device=embeds.device)
        if mask is not None and mask.shape[1] != max_len:
            pad_len = max_len - mask.shape[1]
            mask = torch.cat([mask, mask.new_zeros(mask.shape[0], pad_len)], dim=1)
        return embeds, mask

    prompt_embeds, prompt_embeds_mask = _pad_prompt(prompt_embeds, prompt_embeds_mask)
    negative_prompt_embeds, negative_prompt_embeds_mask = _pad_prompt(
        negative_prompt_embeds, negative_prompt_embeds_mask
    )

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    if prompt_embeds_mask is None and negative_prompt_embeds_mask is None:
        prompt_embeds_mask = None
    else:
        batch_half = prompt_embeds.shape[0] // 2
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones((batch_half, max_len), dtype=torch.long, device=prompt_embeds.device)
        if negative_prompt_embeds_mask is None:
            negative_prompt_embeds_mask = torch.ones(
                (batch_half, max_len), dtype=torch.long, device=prompt_embeds.device
            )
        prompt_embeds_mask = torch.cat([negative_prompt_embeds_mask, prompt_embeds_mask], dim=0)

    if prompt_embeds_mask is not None and prompt_embeds_mask.all():
        prompt_embeds_mask = None

    return prompt_embeds, prompt_embeds_mask
