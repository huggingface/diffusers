"""
# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.
#
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from diffusers.models.embeddings import get_1d_rotary_pos_embed


class BooguImageRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(
        axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int], theta: int
    ) -> List[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis,
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Create position IDs
        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add text position ids
            position_ids[i, :cap_seq_len] = (
                torch.arange(cap_seq_len, dtype=torch.int32, device=device).unsqueeze(1).expand(-1, 3)
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len
                    # add image position ids

                    row_ids = (
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device)
                        .unsqueeze(1)
                        .expand(ref_H_tokens, ref_W_tokens)
                        .flatten()
                    )
                    col_ids = (
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device)
                        .unsqueeze(0)
                        .expand(ref_H_tokens, ref_W_tokens)
                        .flatten()
                    )
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = (
                torch.arange(H_tokens, dtype=torch.int32, device=device)
                .unsqueeze(1)
                .expand(H_tokens, W_tokens)
                .flatten()
            )
            col_ids = (
                torch.arange(W_tokens, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .expand(H_tokens, W_tokens)
                .flatten()
            )

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size,
            encoder_seq_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size,
            max_ref_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        img_freqs_cis = torch.zeros(
            batch_size,
            max_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(
                l_effective_cap_len,
                l_effective_ref_img_len,
                l_effective_img_len,
                seq_lengths,
            )
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len,
            ]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
        )


class BooguImageDoubleStreamRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(
        axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int], theta: int
    ) -> List[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis,
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Create position IDs
        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add text position ids
            position_ids[i, :cap_seq_len] = (
                torch.arange(cap_seq_len, dtype=torch.int32, device=device).unsqueeze(1).expand(-1, 3)
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len
                    # add image position ids

                    row_ids = (
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device)
                        .unsqueeze(1)
                        .expand(ref_H_tokens, ref_W_tokens)
                        .flatten()
                    )
                    col_ids = (
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device)
                        .unsqueeze(0)
                        .expand(ref_H_tokens, ref_W_tokens)
                        .flatten()
                    )
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = (
                torch.arange(H_tokens, dtype=torch.int32, device=device)
                .unsqueeze(1)
                .expand(H_tokens, W_tokens)
                .flatten()
            )
            col_ids = (
                torch.arange(W_tokens, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .expand(H_tokens, W_tokens)
                .flatten()
            )

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size,
            encoder_seq_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size,
            max_ref_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        img_freqs_cis = torch.zeros(
            batch_size,
            max_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        # Calculate combined image sequence lengths (ref_img + img) for each sample
        combined_img_seq_lengths = [
            sum(ref_img_len) + img_len for ref_img_len, img_len in zip(l_effective_ref_img_len, l_effective_img_len)
        ]
        max_combined_img_len = max(combined_img_seq_lengths)

        # Create combined image rotary embeddings
        combined_img_freqs_cis = torch.zeros(
            batch_size,
            max_combined_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(
                l_effective_cap_len,
                l_effective_ref_img_len,
                l_effective_img_len,
                seq_lengths,
            )
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len,
            ]

            # Combined image rotary embeddings: ref_img + img (same order as img_patch_embed_and_refine)
            combined_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            combined_img_freqs_cis[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len,
            ]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
            combined_img_freqs_cis,
            combined_img_seq_lengths,
        )
