# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass

import torch

from ...utils import BaseOutput


@dataclass
class RFD4ProteinaPipelineOutput(BaseOutput):
    """Molecular inference results. Coordinates are in nanometers.

    Args:
        coords_nm (`torch.Tensor`):
            Decoded atom37 coordinates `(B, N, 37, 3)`.
        residue_type (`torch.Tensor`):
            Residue vocabulary indices `(B, N)`; -1 where no residue is emitted.
        emit_mask (`torch.Tensor`):
            Boolean mask `(B, N)` of the tokens that were designed.
        atom_mask (`torch.Tensor`):
            Boolean mask `(B, N, 37)` of the valid atoms.
        state (`dict[str, torch.Tensor]`):
            Final product-space state: `bb_ca` coordinates `(B, N, 3)` and `local_latents` `(B, N, D)`.
        clean (`dict[str, torch.Tensor]`):
            Predicted clean product-space state from the last denoising step.
        sequence_logits (`torch.Tensor`):
            Final per-token residue logits `(B, N, 147)`.
        insertion_logits (`torch.Tensor`):
            Final insertion predictions `(B, N, N)` for unindexed motifs.
    """

    coords_nm: torch.Tensor
    residue_type: torch.Tensor
    emit_mask: torch.Tensor
    atom_mask: torch.Tensor
    state: dict[str, torch.Tensor]
    clean: dict[str, torch.Tensor]
    sequence_logits: torch.Tensor
    insertion_logits: torch.Tensor
