"""Checkpoint utilities for parity debugging. No effect when _checkpoints is None."""
from dataclasses import dataclass, field

import torch


@dataclass
class Checkpoint:
    save: bool = False
    stop: bool = False
    load: bool = False
    data: dict = field(default_factory=dict)


def _maybe_checkpoint(checkpoints, name, data):
    if not checkpoints:
        return
    ckpt = checkpoints.get(name)
    if ckpt is None:
        return
    if ckpt.save:
        ckpt.data.update({
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        })
    if ckpt.stop:
        raise StopIteration(name)
