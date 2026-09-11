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

"""Original configuration helpers and model presets for the prx assembly recipe."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PRXBase:
    context_in_dim: int = 2304
    hidden_size: int = 1792
    mlp_ratio: float = 3.5
    num_heads: int = 28
    depth: int = 16
    axes_dim: Tuple[int, int] = (32, 32)
    theta: int = 10_000
    time_factor: float = 1000.0
    time_max_period: int = 10_000
    bottleneck_size: Optional[int] = None
    resolution_embeds: bool = False


@dataclass(frozen=True)
class PRXFlux(PRXBase):
    in_channels: int = 16
    patch_size: int = 2


@dataclass(frozen=True)
class PRXDCAE(PRXBase):
    in_channels: int = 32
    patch_size: int = 1


@dataclass(frozen=True)
class PRXPixel(PRXBase):
    # Pixel-space RGB diffusion (PRXPixel / 7B).
    in_channels: int = 3
    patch_size: int = 16
    context_in_dim: int = 2048  # Qwen3-VL-Embedding-2B hidden size
    hidden_size: int = 3584
    num_heads: int = 28
    depth: int = 24
    axes_dim: Tuple[int, int] = (64, 64)
    bottleneck_size: int = 768
    resolution_embeds: bool = True


VARIANTS = {"flux": PRXFlux, "dc-ae": PRXDCAE, "pixel": PRXPixel}


def build_config(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}. Choose from {list(VARIANTS)}")
    config_dict = asdict(VARIANTS[variant]())
    config_dict["axes_dim"] = list(config_dict["axes_dim"])
    if config_dict["bottleneck_size"] is None:
        # Keep config.json clean for variants that don't use the bottleneck.
        config_dict.pop("bottleneck_size")
    return config_dict


def create_scheduler_config(shift: float):
    return {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": shift}


__all__ = ["build_config", "create_scheduler_config"]
