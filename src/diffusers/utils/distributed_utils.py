# Copyright 2025 The HuggingFace Inc. team.
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


try:
    import torch
except ImportError:
    torch = None


def is_torch_dist_rank_zero() -> bool:
    if torch is None:
        return True

    dist_module = getattr(torch, "distributed", None)
    if dist_module is None or not dist_module.is_available():
        return True

    if not dist_module.is_initialized():
        return True

    try:
        return dist_module.get_rank() == 0
    except (RuntimeError, ValueError):
        return True


def sync_default_rng_seed(group=None) -> None:
    """Reset the default RNGs of every rank in `group` to the seed of the group's first rank.

    Under tensor or context parallelism, every rank must draw the same random numbers (initial latents, scheduler
    noise), or the sharded activations stop matching and the output is noise. Fresh processes start from different
    seeds, so without this, a pipeline call without a `generator` breaks. Ranks are reset even when their seeds already
    match, because the same seed does not mean the same position in the stream.
    """
    seeds = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(seeds, torch.initial_seed(), group=group)
    torch.manual_seed(seeds[0])
