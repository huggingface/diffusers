# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import torch.distributed as dist

from ..utils import get_logger
from ._common import _BATCHED_INPUT_IDENTIFIERS
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name

_CFG_PARALLEL = "cfg_parallel"


class CFGParallelHook(ModelHook):
    def initialize_hook(self, module):
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment not initialized.")
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if len(args) > 0:
            logger.warning(
                "CFGParallelHook is an example hook that does not work with batched positional arguments. Please use with caution."
            )

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        assert world_size == 2, "This is an example hook designed to only work with 2 processes."

        for key in list(kwargs.keys()):
            if key not in _BATCHED_INPUT_IDENTIFIERS or kwargs[key] is None:
                continue
            kwargs[key] = torch.chunk(kwargs[key], world_size, dim=0)[rank].contiguous()

        output = self.fn_ref.original_forward(*args, **kwargs)
        sample = output[0]
        sample_list = [torch.empty_like(sample) for _ in range(world_size)]
        dist.all_gather(sample_list, sample)
        sample = torch.cat(sample_list, dim=0).contiguous()

        return_dict = kwargs.get("return_dict", False)
        if not return_dict:
            return (sample, *output[1:])
        return output.__class__(sample, *output[1:])


def apply_cfg_parallel(module: torch.nn.Module) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = CFGParallelHook()
    registry.register_hook(hook, _CFG_PARALLEL)
