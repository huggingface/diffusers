# Copyright 2026 The VAST-AI Team. All rights reserved.
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

from ...utils.torch_utils import randn_tensor


def sample_noise(pipe, shape, generator, device, dtype):
    parallel_config = getattr(pipe.transformer, "_parallel_config", None)
    context_parallel_config = getattr(parallel_config, "context_parallel_config", None)
    if context_parallel_config is None:
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    group = context_parallel_config._flattened_mesh.get_group()
    source_rank = torch.distributed.get_global_rank(group, 0)
    if torch.distributed.get_rank() == source_rank:
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        noise = torch.empty(shape, device=device, dtype=dtype)
    torch.distributed.broadcast(noise, src=source_rank, group=group)
    return noise


def prepare_model_input(pipe, latents, condition, first_frame, timestep, start=0, causal=False):
    model_input = latents
    pin_first = start == 0 and (causal or not pipe._image_conditioned)
    if pin_first:
        model_input = model_input.clone()
        model_input[:, :, :1] = first_frame
    if pipe._image_conditioned:
        model_input = torch.cat([model_input, condition[:, :, start : start + latents.shape[2]]], dim=1)
    timestep = timestep.expand(latents.shape[0])
    if pin_first:
        patch = pipe.transformer.config.patch_size
        tokens_per_frame = (latents.shape[3] // patch[1]) * (latents.shape[4] // patch[2])
        timestep = timestep[:, None].expand(-1, latents.shape[2] * tokens_per_frame).clone()
        timestep[:, :tokens_per_frame] = 0
    return model_input, timestep
