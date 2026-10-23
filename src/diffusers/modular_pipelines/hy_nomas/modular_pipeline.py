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

from dataclasses import dataclass, field, replace

import torch

from ..modular_pipeline import ModularPipeline


@dataclass
class HyNomasGenerationState:
    """History for one world. Pass the returned state as `session` to continue generation.

    Tensors belong to the execution device. A pipeline call forks the state before writing new cache entries, so an
    earlier state can also be reused to branch the world. Do not mutate tensors in a state while it is in use.
    """

    kv_cache: list[dict[str, torch.Tensor]] = field(default_factory=list)
    latent_chunks: list[torch.Tensor] = field(default_factory=list)
    pixel_lengths: list[int] = field(default_factory=list)
    frame_offset: int = 0
    height: int = 0
    width: int = 0
    noise_bank: torch.Tensor | None = None
    generator_state: torch.Tensor | None = None
    generator_device: str = "cpu"

    def fork(self):
        return replace(
            self,
            kv_cache=[dict(entry) for entry in self.kv_cache],
            latent_chunks=list(self.latent_chunks),
            pixel_lengths=list(self.pixel_lengths),
        )


class HyNomasModularPipeline(ModularPipeline):
    """Modular inference for the distilled Hy-Nomas text and camera conditioned world model.

    A list of prompts describes successive segments of a single video. `segment_latent_lengths` assigns their
    durations; each latent frame corresponds to four pixel frames. `num_frames` is the length of the new rollout.
    Camera matrices are normalized camera-to-world extrinsics in the training coordinate system.
    """

    default_blocks_name = "HyNomasAutoBlocks"
