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

import math
from typing import Generator, List, Optional

import torch


class ContextScheduler:
    def __init__(self, length: int = 16, stride: int = 3, overlap: int = 4, loop: bool = False, type: str = "uniform_constant") -> None:
        self.length = length
        self.stride = stride
        self.overlap = overlap
        self.loop = loop
        self.type = type
    
    def __call__(self, num_frames: int, step_index: int, num_inference_steps: int, generator: Optional[torch.Generator] = None) -> None:
        if self.type == "uniform_original_v1":
            return uniform_original_v1(num_frames, step_index, self.length, self.stride, self.overlap, self.loop)
        elif self.type == "uniform_original_v2":
            return uniform_original_v2(num_frames, step_index, self.length, self.stride, self.overlap, self.loop)
        elif self.type == "uniform_constant":
            return uniform_constant(num_frames, step_index, self.length, self.stride, self.overlap, self.loop)
        elif self.type == "simple_overlap":
            return simple_overlap(num_frames, self.length, self.overlap, self.loop)
        else:
            raise ValueError(f"Unknown context scheduler type: {self.type}")


def ordered_halving(val: int) -> float:
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)
    final = as_int / (1 << 64)
    return final


def _is_sorted(l: List[int]) -> bool:
    return all([l[i] < l[i + 1] for i in range(len(l) - 1)])


def uniform_original_v1(
    num_frames: int,
    step: int,
    length: int,
    stride: int,
    overlap: int,
    loop: bool,
):
    if num_frames <= length:
        yield list(range(num_frames))
        return

    stride = min(stride, int(math.ceil(math.log2(num_frames / length))) + 1)
    strides = [1 << i for i in range(stride)]
    pad = int(round(num_frames * ordered_halving(step)))
    
    for s in strides:
        start_index = int(ordered_halving(step) * s) + pad
        end_index = num_frames + pad + (0 if loop else -overlap)
        step_size = length * s - overlap

        for j in range(start_index, end_index, step_size):
            context_indices = [(j + s * i) % num_frames for i in range(length)]
            yield context_indices


def uniform_original_v2(
    num_frames: int,
    step: int,
    length: int,
    stride: int,
    overlap: int,
    loop: bool,
):
    if num_frames <= length:
        yield list(range(num_frames))
        return

    stride = min(stride, int(math.ceil(math.log2(num_frames / length))) + 1)
    strides = [1 << i for i in range(stride)]
    pad = int(round(num_frames * ordered_halving(step)))
    
    for s in strides:
        start_index = int(ordered_halving(step) * s) + pad
        end_index = num_frames + pad - overlap
        step_size = length * s - overlap

        for j in range(start_index, end_index, step_size):
            if length * s > num_frames:
                yield [e % num_frames for e in range(j, j + num_frames, s)]
                continue
            
            j = j % num_frames
            
            if j > (j + length * s) % num_frames and not loop:
                yield [e for e in range(j, num_frames, s)]
                j_stop = (j + length * s) % num_frames
                yield [e for e in range(0, j_stop, s)]
                continue
            
            yield [(j + i * s) % num_frames for i in range(length)]


def uniform_constant(
    num_frames: int,
    step: int,
    length: int,
    stride: int,
    overlap: int,
    loop: bool,
):
    if num_frames <= length:
        yield list(range(num_frames))
        return

    stride = min(stride, int(math.ceil(math.log2(num_frames / length))) + 1)
    strides = [1 << i for i in range(stride)]

    for s in strides:
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * s) + pad,
            num_frames + pad + (0 if loop else -overlap),
            (length * s - overlap),
        ):
            skip_window = False
            prev_val = -1
            context_window = []
            
            for i in range(length):
                e = (j + i * s) % num_frames
                if not loop and e < prev_val:
                    skip_window = True
                    break
                context_window.append(e)
                prev_val = e
            
            if skip_window:
                continue
            
            yield context_window


def simple_overlap(num_frames: int, length: int, overlap: int, loop: bool) -> Generator[List[int], None, None]:
    if num_frames <= length:
        yield list(range(num_frames))
        return

    for i in range(0, num_frames, length - overlap):
        context_indices = [j % num_frames for j in range(i, i + length)]
        if not loop and not _is_sorted(context_indices):
            continue
        yield context_indices


# def uniform_schedule(num_frames: int, length: int, stride: int, overlap: int, loop: bool, generator: Optional[torch.Generator] = None) -> Generator[List[int], None, None]:
#     if num_frames <= length:
#         yield list(range(num_frames))
#         return

#     stride = min(stride, int(math.ceil(math.log2(num_frames / length)) + 1))
#     strides = [1 << i for i in range(stride)]

#     for s in strides:
#         start_index = int(torch.randint(0, s, (1,), generator=generator).item())
#         end_index = num_frames + (0 if loop else -overlap)
#         step_size = length * s - overlap

#         for index in range(start_index, end_index, step_size):
#             context_indices = [(index + i * s) % num_frames for i in range(length)]
#             if not loop and not _is_sorted(context_indices):
#                 continue
#             yield context_indices
