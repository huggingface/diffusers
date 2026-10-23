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

import json

import torch

from ...models.transformers.transformer_hy_nomas import HyNomasTransformer3DModel
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import HyNomasGenerationState, HyNomasModularPipeline


def _timestamp(value):
    parts = str(value).split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid action timestamp: {value}")
    return sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))


def _action_windows(prompt, frames, fps):
    try:
        obj = json.loads(prompt)
    except (ValueError, TypeError):
        return None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return None
    actions = segments[0].get("contained_actions", [])
    if isinstance(actions, dict):
        actions = list(actions.values())
    if not isinstance(actions, list) or not actions:
        return None
    spans = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or "start_time" not in action or "end_time" not in action:
            return None
        start, end = [
            max(0, min(frames, int(max(_timestamp(action[key]), 0) * fps) // 4)) for key in ("start_time", "end_time")
        ]
        if end > start:
            spans.append((start, end, index))
    if not spans:
        return None
    spans.sort()
    cursor, windows = 0, []
    for start, end, index in spans:
        if start < cursor:
            raise ValueError("Action time windows must not overlap.")
        if start > cursor:
            windows.append((cursor, start, -1))
        windows.append((start, end, index))
        cursor = end
    if cursor < frames:
        windows.append((cursor, frames, -1))
    return None if len(windows) == len(actions) == 1 else windows


# auto_docstring
class HyNomasPrepareStep(ModularPipelineBlocks):
    """
    Validate segment lengths and camera poses, fork the world state, and prepare rollout noise.

      Components:
          transformer (`HyNomasTransformer3DModel`)

      Inputs:
          prompts (`list`):
              Ordered prompts for successive segments of one video, rather than a batch of independent videos.
          height (`int`, *optional*, defaults to 704):
              Output height in pixels; must be a positive multiple of 32 and match the existing session.
          width (`int`, *optional*, defaults to 1280):
              Output width in pixels; must be a positive multiple of 32 and match the existing session.
          num_frames (`int`, *optional*, defaults to 97):
              Exact number of new pixel frames to generate in this call, excluding all prior session history.
          segment_latent_lengths (`list`, *optional*):
              Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).
          camera_extrinsics (`Tensor`, *optional*):
              Normalized camera-to-world matrices in the training coordinate system, shaped (1, num_frames, 4, 4), for
              the new rollout only.
          fps (`float`, *optional*, defaults to 24.0):
              Frames per second used to convert structured action timestamps into latent-frame intervals.
          generator (`Generator`, *optional*):
              Generator for rollout noise and cache-write noise; omitted values restore the session generator or start
              from seed 42.
          latents (`Tensor`, *optional*):
              Optional initial noise for the new rollout, shaped (1, channels, ceil(num_frames / 4), height / 32, width
              / 32).
          noise_capacity (`int`, *optional*):
              Total latent-frame capacity reserved when creating a fixed session noise timeline, including future
              continuation segments.
          session (`HyNomasGenerationState`, *optional*):
              Earlier segment history to continue or branch; this call forks the state before appending new cache
              entries and latents.

      Outputs:
          session (`HyNomasGenerationState`):
              Forked generation state containing prior segment history and the optional fixed noise timeline.
          initial_latents (`Tensor`):
              Initial noise for the new rollout, shaped (1, channels, new_latent_frames, height / 32, width / 32).
          segment_latent_lengths (`list`):
              Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).
          segment_cameras (`list`):
              Normalized camera-to-world poses for each segment, shaped (1, latent_frames * 4, 4, 4), or None without
              camera conditioning.
          action_frame_windows (`list`):
              Per-segment (start, end, action_index) intervals in latent frames; -1 denotes an action gap, and None
              disables action-specific windows.
          generator (`Generator`):
              Generator initialized or restored for this rollout and its cache-write noise.
          segment_pixel_lengths (`list`):
              Exact visible pixel-frame count of each new segment after removing decoder padding from the final
              segment.
    """

    model_name = "hy-nomas"

    @property
    def description(self):
        return "Validate segment lengths and camera poses, fork the world state, and prepare rollout noise."

    @property
    def expected_components(self):
        return [ComponentSpec("transformer", HyNomasTransformer3DModel)]

    @property
    def inputs(self):
        return [
            InputParam(
                "prompts",
                type_hint=list[str],
                description="Ordered prompts for successive segments of one video, rather than a batch of independent videos.",
                required=True,
            ),
            InputParam(
                "height",
                type_hint=int,
                description="Output height in pixels; must be a positive multiple of 32 and match the existing session.",
                default=704,
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Output width in pixels; must be a positive multiple of 32 and match the existing session.",
                default=1280,
            ),
            InputParam(
                "num_frames",
                type_hint=int,
                description="Exact number of new pixel frames to generate in this call, excluding all prior session history.",
                default=97,
            ),
            InputParam(
                "segment_latent_lengths",
                type_hint=list[int],
                description="Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).",
            ),
            InputParam(
                "camera_extrinsics",
                type_hint=torch.Tensor,
                description="Normalized camera-to-world matrices in the training coordinate system, shaped (1, num_frames, 4, 4), for the new rollout only.",
            ),
            InputParam(
                "fps",
                type_hint=float,
                description="Frames per second used to convert structured action timestamps into latent-frame intervals.",
                default=24.0,
            ),
            InputParam(
                "generator",
                type_hint=torch.Generator,
                description="Generator for rollout noise and cache-write noise; omitted values restore the session generator or start from seed 42.",
            ),
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Optional initial noise for the new rollout, shaped (1, channels, ceil(num_frames / 4), height / 32, width / 32).",
            ),
            InputParam(
                "noise_capacity",
                type_hint=int,
                description="Total latent-frame capacity reserved when creating a fixed session noise timeline, including future continuation segments.",
            ),
            InputParam(
                "session",
                type_hint=HyNomasGenerationState,
                description="Earlier segment history to continue or branch; this call forks the state before appending new cache entries and latents.",
            ),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "session",
                type_hint=HyNomasGenerationState,
                description="Forked generation state containing prior segment history and the optional fixed noise timeline.",
            ),
            OutputParam(
                "initial_latents",
                type_hint=torch.Tensor,
                description="Initial noise for the new rollout, shaped (1, channels, new_latent_frames, height / 32, width / 32).",
            ),
            OutputParam(
                "segment_latent_lengths",
                type_hint=list[int],
                description="Number of latent frames in each new segment, one per prompt; the total must equal ceil(num_frames / 4).",
            ),
            OutputParam(
                "segment_cameras",
                type_hint=list[torch.Tensor | None],
                description="Normalized camera-to-world poses for each segment, shaped (1, latent_frames * 4, 4, 4), or None without camera conditioning.",
            ),
            OutputParam(
                "action_frame_windows",
                type_hint=list[list[tuple[int, int, int]] | None],
                description="Per-segment (start, end, action_index) intervals in latent frames; -1 denotes an action gap, and None disables action-specific windows.",
            ),
            OutputParam(
                "generator",
                type_hint=torch.Generator,
                description="Generator initialized or restored for this rollout and its cache-write noise.",
            ),
            OutputParam(
                "segment_pixel_lengths",
                type_hint=list[int],
                description="Exact visible pixel-frame count of each new segment after removing decoder padding from the final segment.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state) -> tuple[HyNomasModularPipeline, PipelineState]:
        s = self.get_block_state(state)
        if min(s.height, s.width) < 32 or s.height % 32 or s.width % 32:
            raise ValueError("height and width must be positive multiples of 32.")
        if not isinstance(s.num_frames, int) or s.num_frames < 1:
            raise ValueError("num_frames must be a positive integer.")
        if s.fps <= 0:
            raise ValueError("fps must be positive.")
        frames = (s.num_frames - 1) // 4 + 1
        count = len(s.prompts)
        if s.segment_latent_lengths is None:
            if frames % count:
                raise ValueError(
                    "Provide segment_latent_lengths when latent frames do not divide equally among prompts."
                )
            s.segment_latent_lengths = [frames // count] * count
        lengths = s.segment_latent_lengths
        if len(lengths) != count or any(not isinstance(x, int) or x < 1 for x in lengths) or sum(lengths) != frames:
            raise ValueError(
                "segment_latent_lengths must contain one positive length per prompt and sum to latent frames."
            )
        session = (
            s.session.fork()
            if s.session is not None
            else HyNomasGenerationState(
                kv_cache=[{} for _ in range(components.transformer.config.num_layers)], height=s.height, width=s.width
            )
        )
        if (session.height, session.width) != (s.height, s.width):
            raise ValueError("A continuation must use the same height and width as its session.")
        if s.generator is None:
            s.generator = torch.Generator(device=session.generator_device)
            if session.generator_state is None:
                s.generator.manual_seed(42)
            else:
                s.generator.set_state(session.generator_state)
        if isinstance(s.generator, list):
            raise ValueError("Hy-Nomas generates one world per call; provide a single generator.")
        shape = (1, components.transformer.config.in_channels, frames, s.height // 32, s.width // 32)
        device, dtype = components._execution_device, components.transformer.dtype
        if s.latents is not None:
            if tuple(s.latents.shape) != shape:
                raise ValueError(f"latents must have shape {shape}.")
            noise = s.latents.to(device=device, dtype=dtype)
        elif session.noise_bank is not None or s.noise_capacity is not None:
            if session.noise_bank is None:
                if s.noise_capacity < frames:
                    raise ValueError("noise_capacity must cover all requested latent frames.")
                session.noise_bank = randn_tensor(
                    (shape[0], shape[1], s.noise_capacity, *shape[3:]),
                    generator=s.generator,
                    device=device,
                    dtype=dtype,
                )
            end = session.frame_offset + frames
            if end > session.noise_bank.shape[2]:
                raise ValueError("The request exceeds the session's fixed noise_capacity.")
            noise = session.noise_bank[:, :, session.frame_offset : end]
        else:
            noise = randn_tensor(shape, generator=s.generator, device=device, dtype=dtype)
        cameras = [None] * count
        if s.camera_extrinsics is not None:
            camera = s.camera_extrinsics
            if tuple(camera.shape) != (1, s.num_frames, 4, 4) or not torch.isfinite(camera).all():
                raise ValueError("camera_extrinsics must be finite with shape (1, num_frames, 4, 4).")
            camera = camera.to(device=device, dtype=dtype)
            # The original camera embedder pads at the beginning to a whole group of four pixel frames.
            padding = frames * 4 - s.num_frames
            camera = torch.cat([camera[:, :1].expand(-1, padding, -1, -1), camera], dim=1)
            cameras = list(camera.split([length * 4 for length in lengths], dim=1))
        s.session, s.initial_latents, s.segment_cameras = session, noise, cameras
        s.segment_pixel_lengths = [length * 4 for length in lengths]
        s.segment_pixel_lengths[-1] -= frames * 4 - s.num_frames
        s.action_frame_windows = [_action_windows(prompt, length, s.fps) for prompt, length in zip(s.prompts, lengths)]
        self.set_block_state(state, s)
        return components, state
