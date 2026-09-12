# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import torch

from ...models import EchoWMTransformer3DModel
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


ECHO_WM_ACTION_KEYS = frozenset("wsadikjl")
ECHO_WM_TRANSLATION_SPEED = 0.05
ECHO_WM_ROTATION_SPEED_DEG = 0.5
ECHO_WM_PITCH_SPEED_DEG = 0.2
ECHO_WM_PITCH_LIMIT_DEG = 60.0
ECHO_WM_TRANSLATION_CALIBRATION = 30.0


def _parse_action(action: str) -> list[tuple[list[str], int]]:
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("`action` must not be empty.")
    segments = []
    for segment in cleaned.split(","):
        if "-" not in segment:
            raise ValueError(f"Invalid action segment {segment!r}; expected '<keys>-<duration>'.")
        keys_part, duration = segment.rsplit("-", 1)
        if not duration.isdigit() or int(duration) <= 0:
            raise ValueError(f"Invalid action duration in {segment!r}.")
        keys = [] if keys_part.lower() == "none" else sorted(set(keys_part.lower()))
        invalid = sorted(set(keys) - ECHO_WM_ACTION_KEYS)
        if invalid:
            raise ValueError(f"Unknown action keys {invalid}; allowed keys are `wasdijkl`.")
        segments.append((keys, int(duration)))
    return segments


def _rotation_x(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=torch.float64,
    )


def _rotation_y(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )


def action_to_camera_trajectory(
    action: str,
    num_frames: int,
    width: int,
    height: int,
    frame_rate: float = 24.0,
    translation_speed: float = ECHO_WM_TRANSLATION_SPEED,
    rotation_speed_deg: float = ECHO_WM_ROTATION_SPEED_DEG,
    pitch_speed_deg: float = ECHO_WM_PITCH_SPEED_DEG,
    pitch_limit_deg: float = ECHO_WM_PITCH_LIMIT_DEG,
    fov_deg: float = 70.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert Echo-WM's compact WASD/IJKL action language to camera poses and intrinsics."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    if frame_rate <= 0:
        raise ValueError(f"`frame_rate` must be positive, got {frame_rate}.")
    output_device = torch.device(device)
    frame_actions = []
    for keys, duration in _parse_action(action):
        frame_actions.extend([keys] * duration)

    # Match the reference NumPy trajectory in float64 on CPU, then cast once. MPS does not support float64 tensors.
    pose = torch.eye(4, dtype=torch.float64)
    velocity = torch.zeros(4, dtype=torch.float64)
    poses = [pose.clone()]
    previous = set()
    pitch = 0.0
    dt = 1.0 / frame_rate
    for keys in frame_actions:
        current = set(keys)
        target = torch.tensor(
            [
                float("w" in current) - float("s" in current),
                float("d" in current) - float("a" in current),
                float("l" in current) - float("j" in current),
                float("i" in current) - float("k" in current),
            ],
            dtype=torch.float64,
        )
        target *= torch.tensor(
            [
                translation_speed,
                translation_speed,
                math.radians(rotation_speed_deg),
                math.radians(pitch_speed_deg),
            ],
            dtype=torch.float64,
        )
        if current - previous:
            velocity = target
        else:
            time_constant = 0.45 if torch.any(target) else 1.0
            velocity += (target - velocity) * (1.0 - math.exp(-dt / time_constant))
        previous = current
        new_pitch = min(max(pitch + velocity[3].item(), -math.radians(pitch_limit_deg)), math.radians(pitch_limit_deg))
        pitch_step, pitch = new_pitch - pitch, new_pitch
        rotation = _rotation_y(velocity[2].item()) @ pose[:3, :3] @ _rotation_x(pitch_step)
        forward = rotation[:, 2].clone()
        forward[1] = 0
        right = rotation[:, 0].clone()
        right[1] = 0
        forward /= torch.linalg.vector_norm(forward).clamp_min(1e-6)
        right /= torch.linalg.vector_norm(right).clamp_min(1e-6)
        pose = torch.eye(4, dtype=torch.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = poses[-1][:3, 3] + forward * velocity[0] + right * velocity[1]
        poses.append(pose.clone())

    poses = torch.stack(poses).to(dtype=torch.float32, device=output_device)
    if poses.shape[0] < num_frames:
        poses = torch.cat([poses, poses[-1:].expand(num_frames - poses.shape[0], -1, -1)])
    poses = poses[:num_frames]
    poses = torch.linalg.inv(poses[:1]) @ poses
    poses[..., :3, 3] /= ECHO_WM_TRANSLATION_CALIBRATION

    focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    intrinsics = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=output_device,
        dtype=torch.float32,
    )
    return poses, intrinsics


class EchoWMCameraConditionStep(ModularPipelineBlocks):
    """Prepare the per-latent-frame camera matrices consumed by Echo-WM's UCPE transformer branches."""

    model_name = "echo-wm"
    translation_speed = ECHO_WM_TRANSLATION_SPEED
    rotation_speed_deg = ECHO_WM_ROTATION_SPEED_DEG
    pitch_limit_deg = ECHO_WM_PITCH_LIMIT_DEG
    output_dtype = None

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", EchoWMTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("action", type_hint=str, required=True, description="WASD/IJKL action program."),
            InputParam.template("height", default=704),
            InputParam.template("width", default=1280),
            InputParam("num_frames", type_hint=int, default=241, description="Number of output video frames."),
            InputParam("frame_rate", type_hint=float, default=24.0, description="Output video frame rate."),
            InputParam(
                "translation_speed",
                type_hint=float,
                default=self.translation_speed,
                description="Per-frame camera translation speed for W/A/S/D actions.",
            ),
            InputParam(
                "rotation_speed_deg",
                type_hint=float,
                default=self.rotation_speed_deg,
                description="Per-frame camera yaw speed in degrees for J/L actions.",
            ),
            InputParam(
                "pitch_speed_deg",
                type_hint=float,
                default=ECHO_WM_PITCH_SPEED_DEG,
                description="Per-frame camera pitch speed in degrees for I/K actions.",
            ),
            InputParam(
                "pitch_limit_deg",
                type_hint=float,
                default=self.pitch_limit_deg,
                description="Maximum absolute camera pitch in degrees.",
            ),
            InputParam(
                "fov_deg", type_hint=float, default=70.0, description="Horizontal camera field of view in degrees."
            ),
            InputParam.template("num_images_per_prompt", name="num_videos_per_prompt"),
            InputParam("batch_size", type_hint=int, required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("ucpe_viewmats", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
            OutputParam("ucpe_intrinsics", type_hint=torch.Tensor, kwargs_type="denoiser_input_fields"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        config = components.transformer.config
        if not config.ucpe_block_indices:
            raise ValueError("Echo-WM requires a transformer configured with `ucpe_block_indices`.")
        expected_grid = (
            block_state.height
            // components.vae_spatial_compression_ratio
            // components.transformer_spatial_patch_size,
            block_state.width // components.vae_spatial_compression_ratio // components.transformer_spatial_patch_size,
        )
        configured_grid = (config.ucpe_patches_y, config.ucpe_patches_x)
        if expected_grid != configured_grid:
            raise ValueError(
                f"The requested {block_state.height}x{block_state.width} resolution produces an UCPE grid of "
                f"{expected_grid}, but the checkpoint is configured for {configured_grid}."
            )
        poses, intrinsics = action_to_camera_trajectory(
            block_state.action,
            block_state.num_frames,
            block_state.width,
            block_state.height,
            frame_rate=block_state.frame_rate,
            translation_speed=block_state.translation_speed,
            rotation_speed_deg=block_state.rotation_speed_deg,
            pitch_speed_deg=block_state.pitch_speed_deg,
            pitch_limit_deg=block_state.pitch_limit_deg,
            fov_deg=block_state.fov_deg,
            device=components._execution_device,
        )
        latent_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        poses = poses[:: components.vae_temporal_compression_ratio][:latent_frames]
        output_dtype = components.transformer.dtype if self.output_dtype is None else self.output_dtype
        poses, intrinsics = poses.to(output_dtype), intrinsics.to(output_dtype)
        batch_size = block_state.batch_size * block_state.num_videos_per_prompt
        block_state.ucpe_viewmats = poses[None].expand(batch_size, -1, -1, -1).contiguous()
        block_state.ucpe_intrinsics = intrinsics[None, None].expand(batch_size, latent_frames, -1, -1).contiguous()
        self.set_block_state(state, block_state)
        return components, state


class EchoWMFlashCameraConditionStep(EchoWMCameraConditionStep):
    """Flash camera preparation, retaining FP32 matrices for bounded-anchor translation."""

    model_name = "echo-wm-flash"
    rotation_speed_deg = 0.4
    pitch_limit_deg = 40.0
    output_dtype = torch.float32
