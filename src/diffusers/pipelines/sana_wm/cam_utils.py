# Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.
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

"""Camera + image utilities for the SANA-WM pipeline.

* Action-string DSL → camera-to-world trajectory.
* Resize-and-center-crop to (704, 1280) with intrinsics adjustment.
* Plücker / raymap packing for the DiT camera-control branch.
* Optional Pi3X-based intrinsics estimation (only if `pi3` is installed).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image


TARGET_HEIGHT = 704
TARGET_WIDTH = 1280

DEFAULT_TRANSLATION_SPEED = 0.05
DEFAULT_ROTATION_SPEED_DEG = 1.2
DEFAULT_PITCH_LIMIT_DEG = 85.0
ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")


# ---------------------------------------------------------------------------
# Action DSL → camera-to-world trajectory
# ---------------------------------------------------------------------------


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _parse_action_string(action: str) -> list[list[str]]:
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(f"Invalid action segment {segment!r}: expected '<keys>-<duration>'.")
        keys_part, dur_str = segment.rsplit("-", 1)
        if not dur_str.isdigit() or int(dur_str) <= 0:
            raise ValueError(f"Action segment {segment!r} has a non-positive duration {dur_str!r}.")
        n = int(dur_str)
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted({c for c in keys_lower if c not in ALLOWED_ACTION_KEYS})
            if bad:
                raise ValueError(
                    f"Action segment {segment!r} contains unknown keys {bad}; "
                    f"allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(n)])
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
) -> np.ndarray:
    """Roll out a ``(N+1, 4, 4)`` c2w trajectory from a WASD+IJKL action DSL.

    Coordinate convention: OpenCV (``+X right, +Y down, +Z forward``). WASD translates on the world XZ plane; IJKL
    applies pitch / yaw.
    """
    per_frame = _parse_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    current_pitch = 0.0

    for keys in per_frame:
        held = set(keys)
        R = current[:3, :3]
        T_ = current[:3, 3]

        pitch_delta = (rotate_rad if "i" in held else 0.0) - (rotate_rad if "k" in held else 0.0)
        new_pitch = current_pitch + pitch_delta
        if not (-pitch_limit_rad <= new_pitch <= pitch_limit_rad):
            pitch_delta = 0.0
        else:
            current_pitch = new_pitch

        yaw_delta = (rotate_rad if "l" in held else 0.0) - (rotate_rad if "j" in held else 0.0)
        R_new = _rot_y(yaw_delta) @ R @ _rot_x(pitch_delta)

        forward = R_new[:, 2].copy()
        forward[1] = 0.0
        right = R_new[:, 0].copy()
        right[1] = 0.0
        if (fn := float(np.linalg.norm(forward))) > 0:
            forward /= fn + 1e-6
        if (rn := float(np.linalg.norm(right))) > 0:
            right /= rn + 1e-6
        move = np.zeros(3, dtype=np.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = R_new
        current[:3, 3] = T_ + move
        poses.append(current.copy())

    return np.stack(poses, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Intrinsics handling
# ---------------------------------------------------------------------------


def transform_intrinsics_for_crop(
    intrinsics_vec4: np.ndarray,
    src_size: tuple[int, int],
    resized_size: tuple[int, int],
    crop_offset: tuple[int, int],
) -> np.ndarray:
    """Adjust ``[fx, fy, cx, cy]`` to match a resize-then-center-crop image."""
    src_w, src_h = src_size
    rw, rh = resized_size
    cl, ct = crop_offset
    sx, sy = rw / src_w, rh / src_h
    out = intrinsics_vec4.copy()
    out[..., 0] *= sx
    out[..., 2] = out[..., 2] * sx - cl
    out[..., 1] *= sy
    out[..., 3] = out[..., 3] * sy - ct
    return out


def estimate_intrinsics_with_pi3x(image: Image.Image, device: torch.device | str = "cuda") -> np.ndarray:
    """Estimate ``[fx, fy, cx, cy]`` for ``image`` using Pi3X.

    Optional helper — requires ``pip install pi3-vision``. The result is in the **original image** pixel grid (not the
    cropped one); pass it to [`SanaWMPipeline.__call__`] as ``intrinsics=...``.
    """
    try:
        from pi3.models.pi3x import Pi3X  # type: ignore
        from pi3.utils.geometry import recover_intrinsic_from_rays_d  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pi3 is required for intrinsics estimation. Pass `intrinsics` explicitly or `pip install pi3-vision`."
        ) from e

    from torchvision import transforms as T  # noqa: PLC0415

    device_t = torch.device(device)
    W_orig, H_orig = image.size
    pixel_limit = 255_000
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    W_t, H_t = W_orig * scale, H_orig * scale
    k, m = max(1, round(W_t / 14)), max(1, round(H_t / 14))
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_t / H_t:
            k -= 1
        else:
            m -= 1
    W_model, H_model = max(1, k) * 14, max(1, m) * 14
    resized = image.resize((W_model, H_model), Image.Resampling.LANCZOS)
    tensor = T.ToTensor()(resized).unsqueeze(0).unsqueeze(0).to(device_t)

    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device_t).eval()
    model.disable_multimodal()
    model.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model(imgs=tensor)
    rays_d = torch.nn.functional.normalize(out["local_points"], dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)[0, 0]
    K = K.detach().cpu().float().numpy()
    sx, sy = W_orig / W_model, H_orig / H_model
    return np.array([K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def resize_and_center_crop(
    image: Image.Image,
    target_h: int = TARGET_HEIGHT,
    target_w: int = TARGET_WIDTH,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Aspect-preserving resize then center-crop to ``(target_h, target_w)``."""
    src_w, src_h = image.size
    scale = max(target_h / src_h, target_w / src_w)
    rw = max(target_w, int(round(src_w * scale)))
    rh = max(target_h, int(round(src_h * scale)))
    resized = image.resize((rw, rh), Image.LANCZOS)
    left = (rw - target_w) // 2
    top = (rh - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return cropped, (src_w, src_h), (rw, rh), (left, top)


# ---------------------------------------------------------------------------
# Camera condition packing — Plücker + raymap
# ---------------------------------------------------------------------------


def compute_raymap(
    intrinsics: torch.Tensor,
    poses: torch.Tensor,
    H: int,
    W: int,
    *,
    use_plucker: bool = True,
) -> torch.Tensor:
    """Compute a per-pixel ray geometry map.

    Args:
        intrinsics: ``(T, 4)`` ``[fx, fy, cx, cy]`` per frame.
        poses: ``(T, 4, 4)`` camera-to-world poses (OpenCV convention).
        H: spatial height.
        W: spatial width.
        use_plucker: if True returns Plücker coordinates ``(d, m)``; otherwise
            returns ``(origin, direction)``.

    Returns:
        ``(T, H, W, 6)`` tensor.
    """
    T = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x_grid = x_grid[None].expand(T, -1, -1)
    y_grid = y_grid[None].expand(T, -1, -1)
    fx = intrinsics[:, 0].view(T, 1, 1)
    fy = intrinsics[:, 1].view(T, 1, 1)
    cx = intrinsics[:, 2].view(T, 1, 1)
    cy = intrinsics[:, 3].view(T, 1, 1)
    dirs_cam = torch.stack(
        [(x_grid - cx) / fx, (y_grid - cy) / fy, torch.ones_like(x_grid)],
        dim=-1,
    )
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    dirs_world = torch.einsum("tij,thwj->thwi", R, dirs_cam)
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
    origins = t.view(T, 1, 1, 3).expand_as(dirs_world)
    if use_plucker:
        moments = torch.cross(origins, dirs_world, dim=-1)
        return torch.cat([dirs_world, moments], dim=-1)
    return torch.cat([origins, dirs_world], dim=-1)


def _pose_inverse(T44: torch.Tensor) -> torch.Tensor:
    R = T44[..., :3, :3]
    t = T44[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T44)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def prepare_camera(
    poses_c2w: np.ndarray,
    intrinsics_vec4: np.ndarray,
    *,
    target_size: tuple[int, int],
    vae_stride: tuple[int, int, int],
) -> dict[str, torch.Tensor]:
    """Build the DiT-input camera tensors.

    Returns a dict with:

      * ``raymap`` ``(T_lat, 20)`` — flattened (rel-pose, intrinsics) per latent frame
      * ``chunk_plucker`` ``(6 * vae_time_stride, T_lat, H_lat, W_lat)`` — Plücker coordinates packed by chunk.
    """
    num_frames = poses_c2w.shape[0]
    vae_time_stride, vae_spatial_stride = vae_stride[0], vae_stride[-1]
    H_pixel, W_pixel = target_size
    latent_h = H_pixel // vae_spatial_stride
    latent_w = W_pixel // vae_spatial_stride
    latent_frames = (num_frames - 1) // vae_time_stride + 1

    poses = torch.from_numpy(poses_c2w).float()
    first_inv = _pose_inverse(poses[0:1]).squeeze(0)
    poses_rel = torch.matmul(first_inv, poses[1:])
    poses = torch.cat([torch.eye(4).unsqueeze(0), poses_rel], dim=0)

    intrinsics = torch.from_numpy(intrinsics_vec4).float()
    intrinsics_latent = intrinsics.clone()
    intrinsics_latent[:, [0, 2]] *= latent_w / float(W_pixel)
    intrinsics_latent[:, [1, 3]] *= latent_h / float(H_pixel)

    time_indices = torch.arange(0, num_frames, vae_time_stride)
    if len(time_indices) > latent_frames:
        time_indices = time_indices[:latent_frames]

    raymap = torch.cat(
        [poses[time_indices].reshape(len(time_indices), -1), intrinsics_latent[time_indices]],
        dim=-1,
    )

    chunk_starts = time_indices - (vae_time_stride - 1)
    chunks = []
    for start in chunk_starts:
        s = max(0, int(start))
        e = s + vae_time_stride
        chunk_poses, chunk_intrs = poses[s:e], intrinsics_latent[s:e]
        if chunk_poses.shape[0] < vae_time_stride:
            pad = vae_time_stride - chunk_poses.shape[0]
            chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].repeat(pad, 1, 1)], dim=0)
            chunk_intrs = torch.cat([chunk_intrs, chunk_intrs[-1:].repeat(pad, 1)], dim=0)
        plucker = compute_raymap(chunk_intrs, chunk_poses, latent_h, latent_w, use_plucker=True)
        chunks.append(plucker.permute(0, 3, 1, 2).reshape(-1, latent_h, latent_w))
    chunk_plucker = torch.stack(chunks).permute(1, 0, 2, 3)
    return {"raymap": raymap, "chunk_plucker": chunk_plucker}


def snap_num_frames(n: int, stride: int = 8, *, upper_bound: int | None = None) -> int:
    """Snap ``n`` to the nearest ``stride*k + 1`` (LTX-2 VAE constraint)."""
    if n < 1:
        return 1
    if (n - 1) % stride == 0:
        return n
    floor_cand = n - ((n - 1) % stride)
    ceil_cand = floor_cand + stride
    snapped = floor_cand if (n - floor_cand) < (ceil_cand - n) else ceil_cand
    if upper_bound is not None and snapped > upper_bound:
        snapped = floor_cand
    return max(snapped, 1)
