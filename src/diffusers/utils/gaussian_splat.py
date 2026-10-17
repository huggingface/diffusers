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


from pathlib import Path

import numpy as np
import torch


def _transform_gaussians(gaussians, transform):
    parameters = (
        gaussians.detach().float().cpu().numpy()
        if isinstance(gaussians, torch.Tensor)
        else np.asarray(gaussians, dtype=np.float32)
    )
    if parameters.ndim != 2 or parameters.shape[-1] != 14:
        raise ValueError("Pass one Gaussian set with shape (num_gaussians, 14).")
    positions = parameters[:, :3]
    rotations = parameters[:, 9:13]
    transform = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]] if transform is None else transform, dtype=np.float32)
    if transform.shape != (3, 3):
        raise ValueError("transform must have shape (3, 3).")
    positions = positions @ transform.T
    quaternion = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)
    w, x, y, z = quaternion.T
    matrix = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    matrix = transform @ matrix
    trace = np.trace(matrix, axis1=1, axis2=2)
    result = np.zeros_like(rotations)
    scale = np.sqrt(np.maximum(trace + 1, 0)) * 2
    result[:, 0] = 0.25 * scale
    denominator = np.where(scale != 0, scale, 1)
    result[:, 1] = (matrix[:, 2, 1] - matrix[:, 1, 2]) / denominator
    result[:, 2] = (matrix[:, 0, 2] - matrix[:, 2, 0]) / denominator
    result[:, 3] = (matrix[:, 1, 0] - matrix[:, 0, 1]) / denominator
    for axis in range(3):
        other1, other2 = (axis + 1) % 3, (axis + 2) % 3
        mask = (
            (scale == 0)
            & (matrix[:, axis, axis] >= matrix[:, other1, other1])
            & (matrix[:, axis, axis] >= matrix[:, other2, other2])
        )
        for previous in range(axis):
            mask &= matrix[:, axis, axis] > matrix[:, previous, previous]
        axis_scale = (
            np.sqrt(np.maximum(1 + matrix[:, axis, axis] - matrix[:, other1, other1] - matrix[:, other2, other2], 0))
            * 2
        )
        axis_denominator = np.where(axis_scale != 0, axis_scale, 1)
        result[mask, 0] = (matrix[mask, other2, other1] - matrix[mask, other1, other2]) / axis_denominator[mask]
        result[mask, axis + 1] = 0.25 * axis_scale[mask]
        result[mask, other1 + 1] = (matrix[mask, other1, axis] + matrix[mask, axis, other1]) / axis_denominator[mask]
        result[mask, other2 + 1] = (matrix[mask, other2, axis] + matrix[mask, axis, other2]) / axis_denominator[mask]
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return parameters, positions, result


def export_to_gaussian_ply(gaussians, output_path, transform=None) -> str:
    """Save `(num_gaussians, 14)` parameters as a binary Gaussian PLY file.

    Parameters contain xyz, SH DC color, physical scale, wxyz rotation, and opacity. The default transform maps xyz to
    `(x, -z, y)`. Pass a 3-by-3 rotation matrix to use a different orientation.
    """
    parameters, positions, rotations = _transform_gaussians(gaussians, transform)
    colors = parameters[:, 3:6]
    opacity = parameters[:, 13:14]
    scales = parameters[:, 6:9]
    with np.errstate(divide="ignore"):
        logits = np.log(opacity / (1 - opacity))
    values = np.concatenate([positions, np.zeros_like(positions), colors, logits, np.log(scales), rotations], axis=-1)
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(values)}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    output_path = Path(output_path)
    output_path.write_bytes(header.encode("ascii") + values.astype("<f4").tobytes())
    return str(output_path)


def export_to_splat(gaussians, output_path, transform=None) -> str:
    """Save `(num_gaussians, 14)` parameters as 32-byte records for Gaussian splat viewers.

    Parameters contain xyz, SH DC color, physical scale, wxyz rotation, and opacity. The default transform maps xyz to
    `(x, -z, y)`. Pass a 3-by-3 rotation matrix to use a different orientation.
    """
    parameters, positions, rotations = _transform_gaussians(gaussians, transform)
    scales = parameters[:, 6:9]
    opacity = parameters[:, 13:14]
    rgb = np.clip((parameters[:, 3:6] * 0.28209479177387814 + 0.5) * 255, 0, 255).astype(np.uint8)
    rgba = np.concatenate([rgb, np.clip(opacity * 255, 0, 255).astype(np.uint8)], axis=-1)
    rotation_bytes = np.clip(rotations * 128 + 128, 0, 255).astype(np.uint8)
    order = np.argsort(-opacity[:, 0] * np.prod(scales, axis=-1))
    records = np.concatenate(
        [
            positions.astype("<f4").view(np.uint8).reshape(-1, 12),
            scales.astype("<f4").view(np.uint8).reshape(-1, 12),
            rgba,
            rotation_bytes,
        ],
        axis=-1,
    )[order]
    output_path = Path(output_path)
    output_path.write_bytes(records.tobytes())
    return str(output_path)
