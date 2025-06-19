# Copyright 2025 Open AI and The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...utils import BaseOutput
from .camera import create_pan_cameras


def sample_pmf(pmf: torch.Tensor, n_samples: int) -> torch.Tensor:
    r"""
    Sample from the given discrete probability distribution with replacement.

    The i-th bin is assumed to have mass pmf[i].

    Args:
        pmf: [batch_size, *shape, n_samples, 1] where (pmf.sum(dim=-2) == 1).all()
        n_samples: number of samples

    Return:
        indices sampled with replacement
    """

    *shape, support_size, last_dim = pmf.shape
    assert last_dim == 1

    cdf = torch.cumsum(pmf.view(-1, support_size), dim=1)
    inds = torch.searchsorted(cdf, torch.rand(cdf.shape[0], n_samples, device=cdf.device))

    return inds.view(*shape, n_samples, 1).clamp(0, support_size - 1)


def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://huggingface.co/papers/2210.04628
    """
    if min_deg == max_deg:
        return x

    scales = 2.0 ** torch.arange(min_deg, max_deg, dtype=x.dtype, device=x.device)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)


def encode_position(position):
    return posenc_nerf(position, min_deg=0, max_deg=15)


def encode_direction(position, direction=None):
    if direction is None:
        return torch.zeros_like(posenc_nerf(position, min_deg=0, max_deg=8))
    else:
        return posenc_nerf(direction, min_deg=0, max_deg=8)


def _sanitize_name(x: str) -> str:
    return x.replace(".", "__")


def integrate_samples(volume_range, ts, density, channels):
    r"""
    Function integrating the model output.

    Args:
        volume_range: Specifies the integral range [t0, t1]
        ts: timesteps
        density: torch.Tensor [batch_size, *shape, n_samples, 1]
        channels: torch.Tensor [batch_size, *shape, n_samples, n_channels]
    returns:
        channels: integrated rgb output weights: torch.Tensor [batch_size, *shape, n_samples, 1] (density
        *transmittance)[i] weight for each rgb output at [..., i, :]. transmittance: transmittance of this volume
    )
    """

    # 1. Calculate the weights
    _, _, dt = volume_range.partition(ts)
    ddensity = density * dt

    mass = torch.cumsum(ddensity, dim=-2)
    transmittance = torch.exp(-mass[..., -1, :])

    alphas = 1.0 - torch.exp(-ddensity)
    Ts = torch.exp(torch.cat([torch.zeros_like(mass[..., :1, :]), -mass[..., :-1, :]], dim=-2))
    # This is the probability of light hitting and reflecting off of
    # something at depth [..., i, :].
    weights = alphas * Ts

    # 2. Integrate channels
    channels = torch.sum(channels * weights, dim=-2)

    return channels, weights, transmittance


def volume_query_points(volume, grid_size):
    indices = torch.arange(grid_size**3, device=volume.bbox_min.device)
    zs = indices % grid_size
    ys = torch.div(indices, grid_size, rounding_mode="trunc") % grid_size
    xs = torch.div(indices, grid_size**2, rounding_mode="trunc") % grid_size
    combined = torch.stack([xs, ys, zs], dim=1)
    return (combined.float() / (grid_size - 1)) * (volume.bbox_max - volume.bbox_min) + volume.bbox_min


def _convert_srgb_to_linear(u: torch.Tensor):
    return torch.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _create_flat_edge_indices(
    flat_cube_indices: torch.Tensor,
    grid_size: Tuple[int, int, int],
):
    num_xs = (grid_size[0] - 1) * grid_size[1] * grid_size[2]
    y_offset = num_xs
    num_ys = grid_size[0] * (grid_size[1] - 1) * grid_size[2]
    z_offset = num_xs + num_ys
    return torch.stack(
        [
            # Edges spanning x-axis.
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2],
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]
            + flat_cube_indices[:, 2],
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            # Edges spanning y-axis.
            (
                y_offset
                + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
            ),
            (
                y_offset
                + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
            ),
            (
                y_offset
                + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
                + 1
            ),
            (
                y_offset
                + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]
                + flat_cube_indices[:, 1] * grid_size[2]
                + flat_cube_indices[:, 2]
                + 1
            ),
            # Edges spanning z-axis.
            (
                z_offset
                + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
            (
                z_offset
                + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 1] * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
            (
                z_offset
                + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)
                + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
            (
                z_offset
                + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)
                + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)
                + flat_cube_indices[:, 2]
            ),
        ],
        dim=-1,
    )


class VoidNeRFModel(nn.Module):
    """
    Implements the default empty space model where all queries are rendered as background.
    """

    def __init__(self, background, channel_scale=255.0):
        super().__init__()
        background = nn.Parameter(torch.from_numpy(np.array(background)).to(dtype=torch.float32) / channel_scale)

        self.register_buffer("background", background)

    def forward(self, position):
        background = self.background[None].to(position.device)

        shape = position.shape[:-1]
        ones = [1] * (len(shape) - 1)
        n_channels = background.shape[-1]
        background = torch.broadcast_to(background.view(background.shape[0], *ones, n_channels), [*shape, n_channels])

        return background


@dataclass
class VolumeRange:
    t0: torch.Tensor
    t1: torch.Tensor
    intersected: torch.Tensor

    def __post_init__(self):
        assert self.t0.shape == self.t1.shape == self.intersected.shape

    def partition(self, ts):
        """
        Partitions t0 and t1 into n_samples intervals.

        Args:
            ts: [batch_size, *shape, n_samples, 1]

        Return:

            lower: [batch_size, *shape, n_samples, 1] upper: [batch_size, *shape, n_samples, 1] delta: [batch_size,
            *shape, n_samples, 1]

        where
            ts \\in [lower, upper] deltas = upper - lower
        """

        mids = (ts[..., 1:, :] + ts[..., :-1, :]) * 0.5
        lower = torch.cat([self.t0[..., None, :], mids], dim=-2)
        upper = torch.cat([mids, self.t1[..., None, :]], dim=-2)
        delta = upper - lower
        assert lower.shape == upper.shape == delta.shape == ts.shape
        return lower, upper, delta


class BoundingBoxVolume(nn.Module):
    """
    Axis-aligned bounding box defined by the two opposite corners.
    """

    def __init__(
        self,
        *,
        bbox_min,
        bbox_max,
        min_dist: float = 0.0,
        min_t_range: float = 1e-3,
    ):
        """
        Args:
            bbox_min: the left/bottommost corner of the bounding box
            bbox_max: the other corner of the bounding box
            min_dist: all rays should start at least this distance away from the origin.
        """
        super().__init__()

        self.min_dist = min_dist
        self.min_t_range = min_t_range

        self.bbox_min = torch.tensor(bbox_min)
        self.bbox_max = torch.tensor(bbox_max)
        self.bbox = torch.stack([self.bbox_min, self.bbox_max])
        assert self.bbox.shape == (2, 3)
        assert min_dist >= 0.0
        assert min_t_range > 0.0

    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        epsilon=1e-6,
    ):
        """
        Args:
            origin: [batch_size, *shape, 3]
            direction: [batch_size, *shape, 3]
            t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
            params: Optional meta parameters in case Volume is parametric
            epsilon: to stabilize calculations

        Return:
            A tuple of (t0, t1, intersected) where each has a shape [batch_size, *shape, 1]. If a ray intersects with
            the volume, `o + td` is in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed to
            be on the boundary of the volume.
        """

        batch_size, *shape, _ = origin.shape
        ones = [1] * len(shape)
        bbox = self.bbox.view(1, *ones, 2, 3).to(origin.device)

        def _safe_divide(a, b, epsilon=1e-6):
            return a / torch.where(b < 0, b - epsilon, b + epsilon)

        ts = _safe_divide(bbox - origin[..., None, :], direction[..., None, :], epsilon=epsilon)

        # Cases to think about:
        #
        #   1. t1 <= t0: the ray does not pass through the AABB.
        #   2. t0 < t1 <= 0: the ray intersects but the BB is behind the origin.
        #   3. t0 <= 0 <= t1: the ray starts from inside the BB
        #   4. 0 <= t0 < t1: the ray is not inside and intersects with the BB twice.
        #
        # 1 and 4 are clearly handled from t0 < t1 below.
        # Making t0 at least min_dist (>= 0) takes care of 2 and 3.
        t0 = ts.min(dim=-2).values.max(dim=-1, keepdim=True).values.clamp(self.min_dist)
        t1 = ts.max(dim=-2).values.min(dim=-1, keepdim=True).values
        assert t0.shape == t1.shape == (batch_size, *shape, 1)
        if t0_lower is not None:
            assert t0.shape == t0_lower.shape
            t0 = torch.maximum(t0, t0_lower)

        intersected = t0 + self.min_t_range < t1
        t0 = torch.where(intersected, t0, torch.zeros_like(t0))
        t1 = torch.where(intersected, t1, torch.ones_like(t1))

        return VolumeRange(t0=t0, t1=t1, intersected=intersected)


class StratifiedRaySampler(nn.Module):
    """
    Instead of fixed intervals, a sample is drawn uniformly at random from each interval.
    """

    def __init__(self, depth_mode: str = "linear"):
        """
        :param depth_mode: linear samples ts linearly in depth. harmonic ensures
            closer points are sampled more densely.
        """
        self.depth_mode = depth_mode
        assert self.depth_mode in ("linear", "geometric", "harmonic")

    def sample(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        n_samples: int,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        """
        Args:
            t0: start time has shape [batch_size, *shape, 1]
            t1: finish time has shape [batch_size, *shape, 1]
            n_samples: number of ts to sample
        Return:
            sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        ones = [1] * (len(t0.shape) - 1)
        ts = torch.linspace(0, 1, n_samples).view(*ones, n_samples).to(t0.dtype).to(t0.device)

        if self.depth_mode == "linear":
            ts = t0 * (1.0 - ts) + t1 * ts
        elif self.depth_mode == "geometric":
            ts = (t0.clamp(epsilon).log() * (1.0 - ts) + t1.clamp(epsilon).log() * ts).exp()
        elif self.depth_mode == "harmonic":
            # The original NeRF recommends this interpolation scheme for
            # spherical scenes, but there could be some weird edge cases when
            # the observer crosses from the inner to outer volume.
            ts = 1.0 / (1.0 / t0.clamp(epsilon) * (1.0 - ts) + 1.0 / t1.clamp(epsilon) * ts)

        mids = 0.5 * (ts[..., 1:] + ts[..., :-1])
        upper = torch.cat([mids, t1], dim=-1)
        lower = torch.cat([t0, mids], dim=-1)
        # yiyi notes: add a random seed here for testing, don't forget to remove
        torch.manual_seed(0)
        t_rand = torch.rand_like(ts)

        ts = lower + (upper - lower) * t_rand
        return ts.unsqueeze(-1)


class ImportanceRaySampler(nn.Module):
    """
    Given the initial estimate of densities, this samples more from regions/bins expected to have objects.
    """

    def __init__(
        self,
        volume_range: VolumeRange,
        ts: torch.Tensor,
        weights: torch.Tensor,
        blur_pool: bool = False,
        alpha: float = 1e-5,
    ):
        """
        Args:
            volume_range: the range in which a ray intersects the given volume.
            ts: earlier samples from the coarse rendering step
            weights: discretized version of density * transmittance
            blur_pool: if true, use 2-tap max + 2-tap blur filter from mip-NeRF.
            alpha: small value to add to weights.
        """
        self.volume_range = volume_range
        self.ts = ts.clone().detach()
        self.weights = weights.clone().detach()
        self.blur_pool = blur_pool
        self.alpha = alpha

    @torch.no_grad()
    def sample(self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Args:
            t0: start time has shape [batch_size, *shape, 1]
            t1: finish time has shape [batch_size, *shape, 1]
            n_samples: number of ts to sample
        Return:
            sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        lower, upper, _ = self.volume_range.partition(self.ts)

        batch_size, *shape, n_coarse_samples, _ = self.ts.shape

        weights = self.weights
        if self.blur_pool:
            padded = torch.cat([weights[..., :1, :], weights, weights[..., -1:, :]], dim=-2)
            maxes = torch.maximum(padded[..., :-1, :], padded[..., 1:, :])
            weights = 0.5 * (maxes[..., :-1, :] + maxes[..., 1:, :])
        weights = weights + self.alpha
        pmf = weights / weights.sum(dim=-2, keepdim=True)
        inds = sample_pmf(pmf, n_samples)
        assert inds.shape == (batch_size, *shape, n_samples, 1)
        assert (inds >= 0).all() and (inds < n_coarse_samples).all()

        t_rand = torch.rand(inds.shape, device=inds.device)
        lower_ = torch.gather(lower, -2, inds)
        upper_ = torch.gather(upper, -2, inds)

        ts = lower_ + (upper_ - lower_) * t_rand
        ts = torch.sort(ts, dim=-2).values
        return ts


@dataclass
class MeshDecoderOutput(BaseOutput):
    """
    A 3D triangle mesh with optional data at the vertices and faces.

    Args:
        verts (`torch.Tensor` of shape `(N, 3)`):
            array of vertext coordinates
        faces (`torch.Tensor` of shape `(N, 3)`):
            array of triangles, pointing to indices in verts.
        vertext_channels (Dict):
            vertext coordinates for each color channel
    """

    verts: torch.Tensor
    faces: torch.Tensor
    vertex_channels: Dict[str, torch.Tensor]


class MeshDecoder(nn.Module):
    """
    Construct meshes from Signed distance functions (SDFs) using marching cubes method
    """

    def __init__(self):
        super().__init__()
        cases = torch.zeros(256, 5, 3, dtype=torch.long)
        masks = torch.zeros(256, 5, dtype=torch.bool)

        self.register_buffer("cases", cases)
        self.register_buffer("masks", masks)

    def forward(self, field: torch.Tensor, min_point: torch.Tensor, size: torch.Tensor):
        """
        For a signed distance field, produce a mesh using marching cubes.

        :param field: a 3D tensor of field values, where negative values correspond
                    to the outside of the shape. The dimensions correspond to the x, y, and z directions, respectively.
        :param min_point: a tensor of shape [3] containing the point corresponding
                        to (0, 0, 0) in the field.
        :param size: a tensor of shape [3] containing the per-axis distance from the
                    (0, 0, 0) field corner and the (-1, -1, -1) field corner.
        """
        assert len(field.shape) == 3, "input must be a 3D scalar field"
        dev = field.device

        cases = self.cases.to(dev)
        masks = self.masks.to(dev)

        min_point = min_point.to(dev)
        size = size.to(dev)

        grid_size = field.shape
        grid_size_tensor = torch.tensor(grid_size).to(size)

        # Create bitmasks between 0 and 255 (inclusive) indicating the state
        # of the eight corners of each cube.
        bitmasks = (field > 0).to(torch.uint8)
        bitmasks = bitmasks[:-1, :, :] | (bitmasks[1:, :, :] << 1)
        bitmasks = bitmasks[:, :-1, :] | (bitmasks[:, 1:, :] << 2)
        bitmasks = bitmasks[:, :, :-1] | (bitmasks[:, :, 1:] << 4)

        # Compute corner coordinates across the entire grid.
        corner_coords = torch.empty(*grid_size, 3, device=dev, dtype=field.dtype)
        corner_coords[range(grid_size[0]), :, :, 0] = torch.arange(grid_size[0], device=dev, dtype=field.dtype)[
            :, None, None
        ]
        corner_coords[:, range(grid_size[1]), :, 1] = torch.arange(grid_size[1], device=dev, dtype=field.dtype)[
            :, None
        ]
        corner_coords[:, :, range(grid_size[2]), 2] = torch.arange(grid_size[2], device=dev, dtype=field.dtype)

        # Compute all vertices across all edges in the grid, even though we will
        # throw some out later. We have (X-1)*Y*Z + X*(Y-1)*Z + X*Y*(Z-1) vertices.
        # These are all midpoints, and don't account for interpolation (which is
        # done later based on the used edge midpoints).
        edge_midpoints = torch.cat(
            [
                ((corner_coords[:-1] + corner_coords[1:]) / 2).reshape(-1, 3),
                ((corner_coords[:, :-1] + corner_coords[:, 1:]) / 2).reshape(-1, 3),
                ((corner_coords[:, :, :-1] + corner_coords[:, :, 1:]) / 2).reshape(-1, 3),
            ],
            dim=0,
        )

        # Create a flat array of [X, Y, Z] indices for each cube.
        cube_indices = torch.zeros(
            grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1, 3, device=dev, dtype=torch.long
        )
        cube_indices[range(grid_size[0] - 1), :, :, 0] = torch.arange(grid_size[0] - 1, device=dev)[:, None, None]
        cube_indices[:, range(grid_size[1] - 1), :, 1] = torch.arange(grid_size[1] - 1, device=dev)[:, None]
        cube_indices[:, :, range(grid_size[2] - 1), 2] = torch.arange(grid_size[2] - 1, device=dev)
        flat_cube_indices = cube_indices.reshape(-1, 3)

        # Create a flat array mapping each cube to 12 global edge indices.
        edge_indices = _create_flat_edge_indices(flat_cube_indices, grid_size)

        # Apply the LUT to figure out the triangles.
        flat_bitmasks = bitmasks.reshape(-1).long()  # must cast to long for indexing to believe this not a mask
        local_tris = cases[flat_bitmasks]
        local_masks = masks[flat_bitmasks]
        # Compute the global edge indices for the triangles.
        global_tris = torch.gather(edge_indices, 1, local_tris.reshape(local_tris.shape[0], -1)).reshape(
            local_tris.shape
        )
        # Select the used triangles for each cube.
        selected_tris = global_tris.reshape(-1, 3)[local_masks.reshape(-1)]

        # Now we have a bunch of indices into the full list of possible vertices,
        # but we want to reduce this list to only the used vertices.
        used_vertex_indices = torch.unique(selected_tris.view(-1))
        used_edge_midpoints = edge_midpoints[used_vertex_indices]
        old_index_to_new_index = torch.zeros(len(edge_midpoints), device=dev, dtype=torch.long)
        old_index_to_new_index[used_vertex_indices] = torch.arange(
            len(used_vertex_indices), device=dev, dtype=torch.long
        )

        # Rewrite the triangles to use the new indices
        faces = torch.gather(old_index_to_new_index, 0, selected_tris.view(-1)).reshape(selected_tris.shape)

        # Compute the actual interpolated coordinates corresponding to edge midpoints.
        v1 = torch.floor(used_edge_midpoints).to(torch.long)
        v2 = torch.ceil(used_edge_midpoints).to(torch.long)
        s1 = field[v1[:, 0], v1[:, 1], v1[:, 2]]
        s2 = field[v2[:, 0], v2[:, 1], v2[:, 2]]
        p1 = (v1.float() / (grid_size_tensor - 1)) * size + min_point
        p2 = (v2.float() / (grid_size_tensor - 1)) * size + min_point
        # The signs of s1 and s2 should be different. We want to find
        # t such that t*s2 + (1-t)*s1 = 0.
        t = (s1 / (s1 - s2))[:, None]
        verts = t * p2 + (1 - t) * p1

        return MeshDecoderOutput(verts=verts, faces=faces, vertex_channels=None)


@dataclass
class MLPNeRFModelOutput(BaseOutput):
    density: torch.Tensor
    signed_distance: torch.Tensor
    channels: torch.Tensor
    ts: torch.Tensor


class MLPNeRSTFModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        d_hidden: int = 256,
        n_output: int = 12,
        n_hidden_layers: int = 6,
        act_fn: str = "swish",
        insert_direction_at: int = 4,
    ):
        super().__init__()

        # Instantiate the MLP

        # Find out the dimension of encoded position and direction
        dummy = torch.eye(1, 3)
        d_posenc_pos = encode_position(position=dummy).shape[-1]
        d_posenc_dir = encode_direction(position=dummy).shape[-1]

        mlp_widths = [d_hidden] * n_hidden_layers
        input_widths = [d_posenc_pos] + mlp_widths
        output_widths = mlp_widths + [n_output]

        if insert_direction_at is not None:
            input_widths[insert_direction_at] += d_posenc_dir

        self.mlp = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(input_widths, output_widths)])

        if act_fn == "swish":
            # self.activation = swish
            # yiyi testing:
            self.activation = lambda x: F.silu(x)
        else:
            raise ValueError(f"Unsupported activation function {act_fn}")

        self.sdf_activation = torch.tanh
        self.density_activation = torch.nn.functional.relu
        self.channel_activation = torch.sigmoid

    def map_indices_to_keys(self, output):
        h_map = {
            "sdf": (0, 1),
            "density_coarse": (1, 2),
            "density_fine": (2, 3),
            "stf": (3, 6),
            "nerf_coarse": (6, 9),
            "nerf_fine": (9, 12),
        }

        mapped_output = {k: output[..., start:end] for k, (start, end) in h_map.items()}

        return mapped_output

    def forward(self, *, position, direction, ts, nerf_level="coarse", rendering_mode="nerf"):
        h = encode_position(position)

        h_preact = h
        h_directionless = None
        for i, layer in enumerate(self.mlp):
            if i == self.config.insert_direction_at:  # 4 in the config
                h_directionless = h_preact
                h_direction = encode_direction(position, direction=direction)
                h = torch.cat([h, h_direction], dim=-1)

            h = layer(h)

            h_preact = h

            if i < len(self.mlp) - 1:
                h = self.activation(h)

        h_final = h
        if h_directionless is None:
            h_directionless = h_preact

        activation = self.map_indices_to_keys(h_final)

        if nerf_level == "coarse":
            h_density = activation["density_coarse"]
        else:
            h_density = activation["density_fine"]

        if rendering_mode == "nerf":
            if nerf_level == "coarse":
                h_channels = activation["nerf_coarse"]
            else:
                h_channels = activation["nerf_fine"]

        elif rendering_mode == "stf":
            h_channels = activation["stf"]

        density = self.density_activation(h_density)
        signed_distance = self.sdf_activation(activation["sdf"])
        channels = self.channel_activation(h_channels)

        # yiyi notes: I think signed_distance is not used
        return MLPNeRFModelOutput(density=density, signed_distance=signed_distance, channels=channels, ts=ts)


class ChannelsProj(nn.Module):
    def __init__(
        self,
        *,
        vectors: int,
        channels: int,
        d_latent: int,
    ):
        super().__init__()
        self.proj = nn.Linear(d_latent, vectors * channels)
        self.norm = nn.LayerNorm(channels)
        self.d_latent = d_latent
        self.vectors = vectors
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bvd = x
        w_vcd = self.proj.weight.view(self.vectors, self.channels, self.d_latent)
        b_vc = self.proj.bias.view(1, self.vectors, self.channels)
        h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        h = self.norm(h)

        h = h + b_vc
        return h


class ShapEParamsProjModel(ModelMixin, ConfigMixin):
    """
    project the latent representation of a 3D asset to obtain weights of a multi-layer perceptron (MLP).

    For more details, see the original paper:
    """

    @register_to_config
    def __init__(
        self,
        *,
        param_names: Tuple[str] = (
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = (
            (256, 93),
            (256, 256),
            (256, 256),
            (256, 256),
        ),
        d_latent: int = 1024,
    ):
        super().__init__()

        # check inputs
        if len(param_names) != len(param_shapes):
            raise ValueError("Must provide same number of `param_names` as `param_shapes`")
        self.projections = nn.ModuleDict({})
        for k, (vectors, channels) in zip(param_names, param_shapes):
            self.projections[_sanitize_name(k)] = ChannelsProj(
                vectors=vectors,
                channels=channels,
                d_latent=d_latent,
            )

    def forward(self, x: torch.Tensor):
        out = {}
        start = 0
        for k, shape in zip(self.config.param_names, self.config.param_shapes):
            vectors, _ = shape
            end = start + vectors
            x_bvd = x[:, start:end]
            out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape(len(x), *shape)
            start = end
        return out


class ShapERenderer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        *,
        param_names: Tuple[str] = (
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = (
            (256, 93),
            (256, 256),
            (256, 256),
            (256, 256),
        ),
        d_latent: int = 1024,
        d_hidden: int = 256,
        n_output: int = 12,
        n_hidden_layers: int = 6,
        act_fn: str = "swish",
        insert_direction_at: int = 4,
        background: Tuple[float] = (
            255.0,
            255.0,
            255.0,
        ),
    ):
        super().__init__()

        self.params_proj = ShapEParamsProjModel(
            param_names=param_names,
            param_shapes=param_shapes,
            d_latent=d_latent,
        )
        self.mlp = MLPNeRSTFModel(d_hidden, n_output, n_hidden_layers, act_fn, insert_direction_at)
        self.void = VoidNeRFModel(background=background, channel_scale=255.0)
        self.volume = BoundingBoxVolume(bbox_max=[1.0, 1.0, 1.0], bbox_min=[-1.0, -1.0, -1.0])
        self.mesh_decoder = MeshDecoder()

    @torch.no_grad()
    def render_rays(self, rays, sampler, n_samples, prev_model_out=None, render_with_direction=False):
        """
        Perform volumetric rendering over a partition of possible t's in the union of rendering volumes (written below
        with some abuse of notations)

            C(r) := sum(
                transmittance(t[i]) * integrate(
                    lambda t: density(t) * channels(t) * transmittance(t), [t[i], t[i + 1]],
                ) for i in range(len(parts))
            ) + transmittance(t[-1]) * void_model(t[-1]).channels

        where

        1) transmittance(s) := exp(-integrate(density, [t[0], s])) calculates the probability of light passing through
        the volume specified by [t[0], s]. (transmittance of 1 means light can pass freely) 2) density and channels are
        obtained by evaluating the appropriate part.model at time t. 3) [t[i], t[i + 1]] is defined as the range of t
        where the ray intersects (parts[i].volume \\ union(part.volume for part in parts[:i])) at the surface of the
        shell (if bounded). If the ray does not intersect, the integral over this segment is evaluated as 0 and
        transmittance(t[i + 1]) := transmittance(t[i]). 4) The last term is integration to infinity (e.g. [t[-1],
        math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).

        Args:
            rays: [batch_size x ... x 2 x 3] origin and direction. sampler: disjoint volume integrals. n_samples:
            number of ts to sample. prev_model_outputs: model outputs from the previous rendering step, including

        :return: A tuple of
            - `channels`
            - A importance samplers for additional fine-grained rendering
            - raw model output
        """
        origin, direction = rays[..., 0, :], rays[..., 1, :]

        # Integrate over [t[i], t[i + 1]]

        # 1 Intersect the rays with the current volume and sample ts to integrate along.
        vrange = self.volume.intersect(origin, direction, t0_lower=None)
        ts = sampler.sample(vrange.t0, vrange.t1, n_samples)
        ts = ts.to(rays.dtype)

        if prev_model_out is not None:
            # Append the previous ts now before fprop because previous
            # rendering used a different model and we can't reuse the output.
            ts = torch.sort(torch.cat([ts, prev_model_out.ts], dim=-2), dim=-2).values

        batch_size, *_shape, _t0_dim = vrange.t0.shape
        _, *ts_shape, _ts_dim = ts.shape

        # 2. Get the points along the ray and query the model
        directions = torch.broadcast_to(direction.unsqueeze(-2), [batch_size, *ts_shape, 3])
        positions = origin.unsqueeze(-2) + ts * directions

        directions = directions.to(self.mlp.dtype)
        positions = positions.to(self.mlp.dtype)

        optional_directions = directions if render_with_direction else None

        model_out = self.mlp(
            position=positions,
            direction=optional_directions,
            ts=ts,
            nerf_level="coarse" if prev_model_out is None else "fine",
        )

        # 3. Integrate the model results
        channels, weights, transmittance = integrate_samples(
            vrange, model_out.ts, model_out.density, model_out.channels
        )

        # 4. Clean up results that do not intersect with the volume.
        transmittance = torch.where(vrange.intersected, transmittance, torch.ones_like(transmittance))
        channels = torch.where(vrange.intersected, channels, torch.zeros_like(channels))
        # 5. integration to infinity (e.g. [t[-1], math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).
        channels = channels + transmittance * self.void(origin)

        weighted_sampler = ImportanceRaySampler(vrange, ts=model_out.ts, weights=weights)

        return channels, weighted_sampler, model_out

    @torch.no_grad()
    def decode_to_image(
        self,
        latents,
        device,
        size: int = 64,
        ray_batch_size: int = 4096,
        n_coarse_samples=64,
        n_fine_samples=128,
    ):
        # project the parameters from the generated latents
        projected_params = self.params_proj(latents)

        # update the mlp layers of the renderer
        for name, param in self.mlp.state_dict().items():
            if f"nerstf.{name}" in projected_params.keys():
                param.copy_(projected_params[f"nerstf.{name}"].squeeze(0))

        # create cameras object
        camera = create_pan_cameras(size)
        rays = camera.camera_rays
        rays = rays.to(device)
        n_batches = rays.shape[1] // ray_batch_size

        coarse_sampler = StratifiedRaySampler()

        images = []

        for idx in range(n_batches):
            rays_batch = rays[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]

            # render rays with coarse, stratified samples.
            _, fine_sampler, coarse_model_out = self.render_rays(rays_batch, coarse_sampler, n_coarse_samples)
            # Then, render with additional importance-weighted ray samples.
            channels, _, _ = self.render_rays(
                rays_batch, fine_sampler, n_fine_samples, prev_model_out=coarse_model_out
            )

            images.append(channels)

        images = torch.cat(images, dim=1)
        images = images.view(*camera.shape, camera.height, camera.width, -1).squeeze(0)

        return images

    @torch.no_grad()
    def decode_to_mesh(
        self,
        latents,
        device,
        grid_size: int = 128,
        query_batch_size: int = 4096,
        texture_channels: Tuple = ("R", "G", "B"),
    ):
        # 1. project the parameters from the generated latents
        projected_params = self.params_proj(latents)

        # 2. update the mlp layers of the renderer
        for name, param in self.mlp.state_dict().items():
            if f"nerstf.{name}" in projected_params.keys():
                param.copy_(projected_params[f"nerstf.{name}"].squeeze(0))

        # 3. decoding with STF rendering
        # 3.1 query the SDF values at vertices along a regular 128**3 grid

        query_points = volume_query_points(self.volume, grid_size)
        query_positions = query_points[None].repeat(1, 1, 1).to(device=device, dtype=self.mlp.dtype)

        fields = []

        for idx in range(0, query_positions.shape[1], query_batch_size):
            query_batch = query_positions[:, idx : idx + query_batch_size]

            model_out = self.mlp(
                position=query_batch, direction=None, ts=None, nerf_level="fine", rendering_mode="stf"
            )
            fields.append(model_out.signed_distance)

        # predicted SDF values
        fields = torch.cat(fields, dim=1)
        fields = fields.float()

        assert len(fields.shape) == 3 and fields.shape[-1] == 1, (
            f"expected [meta_batch x inner_batch] SDF results, but got {fields.shape}"
        )

        fields = fields.reshape(1, *([grid_size] * 3))

        # create grid 128 x 128 x 128
        # - force a negative border around the SDFs to close off all the models.
        full_grid = torch.zeros(
            1,
            grid_size + 2,
            grid_size + 2,
            grid_size + 2,
            device=fields.device,
            dtype=fields.dtype,
        )
        full_grid.fill_(-1.0)
        full_grid[:, 1:-1, 1:-1, 1:-1] = fields
        fields = full_grid

        # apply a differentiable implementation of Marching Cubes to construct meshs
        raw_meshes = []
        mesh_mask = []

        for field in fields:
            raw_mesh = self.mesh_decoder(field, self.volume.bbox_min, self.volume.bbox_max - self.volume.bbox_min)
            mesh_mask.append(True)
            raw_meshes.append(raw_mesh)

        mesh_mask = torch.tensor(mesh_mask, device=fields.device)
        max_vertices = max(len(m.verts) for m in raw_meshes)

        # 3.2. query the texture color head at each vertex of the resulting mesh.
        texture_query_positions = torch.stack(
            [m.verts[torch.arange(0, max_vertices) % len(m.verts)] for m in raw_meshes],
            dim=0,
        )
        texture_query_positions = texture_query_positions.to(device=device, dtype=self.mlp.dtype)

        textures = []

        for idx in range(0, texture_query_positions.shape[1], query_batch_size):
            query_batch = texture_query_positions[:, idx : idx + query_batch_size]

            texture_model_out = self.mlp(
                position=query_batch, direction=None, ts=None, nerf_level="fine", rendering_mode="stf"
            )
            textures.append(texture_model_out.channels)

        # predict texture color
        textures = torch.cat(textures, dim=1)

        textures = _convert_srgb_to_linear(textures)
        textures = textures.float()

        # 3.3 augment the mesh with texture data
        assert len(textures.shape) == 3 and textures.shape[-1] == len(texture_channels), (
            f"expected [meta_batch x inner_batch x texture_channels] field results, but got {textures.shape}"
        )

        for m, texture in zip(raw_meshes, textures):
            texture = texture[: len(m.verts)]
            m.vertex_channels = dict(zip(texture_channels, texture.unbind(-1)))

        return raw_meshes[0]
