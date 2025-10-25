# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import get_logger
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import AutoencoderMixin, DecoderOutput, IdentityDistribution


logger = get_logger(__name__)


# fmt: off
# These latents and means are from CV8x8x8-1.0. Each checkpoint has different values, but since this is the main VAE used,
# we will default to these values.
LATENTS_MEAN = [0.11362758, -0.0171717, 0.03071163, 0.02046862, 0.01931456, 0.02138567, 0.01999342, 0.02189187, 0.02011935, 0.01872694, 0.02168613, 0.02207148, 0.01986941, 0.01770413, 0.02067643, 0.02028245, 0.19125476, 0.04556972, 0.0595558, 0.05315534, 0.05496629, 0.05356264, 0.04856596, 0.05327453, 0.05410472, 0.05597149, 0.05524866, 0.05181874, 0.05071663, 0.05204537, 0.0564108, 0.05518042, 0.01306714, 0.03341161, 0.03847246, 0.02810185, 0.02790166, 0.02920026, 0.02823597, 0.02631033, 0.0278531, 0.02880507, 0.02977769, 0.03145441, 0.02888389, 0.03280773, 0.03484927, 0.03049198, -0.00197727, 0.07534957, 0.04963879, 0.05530893, 0.05410828, 0.05252541, 0.05029899, 0.05321025, 0.05149245, 0.0511921, 0.04643495, 0.04604527, 0.04631618, 0.04404101, 0.04403536, 0.04499495, -0.02994183, -0.04787003, -0.01064558, -0.01779824, -0.01490502, -0.02157517, -0.0204778, -0.02180816, -0.01945375, -0.02062863, -0.02192209, -0.02520639, -0.02246656, -0.02427533, -0.02683363, -0.02762006, 0.08019473, -0.13005368, -0.07568636, -0.06082374, -0.06036175, -0.05875364, -0.05921887, -0.05869788, -0.05273941, -0.052565, -0.05346428, -0.05456541, -0.053657, -0.05656897, -0.05728589, -0.05321847, 0.16718403, -0.00390146, 0.0379406, 0.0356561, 0.03554131, 0.03924074, 0.03873615, 0.04187329, 0.04226924, 0.04378717, 0.04684274, 0.05117614, 0.04547792, 0.05251586, 0.05048339, 0.04950784, 0.09564418, 0.0547128, 0.08183969, 0.07978633, 0.08076023, 0.08108605, 0.08011818, 0.07965573, 0.08187773, 0.08350263, 0.08101469, 0.0786941, 0.0774442, 0.07724521, 0.07830418, 0.07599796, -0.04987567, 0.05923908, -0.01058746, -0.01177603, -0.01116162, -0.01364149, -0.01546014, -0.0117213, -0.01780043, -0.01648314, -0.02100247, -0.02104417, -0.02482123, -0.02611689, -0.02561143, -0.02597336, -0.05364667, 0.08211684, 0.04686937, 0.04605641, 0.04304186, 0.0397355, 0.03686767, 0.04087112, 0.03704741, 0.03706401, 0.03120073, 0.03349091, 0.03319963, 0.03205781, 0.03195127, 0.03180481, 0.16427967, -0.11048453, -0.04595276, -0.04982893, -0.05213465, -0.04809378, -0.05080318, -0.04992863, -0.04493337, -0.0467619, -0.04884703, -0.04627892, -0.04913311, -0.04955709, -0.04533982, -0.04570218, -0.10612928, -0.05121198, -0.06761009, -0.07251801, -0.07265285, -0.07417855, -0.07202412, -0.07499027, -0.07625481, -0.07535747, -0.07638787, -0.07920305, -0.07596069, -0.07959418, -0.08265036, -0.07955471, -0.16888915, 0.0753242, 0.04062594, 0.03375093, 0.03337452, 0.03699376, 0.03651138, 0.03611023, 0.03555622, 0.03378554, 0.0300498, 0.03395559, 0.02941847, 0.03156432, 0.03431173, 0.03016853, -0.03415358, -0.01699573, -0.04029295, -0.04912157, -0.0498858, -0.04917918, -0.04918056, -0.0525189, -0.05325506, -0.05341973, -0.04983329, -0.04883146, -0.04985548, -0.04736718, -0.0462027, -0.04836091, 0.02055675, 0.03419799, -0.02907669, -0.04350509, -0.04156144, -0.04234421, -0.04446109, -0.04461774, -0.04882839, -0.04822346, -0.04502493, -0.0506244, -0.05146913, -0.04655267, -0.04862994, -0.04841615, 0.20312774, -0.07208502, -0.03635615, -0.03556088, -0.04246174, -0.04195838, -0.04293778, -0.04071276, -0.04240569, -0.04125213, -0.04395144, -0.03959096, -0.04044993, -0.04015875, -0.04088107, -0.03885176]
LATENTS_STD = [0.56700271, 0.65488982, 0.65589428, 0.66524369, 0.66619784, 0.6666382, 0.6720838, 0.66955978, 0.66928875, 0.67108786, 0.67092526, 0.67397463, 0.67894882, 0.67668313, 0.67769569, 0.67479557, 0.85245121, 0.8688373, 0.87348086, 0.88459337, 0.89135885, 0.8910504, 0.89714909, 0.89947474, 0.90201765, 0.90411824, 0.90692616, 0.90847772, 0.90648711, 0.91006982, 0.91033435, 0.90541548, 0.84960359, 0.85863352, 0.86895317, 0.88460612, 0.89245003, 0.89451706, 0.89931005, 0.90647358, 0.90338236, 0.90510076, 0.91008312, 0.90961218, 0.9123717, 0.91313171, 0.91435546, 0.91565102, 0.91877103, 0.85155135, 0.857804, 0.86998034, 0.87365264, 0.88161767, 0.88151032, 0.88758916, 0.89015514, 0.89245576, 0.89276224, 0.89450496, 0.90054202, 0.89994133, 0.90136105, 0.90114892, 0.77755755, 0.81456852, 0.81911844, 0.83137071, 0.83820474, 0.83890373, 0.84401101, 0.84425181, 0.84739357, 0.84798753, 0.85249585, 0.85114998, 0.85160935, 0.85626358, 0.85677862, 0.85641026, 0.69903517, 0.71697885, 0.71696913, 0.72583169, 0.72931731, 0.73254126, 0.73586977, 0.73734969, 0.73664582, 0.74084908, 0.74399322, 0.74471819, 0.74493188, 0.74824578, 0.75024873, 0.75274801, 0.8187142, 0.82251883, 0.82616025, 0.83164483, 0.84072375, 0.8396467, 0.84143305, 0.84880769, 0.8503468, 0.85196948, 0.85211051, 0.85386664, 0.85410017, 0.85439342, 0.85847849, 0.85385275, 0.67583984, 0.68259847, 0.69198853, 0.69928843, 0.70194328, 0.70467001, 0.70755547, 0.70917857, 0.71007699, 0.70963502, 0.71064079, 0.71027333, 0.71291167, 0.71537536, 0.71902508, 0.71604162, 0.72450989, 0.71979928, 0.72057378, 0.73035461, 0.73329622, 0.73660028, 0.73891461, 0.74279994, 0.74105692, 0.74002433, 0.74257588, 0.74416119, 0.74543899, 0.74694443, 0.74747062, 0.74586403, 0.90176988, 0.90990674, 0.91106802, 0.92163783, 0.92390233, 0.93056196, 0.93482202, 0.93642414, 0.93858379, 0.94064975, 0.94078934, 0.94325715, 0.94955301, 0.94814706, 0.95144123, 0.94923073, 0.49853548, 0.64968109, 0.6427654, 0.64966393, 0.6487664, 0.65203559, 0.6584242, 0.65351611, 0.65464371, 0.6574859, 0.65626335, 0.66123748, 0.66121179, 0.66077942, 0.66040152, 0.66474909, 0.61986589, 0.69138134, 0.6884557, 0.6955843, 0.69765401, 0.70015347, 0.70529598, 0.70468754, 0.70399523, 0.70479989, 0.70887572, 0.71126866, 0.7097227, 0.71249932, 0.71231949, 0.71175605, 0.35586974, 0.68723857, 0.68973219, 0.69958478, 0.6943453, 0.6995818, 0.70980215, 0.69899458, 0.70271689, 0.70095056, 0.69912851, 0.70522696, 0.70392174, 0.70916915, 0.70585734, 0.70373541, 0.98101336, 0.89024764, 0.89607251, 0.90678179, 0.91308665, 0.91812348, 0.91980827, 0.92480654, 0.92635667, 0.92887944, 0.93338072, 0.93468094, 0.93619436, 0.93906063, 0.94191772, 0.94471723, 0.83202779, 0.84106231, 0.84463632, 0.85829508, 0.86319661, 0.86751342, 0.86914337, 0.87085921, 0.87286359, 0.87537396, 0.87931138, 0.88054478, 0.8811838, 0.88872558, 0.88942474, 0.88934827, 0.44025335, 0.63061613, 0.63110614, 0.63601959, 0.6395812, 0.64104342, 0.65019929, 0.6502797, 0.64355946, 0.64657205, 0.64847094, 0.64728117, 0.64972943, 0.65162975, 0.65328044, 0.64914775]
_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
# fmt: on


class CosmosCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: int = 1,
        pad_mode: str = "constant",
    ) -> None:
        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        stride = (stride, stride, stride) if isinstance(stride, int) else stride

        _, height_kernel_size, width_kernel_size = kernel_size
        assert height_kernel_size % 2 == 1 and width_kernel_size % 2 == 1

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.pad_mode = pad_mode
        self.temporal_pad = dilation[0] * (kernel_size[0] - 1) + (1 - stride[0])
        self.spatial_pad = (padding, padding, padding, padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_prev = hidden_states[:, :, :1, ...].repeat(1, 1, self.temporal_pad, 1, 1)
        hidden_states = torch.cat([hidden_states_prev, hidden_states], dim=2)
        hidden_states = F.pad(hidden_states, (*self.spatial_pad, 0, 0), mode=self.pad_mode, value=0.0)
        return super().forward(hidden_states)


class CosmosCausalGroupNorm(torch.nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.num_groups = num_groups

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_groups == 1:
            batch_size = hidden_states.size(0)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B, C, T, H, W] -> [B * T, C, H, W]
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )  # [B * T, C, H, W] -> [B, C, T, H, W]
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class CosmosPatchEmbed3d(nn.Module):
    def __init__(self, patch_size: int = 1, patch_method: str = "haar") -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_method = patch_method

        wavelets = _WAVELETS.get(patch_method).clone()
        arange = torch.arange(wavelets.shape[0])

        self.register_buffer("wavelets", wavelets, persistent=False)
        self.register_buffer("_arange", arange, persistent=False)

    def _dwt(self, hidden_states: torch.Tensor, mode: str = "reflect", rescale=False) -> torch.Tensor:
        dtype = hidden_states.dtype
        wavelets = self.wavelets

        n = wavelets.shape[0]
        g = hidden_states.shape[1]
        hl = wavelets.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (wavelets * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis
        hidden_states = F.pad(hidden_states, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode).to(
            dtype
        )
        xl = F.conv3d(hidden_states, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(hidden_states, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        hidden_states = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            hidden_states = hidden_states / 8**0.5
        return hidden_states

    def _haar(self, hidden_states: torch.Tensor) -> torch.Tensor:
        xi, xv = torch.split(hidden_states, [1, hidden_states.shape[2] - 1], dim=2)
        hidden_states = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        for _ in range(int(math.log2(self.patch_size))):
            hidden_states = self._dwt(hidden_states, rescale=True)
        return hidden_states

    def _arrange(self, hidden_states: torch.Tensor) -> torch.Tensor:
        xi, xv = torch.split(hidden_states, [1, hidden_states.shape[2] - 1], dim=2)
        hidden_states = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size, num_channels, num_frames // p, p, height // p, p, width // p, p
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4).contiguous()
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.patch_method == "haar":
            return self._haar(hidden_states)
        elif self.patch_method == "rearrange":
            return self._arrange(hidden_states)
        else:
            raise ValueError(f"Unsupported patch method: {self.patch_method}")


class CosmosUnpatcher3d(nn.Module):
    def __init__(self, patch_size: int = 1, patch_method: str = "haar"):
        super().__init__()

        self.patch_size = patch_size
        self.patch_method = patch_method

        wavelets = _WAVELETS.get(patch_method).clone()
        arange = torch.arange(wavelets.shape[0])

        self.register_buffer("wavelets", wavelets, persistent=False)
        self.register_buffer("_arange", arange, persistent=False)

    def _idwt(self, hidden_states: torch.Tensor, rescale: bool = False) -> torch.Tensor:
        device = hidden_states.device
        dtype = hidden_states.dtype
        h = self.wavelets.to(device)

        g = hidden_states.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange.to(device))).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(hidden_states, 8, dim=1)

        # Handle height transposed convolutions
        xll = F.conv_transpose3d(xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xll = F.conv_transpose3d(xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)) + xll

        xlh = F.conv_transpose3d(xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlh = F.conv_transpose3d(xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)) + xlh

        xhl = F.conv_transpose3d(xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhl = F.conv_transpose3d(xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)) + xhl

        xhh = F.conv_transpose3d(xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhh = F.conv_transpose3d(xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)) + xhh

        # Handles width transposed convolutions
        xl = F.conv_transpose3d(xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xl = F.conv_transpose3d(xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)) + xl
        xh = F.conv_transpose3d(xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh = F.conv_transpose3d(xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)) + xh

        # Handles time axis transposed convolutions
        hidden_states = F.conv_transpose3d(xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        hidden_states = (
            F.conv_transpose3d(xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)) + hidden_states
        )

        if rescale:
            hidden_states = hidden_states * 8**0.5

        return hidden_states

    def _ihaar(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for _ in range(int(math.log2(self.patch_size))):
            hidden_states = self._idwt(hidden_states, rescale=True)
        hidden_states = hidden_states[:, :, self.patch_size - 1 :, ...]
        return hidden_states

    def _irearrange(self, hidden_states: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        hidden_states = hidden_states.unflatten(1, (-1, p, p, p))
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        hidden_states = hidden_states[:, :, p - 1 :, ...]
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.patch_method == "haar":
            return self._ihaar(hidden_states)
        elif self.patch_method == "rearrange":
            return self._irearrange(hidden_states)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)


class CosmosConvProjection3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv_s = CosmosCausalConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1)
        self.conv_t = CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_s(hidden_states)
        hidden_states = self.conv_t(hidden_states)
        return hidden_states


class CosmosResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_groups: int = 1,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = CosmosCausalGroupNorm(in_channels, num_groups)
        self.conv1 = CosmosConvProjection3d(in_channels, out_channels)

        self.norm2 = CosmosCausalGroupNorm(out_channels, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CosmosConvProjection3d(out_channels, out_channels)

        if in_channels != out_channels:
            self.conv_shortcut = CosmosCausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        residual = self.conv_shortcut(residual)

        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states + residual


class CosmosDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_downsample: bool = True,
        temporal_downsample: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample

        self.conv1 = nn.Identity()
        self.conv2 = nn.Identity()
        self.conv3 = nn.Identity()

        if spatial_downsample:
            self.conv1 = CosmosCausalConv3d(
                in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0
            )
        if temporal_downsample:
            self.conv2 = CosmosCausalConv3d(
                in_channels, in_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0
            )
        if spatial_downsample or temporal_downsample:
            self.conv3 = CosmosCausalConv3d(
                in_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.spatial_downsample and not self.temporal_downsample:
            return hidden_states

        if self.spatial_downsample:
            pad = (0, 1, 0, 1, 0, 0)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
            conv_out = self.conv1(hidden_states)
            pool_out = F.avg_pool3d(hidden_states, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            hidden_states = conv_out + pool_out

        if self.temporal_downsample:
            hidden_states = torch.cat([hidden_states[:, :, :1, ...], hidden_states], dim=2)
            conv_out = self.conv2(hidden_states)
            pool_out = F.avg_pool3d(hidden_states, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            hidden_states = conv_out + pool_out

        hidden_states = self.conv3(hidden_states)
        return hidden_states


class CosmosUpsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_upsample: bool = True,
        temporal_upsample: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample

        self.conv1 = nn.Identity()
        self.conv2 = nn.Identity()
        self.conv3 = nn.Identity()

        if temporal_upsample:
            self.conv1 = CosmosCausalConv3d(
                in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=0
            )
        if spatial_upsample:
            self.conv2 = CosmosCausalConv3d(
                in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=1
            )
        if spatial_upsample or temporal_upsample:
            self.conv3 = CosmosCausalConv3d(
                in_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.spatial_upsample and not self.temporal_upsample:
            return hidden_states

        if self.temporal_upsample:
            num_frames = hidden_states.size(2)
            time_factor = int(1.0 + 1.0 * (num_frames > 1))
            hidden_states = hidden_states.repeat_interleave(int(time_factor), dim=2)
            hidden_states = hidden_states[..., time_factor - 1 :, :, :]
            hidden_states = self.conv1(hidden_states) + hidden_states

        if self.spatial_upsample:
            hidden_states = hidden_states.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            hidden_states = self.conv2(hidden_states) + hidden_states

        hidden_states = self.conv3(hidden_states)
        return hidden_states


class CosmosCausalAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_groups: int = 1,
        dropout: float = 0.0,
        processor: Union["CosmosSpatialAttentionProcessor2_0", "CosmosTemporalAttentionProcessor2_0"] = None,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads

        self.norm = CosmosCausalGroupNorm(attention_head_dim, num_groups=num_groups)
        self.to_q = CosmosCausalConv3d(attention_head_dim, attention_head_dim, kernel_size=1, stride=1, padding=0)
        self.to_k = CosmosCausalConv3d(attention_head_dim, attention_head_dim, kernel_size=1, stride=1, padding=0)
        self.to_v = CosmosCausalConv3d(attention_head_dim, attention_head_dim, kernel_size=1, stride=1, padding=0)
        self.to_out = nn.ModuleList([])
        self.to_out.append(
            CosmosCausalConv3d(attention_head_dim, attention_head_dim, kernel_size=1, stride=1, padding=0)
        )
        self.to_out.append(nn.Dropout(dropout))

        self.processor = processor
        if self.processor is None:
            raise ValueError("CosmosCausalAttention requires a processor.")

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.processor(self, hidden_states=hidden_states, attention_mask=attention_mask)


class CosmosSpatialAttentionProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CosmosSpatialAttentionProcessor2_0 requires PyTorch 2.0 or higher. To use it, please upgrade PyTorch."
            )

    def __call__(
        self, attn: CosmosCausalAttention, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = attn.norm(hidden_states)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # [B, C, T, H, W] -> [B * T, H * W, C]
        query = query.permute(0, 2, 3, 4, 1).flatten(2, 3).flatten(0, 1)
        key = key.permute(0, 2, 3, 4, 1).flatten(2, 3).flatten(0, 1)
        value = value.permute(0, 2, 3, 4, 1).flatten(2, 3).flatten(0, 1)

        # [B * T, H * W, C] -> [B * T, N, H * W, C // N]
        query = query.unflatten(2, (attn.num_attention_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.num_attention_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.num_attention_heads, -1)).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = hidden_states.unflatten(1, (height, width)).unflatten(0, (batch_size, num_frames))
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states + residual


class CosmosTemporalAttentionProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CosmosSpatialAttentionProcessor2_0 requires PyTorch 2.0 or higher. To use it, please upgrade PyTorch."
            )

    def __call__(
        self, attn: CosmosCausalAttention, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = attn.norm(hidden_states)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # [B, C, T, H, W] -> [B * T, H * W, C]
        query = query.permute(0, 3, 4, 2, 1).flatten(0, 2)
        key = key.permute(0, 3, 4, 2, 1).flatten(0, 2)
        value = value.permute(0, 3, 4, 2, 1).flatten(0, 2)

        # [B * T, H * W, C] -> [B * T, N, H * W, C // N]
        query = query.unflatten(2, (attn.num_attention_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.num_attention_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.num_attention_heads, -1)).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = hidden_states.unflatten(0, (batch_size, height, width))
        hidden_states = hidden_states.permute(0, 4, 3, 1, 2)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states + residual


class CosmosDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        use_attention: bool,
        use_downsample: bool,
        spatial_downsample: bool,
        temporal_downsample: bool,
    ) -> None:
        super().__init__()

        resnets, attentions, temp_attentions = [], [], []
        in_channel, out_channel = in_channels, out_channels

        for _ in range(num_layers):
            resnets.append(CosmosResnetBlock3d(in_channel, out_channel, dropout, num_groups=1))
            in_channel = out_channel

            if use_attention:
                attentions.append(
                    CosmosCausalAttention(
                        num_attention_heads=1,
                        attention_head_dim=out_channel,
                        num_groups=1,
                        dropout=dropout,
                        processor=CosmosSpatialAttentionProcessor2_0(),
                    )
                )
                temp_attentions.append(
                    CosmosCausalAttention(
                        num_attention_heads=1,
                        attention_head_dim=out_channel,
                        num_groups=1,
                        dropout=dropout,
                        processor=CosmosTemporalAttentionProcessor2_0(),
                    )
                )
            else:
                attentions.append(None)
                temp_attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        self.downsamplers = None
        if use_downsample:
            self.downsamplers = nn.ModuleList([])
            self.downsamplers.append(CosmosDownsample3d(out_channel, spatial_downsample, temporal_downsample))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet, attention, temp_attention in zip(self.resnets, self.attentions, self.temp_attentions):
            hidden_states = resnet(hidden_states)
            if attention is not None:
                hidden_states = attention(hidden_states)
            if temp_attention is not None:
                num_frames = hidden_states.size(2)
                attention_mask = torch.tril(hidden_states.new_ones(num_frames, num_frames)).bool()
                hidden_states = temp_attention(hidden_states, attention_mask)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class CosmosMidBlock3d(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, dropout: float, num_groups: int = 1) -> None:
        super().__init__()

        resnets, attentions, temp_attentions = [], [], []

        resnets.append(CosmosResnetBlock3d(in_channels, in_channels, dropout, num_groups))
        for _ in range(num_layers):
            attentions.append(
                CosmosCausalAttention(
                    num_attention_heads=1,
                    attention_head_dim=in_channels,
                    num_groups=num_groups,
                    dropout=dropout,
                    processor=CosmosSpatialAttentionProcessor2_0(),
                )
            )
            temp_attentions.append(
                CosmosCausalAttention(
                    num_attention_heads=1,
                    attention_head_dim=in_channels,
                    num_groups=num_groups,
                    dropout=dropout,
                    processor=CosmosTemporalAttentionProcessor2_0(),
                )
            )
            resnets.append(CosmosResnetBlock3d(in_channels, in_channels, dropout, num_groups))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attention, temp_attention, resnet in zip(self.attentions, self.temp_attentions, self.resnets[1:]):
            num_frames = hidden_states.size(2)
            attention_mask = torch.tril(hidden_states.new_ones(num_frames, num_frames)).bool()

            hidden_states = attention(hidden_states)
            hidden_states = temp_attention(hidden_states, attention_mask)
            hidden_states = resnet(hidden_states)

        return hidden_states


class CosmosUpBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        use_attention: bool,
        use_upsample: bool,
        spatial_upsample: bool,
        temporal_upsample: bool,
    ) -> None:
        super().__init__()

        resnets, attention, temp_attentions = [], [], []
        in_channel, out_channel = in_channels, out_channels

        for _ in range(num_layers):
            resnets.append(CosmosResnetBlock3d(in_channel, out_channel, dropout, num_groups=1))
            in_channel = out_channel

            if use_attention:
                attention.append(
                    CosmosCausalAttention(
                        num_attention_heads=1,
                        attention_head_dim=out_channel,
                        num_groups=1,
                        dropout=dropout,
                        processor=CosmosSpatialAttentionProcessor2_0(),
                    )
                )
                temp_attentions.append(
                    CosmosCausalAttention(
                        num_attention_heads=1,
                        attention_head_dim=out_channel,
                        num_groups=1,
                        dropout=dropout,
                        processor=CosmosTemporalAttentionProcessor2_0(),
                    )
                )
            else:
                attention.append(None)
                temp_attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attention)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        self.upsamplers = None
        if use_upsample:
            self.upsamplers = nn.ModuleList([])
            self.upsamplers.append(CosmosUpsample3d(out_channel, spatial_upsample, temporal_upsample))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet, attention, temp_attention in zip(self.resnets, self.attentions, self.temp_attentions):
            hidden_states = resnet(hidden_states)
            if attention is not None:
                hidden_states = attention(hidden_states)
            if temp_attention is not None:
                num_frames = hidden_states.size(2)
                attention_mask = torch.tril(hidden_states.new_ones(num_frames, num_frames)).bool()
                hidden_states = temp_attention(hidden_states, attention_mask)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CosmosEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        num_resnet_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32,),
        resolution: int = 1024,
        patch_size: int = 4,
        patch_type: str = "haar",
        dropout: float = 0.0,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 8,
    ) -> None:
        super().__init__()
        inner_dim = in_channels * patch_size**3
        num_spatial_layers = int(math.log2(spatial_compression_ratio)) - int(math.log2(patch_size))
        num_temporal_layers = int(math.log2(temporal_compression_ratio)) - int(math.log2(patch_size))

        # 1. Input patching & projection
        self.patch_embed = CosmosPatchEmbed3d(patch_size, patch_type)

        self.conv_in = CosmosConvProjection3d(inner_dim, block_out_channels[0])

        # 2. Down blocks
        current_resolution = resolution // patch_size
        down_blocks = []
        for i in range(len(block_out_channels) - 1):
            in_channel = block_out_channels[i]
            out_channel = block_out_channels[i + 1]

            use_attention = current_resolution in attention_resolutions
            spatial_downsample = temporal_downsample = False
            if i < len(block_out_channels) - 2:
                use_downsample = True
                spatial_downsample = i < num_spatial_layers
                temporal_downsample = i < num_temporal_layers
                current_resolution = current_resolution // 2
            else:
                use_downsample = False

            down_blocks.append(
                CosmosDownBlock3d(
                    in_channel,
                    out_channel,
                    num_resnet_blocks,
                    dropout,
                    use_attention,
                    use_downsample,
                    spatial_downsample,
                    temporal_downsample,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # 3. Mid block
        self.mid_block = CosmosMidBlock3d(block_out_channels[-1], num_layers=1, dropout=dropout, num_groups=1)

        # 4. Output norm & projection
        self.norm_out = CosmosCausalGroupNorm(block_out_channels[-1], num_groups=1)
        self.conv_out = CosmosConvProjection3d(block_out_channels[-1], out_channels)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid_block, hidden_states)
        else:
            for block in self.down_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.mid_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class CosmosDecoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        num_resnet_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (32,),
        resolution: int = 1024,
        patch_size: int = 4,
        patch_type: str = "haar",
        dropout: float = 0.0,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 8,
    ) -> None:
        super().__init__()
        inner_dim = out_channels * patch_size**3
        num_spatial_layers = int(math.log2(spatial_compression_ratio)) - int(math.log2(patch_size))
        num_temporal_layers = int(math.log2(temporal_compression_ratio)) - int(math.log2(patch_size))
        reversed_block_out_channels = list(reversed(block_out_channels))

        # 1. Input projection
        self.conv_in = CosmosConvProjection3d(in_channels, reversed_block_out_channels[0])

        # 2. Mid block
        self.mid_block = CosmosMidBlock3d(reversed_block_out_channels[0], num_layers=1, dropout=dropout, num_groups=1)

        # 3. Up blocks
        current_resolution = (resolution // patch_size) // 2 ** (len(block_out_channels) - 2)
        up_blocks = []
        for i in range(len(block_out_channels) - 1):
            in_channel = reversed_block_out_channels[i]
            out_channel = reversed_block_out_channels[i + 1]

            use_attention = current_resolution in attention_resolutions
            spatial_upsample = temporal_upsample = False
            if i < len(block_out_channels) - 2:
                use_upsample = True
                temporal_upsample = 0 < i < num_temporal_layers + 1
                spatial_upsample = temporal_upsample or (
                    i < num_spatial_layers and num_spatial_layers > num_temporal_layers
                )
                current_resolution = current_resolution * 2
            else:
                use_upsample = False

            up_blocks.append(
                CosmosUpBlock3d(
                    in_channel,
                    out_channel,
                    num_resnet_blocks + 1,
                    dropout,
                    use_attention,
                    use_upsample,
                    spatial_upsample,
                    temporal_upsample,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # 4. Output norm & projection & unpatching
        self.norm_out = CosmosCausalGroupNorm(reversed_block_out_channels[-1], num_groups=1)
        self.conv_out = CosmosConvProjection3d(reversed_block_out_channels[-1], inner_dim)

        self.unpatch_embed = CosmosUnpatcher3d(patch_size, patch_type)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states)

        for block in self.up_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = self.unpatch_embed(hidden_states)
        return hidden_states


class AutoencoderKLCosmos(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    Autoencoder used in [Cosmos](https://huggingface.co/papers/2501.03575).

    Args:
        in_channels (`int`, defaults to `3`):
            Number of input channels.
        out_channels (`int`, defaults to `3`):
            Number of output channels.
        latent_channels (`int`, defaults to `16`):
            Number of latent channels.
        encoder_block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            Number of output channels for each encoder down block.
        decode_block_out_channels (`Tuple[int, ...]`, defaults to `(256, 512, 512, 512)`):
            Number of output channels for each decoder up block.
        attention_resolutions (`Tuple[int, ...]`, defaults to `(32,)`):
            List of image/video resolutions at which to apply attention.
        resolution (`int`, defaults to `1024`):
            Base image/video resolution used for computing whether a block should have attention layers.
        num_layers (`int`, defaults to `2`):
            Number of resnet blocks in each encoder/decoder block.
        patch_size (`int`, defaults to `4`):
            Patch size used for patching the input image/video.
        patch_type (`str`, defaults to `haar`):
            Patch type used for patching the input image/video. Can be either `haar` or `rearrange`.
        scaling_factor (`float`, defaults to `1.0`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) paper. Not applicable in
            Cosmos, but we default to 1.0 for consistency.
        spatial_compression_ratio (`int`, defaults to `8`):
            The spatial compression ratio to apply in the VAE. The number of downsample blocks is determined using
            this.
        temporal_compression_ratio (`int`, defaults to `8`):
            The temporal compression ratio to apply in the VAE. The number of downsample blocks is determined using
            this.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decode_block_out_channels: Tuple[int, ...] = (256, 512, 512, 512),
        attention_resolutions: Tuple[int, ...] = (32,),
        resolution: int = 1024,
        num_layers: int = 2,
        patch_size: int = 4,
        patch_type: str = "haar",
        scaling_factor: float = 1.0,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 8,
        latents_mean: Optional[List[float]] = LATENTS_MEAN,
        latents_std: Optional[List[float]] = LATENTS_STD,
    ) -> None:
        super().__init__()

        self.encoder = CosmosEncoder3d(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
            num_resnet_blocks=num_layers,
            attention_resolutions=attention_resolutions,
            resolution=resolution,
            patch_size=patch_size,
            patch_type=patch_type,
            spatial_compression_ratio=spatial_compression_ratio,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = CosmosDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decode_block_out_channels,
            num_resnet_blocks=num_layers,
            attention_resolutions=attention_resolutions,
            resolution=resolution,
            patch_size=patch_size,
            patch_type=patch_type,
            spatial_compression_ratio=spatial_compression_ratio,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        self.quant_conv = CosmosCausalConv3d(latent_channels, latent_channels, kernel_size=1, padding=0)
        self.post_quant_conv = CosmosCausalConv3d(latent_channels, latent_channels, kernel_size=1, padding=0)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.num_latent_frames_batch_sizes`), the memory requirement can be lowered.
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # This can be configured based on the amount of GPU memory available.
        # `16` for sample frames and `2` for latent frames are sensible defaults for consumer GPUs.
        # Setting it to higher values results in higher memory usage.
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = IdentityDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[Tuple[torch.Tensor], DecoderOutput]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
