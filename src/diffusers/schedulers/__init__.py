# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from collections import OrderedDict
from enum import Enum

from ..utils import is_flax_available, is_scipy_available, is_torch_available


if is_torch_available():
    from .scheduling_ddim import DDIMScheduler
    from .scheduling_ddpm import DDPMScheduler
    from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
    from .scheduling_euler_discrete import EulerDiscreteScheduler
    from .scheduling_ipndm import IPNDMScheduler
    from .scheduling_karras_ve import KarrasVeScheduler
    from .scheduling_pndm import PNDMScheduler
    from .scheduling_repaint import RePaintScheduler
    from .scheduling_sde_ve import ScoreSdeVeScheduler
    from .scheduling_sde_vp import ScoreSdeVpScheduler
    from .scheduling_utils import SchedulerMixin
    from .scheduling_vq_diffusion import VQDiffusionScheduler
else:
    from ..utils.dummy_pt_objects import *  # noqa F403

if is_flax_available():
    from .scheduling_ddim_flax import FlaxDDIMScheduler
    from .scheduling_ddpm_flax import FlaxDDPMScheduler
    from .scheduling_dpmsolver_multistep_flax import FlaxDPMSolverMultistepScheduler
    from .scheduling_karras_ve_flax import FlaxKarrasVeScheduler
    from .scheduling_lms_discrete_flax import FlaxLMSDiscreteScheduler
    from .scheduling_pndm_flax import FlaxPNDMScheduler
    from .scheduling_sde_ve_flax import FlaxScoreSdeVeScheduler
    from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput, broadcast_to_shape_from_left
else:
    from ..utils.dummy_flax_objects import *  # noqa F403


if is_scipy_available() and is_torch_available():
    from .scheduling_lms_discrete import LMSDiscreteScheduler
else:
    from ..utils.dummy_torch_and_scipy_objects import *  # noqa F403


SCHEDULER_TYPE_TO_CLASS_MAPPING = OrderedDict(
    [
        ("ddim", DDIMScheduler),
        ("ddpm", DDPMScheduler),
        ("dpm_multistep", DPMSolverMultistepScheduler),
        ("euler_ancestral_discrete", EulerAncestralDiscreteScheduler),
        ("euler_discrete", EulerDiscreteScheduler),
        ("ipndm", IPNDMScheduler),
        ("karras_ve", KarrasVeScheduler),
        ("pndm", PNDMScheduler),
        ("repaint", RePaintScheduler),
        ("score_sde_ve", ScoreSdeVeScheduler),
        ("score_sde_vp", ScoreSdeVpScheduler),
        ("vq_diffusion", VQDiffusionScheduler),
        ("lms_discrete", LMSDiscreteScheduler),
    ]
)
CLASS_TO_SCHEDULER_TYPE_MAPPING = OrderedDict({v: k for k, v in SCHEDULER_TYPE_TO_CLASS_MAPPING.items()})

SchedulerType = Enum("SchedulerType", list(SCHEDULER_TYPE_TO_CLASS_MAPPING.keys()))
SchedulerType.__doc__ = (
    """Possible values for the `scheduler_type` argument in [`DiffusionPipeline.set_scheduler`]. Useful for tab-completion in
an IDE. Possible values are"""
    + "\n"
    + "\n- ".join(SCHEDULER_TYPE_TO_CLASS_MAPPING.keys())
)
