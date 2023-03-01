# Copyright 2023 The HuggingFace Team. All rights reserved.
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


from ..utils import OptionalDependencyNotAvailable, is_flax_available, is_scipy_available, is_torch_available


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_pt_objects import *  # noqa F403
else:
    from .scheduling_ddim import DDIMScheduler
    from .scheduling_ddim_inverse import DDIMInverseScheduler
    from .scheduling_ddpm import DDPMScheduler
    from .scheduling_deis_multistep import DEISMultistepScheduler
    from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from .scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
    from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
    from .scheduling_euler_discrete import EulerDiscreteScheduler
    from .scheduling_heun_discrete import HeunDiscreteScheduler
    from .scheduling_ipndm import IPNDMScheduler
    from .scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
    from .scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
    from .scheduling_karras_ve import KarrasVeScheduler
    from .scheduling_pndm import PNDMScheduler
    from .scheduling_repaint import RePaintScheduler
    from .scheduling_sde_ve import ScoreSdeVeScheduler
    from .scheduling_sde_vp import ScoreSdeVpScheduler
    from .scheduling_unclip import UnCLIPScheduler
    from .scheduling_unipc_multistep import UniPCMultistepScheduler
    from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
    from .scheduling_vq_diffusion import VQDiffusionScheduler

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_objects import *  # noqa F403
else:
    from .scheduling_ddim_flax import FlaxDDIMScheduler
    from .scheduling_ddpm_flax import FlaxDDPMScheduler
    from .scheduling_dpmsolver_multistep_flax import FlaxDPMSolverMultistepScheduler
    from .scheduling_karras_ve_flax import FlaxKarrasVeScheduler
    from .scheduling_lms_discrete_flax import FlaxLMSDiscreteScheduler
    from .scheduling_pndm_flax import FlaxPNDMScheduler
    from .scheduling_sde_ve_flax import FlaxScoreSdeVeScheduler
    from .scheduling_utils_flax import (
        FlaxKarrasDiffusionSchedulers,
        FlaxSchedulerMixin,
        FlaxSchedulerOutput,
        broadcast_to_shape_from_left,
    )


try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_scipy_objects import *  # noqa F403
else:
    from .scheduling_lms_discrete import LMSDiscreteScheduler
