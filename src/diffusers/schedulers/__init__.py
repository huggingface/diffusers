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

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_flax_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
)


_dummy_modules = {}
_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_pt_objects  # noqa F403

    _dummy_modules.update(get_objects_from_module(dummy_pt_objects))

else:
    _import_structure["scheduling_consistency_models"] = ["CMStochasticIterativeScheduler"]
    _import_structure["scheduling_ddim"] = ["DDIMScheduler"]
    _import_structure["scheduling_ddim_inverse"] = ["DDIMInverseScheduler"]
    _import_structure["scheduling_ddim_parallel"] = ["DDIMParallelScheduler"]
    _import_structure["scheduling_ddpm"] = ["DDPMScheduler"]
    _import_structure["scheduling_ddpm_parallel"] = ["DDPMParallelScheduler"]
    _import_structure["scheduling_ddpm_wuerstchen"] = ["DDPMWuerstchenScheduler"]
    _import_structure["scheduling_deis_multistep"] = ["DEISMultistepScheduler"]
    _import_structure["scheduling_dpmsolver_multistep"] = ["DPMSolverMultistepScheduler"]
    _import_structure["scheduling_dpmsolver_multistep_inverse"] = ["DPMSolverMultistepInverseScheduler"]
    _import_structure["scheduling_dpmsolver_singlestep"] = ["DPMSolverSinglestepScheduler"]
    _import_structure["scheduling_euler_ancestral_discrete"] = ["EulerAncestralDiscreteScheduler"]
    _import_structure["scheduling_euler_discrete"] = ["EulerDiscreteScheduler"]
    _import_structure["scheduling_heun_discrete"] = ["HeunDiscreteScheduler"]
    _import_structure["scheduling_ipndm"] = ["IPNDMScheduler"]
    _import_structure["scheduling_k_dpm_2_ancestral_discrete"] = ["KDPM2AncestralDiscreteScheduler"]
    _import_structure["scheduling_k_dpm_2_discrete"] = ["KDPM2DiscreteScheduler"]
    _import_structure["scheduling_karras_ve"] = ["KarrasVeScheduler"]
    _import_structure["scheduling_lcm"] = ["LCMScheduler"]
    _import_structure["scheduling_pndm"] = ["PNDMScheduler"]
    _import_structure["scheduling_repaint"] = ["RePaintScheduler"]
    _import_structure["scheduling_sde_ve"] = ["ScoreSdeVeScheduler"]
    _import_structure["scheduling_sde_vp"] = ["ScoreSdeVpScheduler"]
    _import_structure["scheduling_unclip"] = ["UnCLIPScheduler"]
    _import_structure["scheduling_unipc_multistep"] = ["UniPCMultistepScheduler"]
    _import_structure["scheduling_utils"] = ["KarrasDiffusionSchedulers", "SchedulerMixin"]
    _import_structure["scheduling_vq_diffusion"] = ["VQDiffusionScheduler"]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_flax_objects  # noqa F403

    _dummy_modules.update(get_objects_from_module(dummy_flax_objects))

else:
    _import_structure["scheduling_ddim_flax"] = ["FlaxDDIMScheduler"]
    _import_structure["scheduling_ddpm_flax"] = ["FlaxDDPMScheduler"]
    _import_structure["scheduling_dpmsolver_multistep_flax"] = ["FlaxDPMSolverMultistepScheduler"]
    _import_structure["scheduling_euler_discrete_flax"] = ["FlaxEulerDiscreteScheduler"]
    _import_structure["scheduling_karras_ve_flax"] = ["FlaxKarrasVeScheduler"]
    _import_structure["scheduling_lms_discrete_flax"] = ["FlaxLMSDiscreteScheduler"]
    _import_structure["scheduling_pndm_flax"] = ["FlaxPNDMScheduler"]
    _import_structure["scheduling_sde_ve_flax"] = ["FlaxScoreSdeVeScheduler"]
    _import_structure["scheduling_utils_flax"] = [
        "FlaxKarrasDiffusionSchedulers",
        "FlaxSchedulerMixin",
        "FlaxSchedulerOutput",
        "broadcast_to_shape_from_left",
    ]


try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_scipy_objects  # noqa F403

    _dummy_modules.update(get_objects_from_module(dummy_torch_and_scipy_objects))

else:
    _import_structure["scheduling_lms_discrete"] = ["LMSDiscreteScheduler"]

try:
    if not (is_torch_available() and is_torchsde_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_torch_and_torchsde_objects  # noqa F403

    _dummy_modules.update(get_objects_from_module(dummy_torch_and_torchsde_objects))

else:
    _import_structure["scheduling_dpmsolver_sde"] = ["DPMSolverSDEScheduler"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from ..utils import (
        OptionalDependencyNotAvailable,
        is_flax_available,
        is_scipy_available,
        is_torch_available,
        is_torchsde_available,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_pt_objects import *  # noqa F403
    else:
        from .scheduling_consistency_models import CMStochasticIterativeScheduler
        from .scheduling_ddim import DDIMScheduler
        from .scheduling_ddim_inverse import DDIMInverseScheduler
        from .scheduling_ddim_parallel import DDIMParallelScheduler
        from .scheduling_ddpm import DDPMScheduler
        from .scheduling_ddpm_parallel import DDPMParallelScheduler
        from .scheduling_ddpm_wuerstchen import DDPMWuerstchenScheduler
        from .scheduling_deis_multistep import DEISMultistepScheduler
        from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        from .scheduling_dpmsolver_multistep_inverse import DPMSolverMultistepInverseScheduler
        from .scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
        from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        from .scheduling_euler_discrete import EulerDiscreteScheduler
        from .scheduling_heun_discrete import HeunDiscreteScheduler
        from .scheduling_ipndm import IPNDMScheduler
        from .scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
        from .scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
        from .scheduling_karras_ve import KarrasVeScheduler
        from .scheduling_lcm import LCMScheduler
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
        from .scheduling_euler_discrete_flax import FlaxEulerDiscreteScheduler
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

    try:
        if not (is_torch_available() and is_torchsde_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_torch_and_torchsde_objects import *  # noqa F403
    else:
        from .scheduling_dpmsolver_sde import DPMSolverSDEScheduler

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    for name, value in _dummy_modules.items():
        setattr(sys.modules[__name__], name, value)
