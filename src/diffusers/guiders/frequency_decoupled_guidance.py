# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ..configuration_utils import register_to_config
from ..utils import is_kornia_available
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


_CAN_USE_KORNIA = is_kornia_available()


if _CAN_USE_KORNIA:
    from kornia.geometry import pyrup as upsample_and_blur_func
    from kornia.geometry.transform import build_laplacian_pyramid as build_laplacian_pyramid_func
else:
    upsample_and_blur_func = None
    build_laplacian_pyramid_func = None


def project(v0: torch.Tensor, v1: torch.Tensor, upcast_to_double: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project vector v0 onto vector v1, returning the parallel and orthogonal components of v0. Implementation from paper
    (Algorithm 2).
    """
    # v0 shape: [B, ...]
    # v1 shape: [B, ...]
    # Assume first dim is a batch dim and all other dims are channel or "spatial" dims
    all_dims_but_first = list(range(1, len(v0.shape)))
    if upcast_to_double:
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=all_dims_but_first)
    v0_parallel = (v0 * v1).sum(dim=all_dims_but_first, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    if upcast_to_double:
        v0_parallel = v0_parallel.to(dtype)
        v0_orthogonal = v0_orthogonal.to(dtype)
    return v0_parallel, v0_orthogonal


def build_image_from_pyramid(pyramid: List[torch.Tensor]) -> torch.Tensor:
    """
    Recovers the data space latents from the Laplacian pyramid frequency space. Implementation from the paper
    (Algorithm 2).
    """
    # pyramid shapes: [[B, C, H, W], [B, C, H/2, W/2], ...]
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = upsample_and_blur_func(img) + pyramid[i]
    return img


class FrequencyDecoupledGuidance(BaseGuidance):
    """
    Frequency-Decoupled Guidance (FDG): https://huggingface.co/papers/2506.19713

    FDG is a technique similar to (and based on) classifier-free guidance (CFG) which is used to improve generation
    quality and condition-following in diffusion models. Like CFG, during training we jointly train the model on both
    conditional and unconditional data, and use a combination of the two during inference. (If you want more details on
    how CFG works, you can check out the CFG guider.)

    FDG differs from CFG in that the normal CFG prediction is instead decoupled into low- and high-frequency components
    using a frequency transform (such as a Laplacian pyramid). The CFG update is then performed in frequency space
    separately for the low- and high-frequency components with different guidance scales. Finally, the inverse
    frequency transform is used to map the CFG frequency predictions back to data space (e.g. pixel space for images)
    to form the final FDG prediction.

    For images, the FDG authors found that using low guidance scales for the low-frequency components retains sample
    diversity and realistic color composition, while using high guidance scales for high-frequency components enhances
    sample quality (such as better visual details). Therefore, they recommend using low guidance scales (low w_low) for
    the low-frequency components and high guidance scales (high w_high) for the high-frequency components. As an
    example, they suggest w_low = 5.0 and w_high = 10.0 for Stable Diffusion XL (see Table 8 in the paper).

    As with CFG, Diffusers implements the scaling and shifting on the unconditional prediction based on the [Imagen
    paper](https://huggingface.co/papers/2205.11487), which is equivalent to what the original CFG paper proposed in
    theory. [x_pred = x_uncond + scale * (x_cond - x_uncond)]

    The `use_original_formulation` argument can be set to `True` to use the original CFG formulation mentioned in the
    paper. By default, we use the diffusers-native implementation that has been in the codebase for a long time.

    Args:
        guidance_scales (`List[float]`, defaults to `[10.0, 5.0]`):
            The scale parameter for frequency-decoupled guidance for each frequency component, listed from highest
            frequency level to lowest. Higher values result in stronger conditioning on the text prompt, while lower
            values allow for more freedom in generation. Higher values may lead to saturation and deterioration of
            image quality. The FDG authors recommend using higher guidance scales for higher frequency components and
            lower guidance scales for lower frequency components (so `guidance_scales` should typically be sorted in
            descending order).
        guidance_rescale (`float` or `List[float]`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891). If a list is supplied, it should be the same length as
            `guidance_scales`.
        parallel_weights (`float` or `List[float]`, *optional*):
            Optional weights for the parallel component of each frequency component of the projected CFG shift. If not
            set, the weights will default to `1.0` for all components, which corresponds to using the normal CFG shift
            (that is, equal weights for the parallel and orthogonal components). If set, a value in `[0, 1]` is
            recommended. If a list is supplied, it should be the same length as `guidance_scales`.
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float` or `List[float]`, defaults to `0.0`):
            The fraction of the total number of denoising steps after which guidance starts. If a list is supplied, it
            should be the same length as `guidance_scales`.
        stop (`float` or `List[float]`, defaults to `1.0`):
            The fraction of the total number of denoising steps after which guidance stops. If a list is supplied, it
            should be the same length as `guidance_scales`.
        guidance_rescale_space (`str`, defaults to `"data"`):
            Whether to performance guidance rescaling in `"data"` space (after the full FDG update in data space) or in
            `"freq"` space (right after the CFG update, for each freq level). Note that frequency space rescaling is
            speculative and may not produce expected results. If `"data"` is set, the first `guidance_rescale` value
            will be used; otherwise, per-frequency-level guidance rescale values will be used if available.
        upcast_to_double (`bool`, defaults to `True`):
            Whether to upcast certain operations, such as the projection operation when using `parallel_weights`, to
            float64 when performing guidance. This may result in better performance at the cost of increased runtime.
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scales: Union[List[float], Tuple[float]] = [10.0, 5.0],
        guidance_rescale: Union[float, List[float], Tuple[float]] = 0.0,
        parallel_weights: Optional[Union[float, List[float], Tuple[float]]] = None,
        use_original_formulation: bool = False,
        start: Union[float, List[float], Tuple[float]] = 0.0,
        stop: Union[float, List[float], Tuple[float]] = 1.0,
        guidance_rescale_space: str = "data",
        upcast_to_double: bool = True,
        enabled: bool = True,
    ):
        if not _CAN_USE_KORNIA:
            raise ImportError(
                "The `FrequencyDecoupledGuidance` guider cannot be instantiated because the `kornia` library on which "
                "it depends is not available in the current environment. You can install `kornia` with `pip install "
                "kornia`."
            )

        # Set start to earliest start for any freq component and stop to latest stop for any freq component
        min_start = start if isinstance(start, float) else min(start)
        max_stop = stop if isinstance(stop, float) else max(stop)
        super().__init__(min_start, max_stop, enabled)

        self.guidance_scales = guidance_scales
        self.levels = len(guidance_scales)

        if isinstance(guidance_rescale, float):
            self.guidance_rescale = [guidance_rescale] * self.levels
        elif len(guidance_rescale) == self.levels:
            self.guidance_rescale = guidance_rescale
        else:
            raise ValueError(
                f"`guidance_rescale` has length {len(guidance_rescale)} but should have the same length as "
                f"`guidance_scales` ({len(self.guidance_scales)})"
            )
        # Whether to perform guidance rescaling in frequency space (right after the CFG update) or data space (after
        # transforming from frequency space back to data space)
        if guidance_rescale_space not in ["data", "freq"]:
            raise ValueError(
                f"Guidance rescale space is {guidance_rescale_space} but must be one of `data` or `freq`."
            )
        self.guidance_rescale_space = guidance_rescale_space

        if parallel_weights is None:
            # Use normal CFG shift (equal weights for parallel and orthogonal components)
            self.parallel_weights = [1.0] * self.levels
        elif isinstance(parallel_weights, float):
            self.parallel_weights = [parallel_weights] * self.levels
        elif len(parallel_weights) == self.levels:
            self.parallel_weights = parallel_weights
        else:
            raise ValueError(
                f"`parallel_weights` has length {len(parallel_weights)} but should have the same length as "
                f"`guidance_scales` ({len(self.guidance_scales)})"
            )

        self.use_original_formulation = use_original_formulation
        self.upcast_to_double = upcast_to_double

        if isinstance(start, float):
            self.guidance_start = [start] * self.levels
        elif len(start) == self.levels:
            self.guidance_start = start
        else:
            raise ValueError(
                f"`start` has length {len(start)} but should have the same length as `guidance_scales` "
                f"({len(self.guidance_scales)})"
            )
        if isinstance(stop, float):
            self.guidance_stop = [stop] * self.levels
        elif len(stop) == self.levels:
            self.guidance_stop = stop
        else:
            raise ValueError(
                f"`stop` has length {len(stop)} but should have the same length as `guidance_scales` "
                f"({len(self.guidance_scales)})"
            )

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> GuiderOutput:
        pred = None

        if not self._is_fdg_enabled():
            pred = pred_cond
        else:
            # Apply the frequency transform (e.g. Laplacian pyramid) to the conditional and unconditional predictions.
            pred_cond_pyramid = build_laplacian_pyramid_func(pred_cond, self.levels)
            pred_uncond_pyramid = build_laplacian_pyramid_func(pred_uncond, self.levels)

            # From high frequencies to low frequencies, following the paper implementation
            pred_guided_pyramid = []
            parameters = zip(self.guidance_scales, self.parallel_weights, self.guidance_rescale)
            for level, (guidance_scale, parallel_weight, guidance_rescale) in enumerate(parameters):
                if self._is_fdg_enabled_for_level(level):
                    # Get the cond/uncond preds (in freq space) at the current frequency level
                    pred_cond_freq = pred_cond_pyramid[level]
                    pred_uncond_freq = pred_uncond_pyramid[level]

                    shift = pred_cond_freq - pred_uncond_freq

                    # Apply parallel weights, if used (1.0 corresponds to using the normal CFG shift)
                    if not math.isclose(parallel_weight, 1.0):
                        shift_parallel, shift_orthogonal = project(shift, pred_cond_freq, self.upcast_to_double)
                        shift = parallel_weight * shift_parallel + shift_orthogonal

                    # Apply CFG update for the current frequency level
                    pred = pred_cond_freq if self.use_original_formulation else pred_uncond_freq
                    pred = pred + guidance_scale * shift

                    if self.guidance_rescale_space == "freq" and guidance_rescale > 0.0:
                        pred = rescale_noise_cfg(pred, pred_cond_freq, guidance_rescale)

                    # Add the current FDG guided level to the FDG prediction pyramid
                    pred_guided_pyramid.append(pred)
                else:
                    # Add the current pred_cond_pyramid level as the "non-FDG" prediction
                    pred_guided_pyramid.append(pred_cond_freq)

            # Convert from frequency space back to data (e.g. pixel) space by applying inverse freq transform
            pred = build_image_from_pyramid(pred_guided_pyramid)

            # If rescaling in data space, use the first elem of self.guidance_rescale as the "global" rescale value
            # across all freq levels
            if self.guidance_rescale_space == "data" and self.guidance_rescale[0] > 0.0:
                pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale[0])

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_fdg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_fdg_enabled(self) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step

        is_close = False
        if self.use_original_formulation:
            is_close = all(math.isclose(guidance_scale, 0.0) for guidance_scale in self.guidance_scales)
        else:
            is_close = all(math.isclose(guidance_scale, 1.0) for guidance_scale in self.guidance_scales)

        return is_within_range and not is_close

    def _is_fdg_enabled_for_level(self, level: int) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self.guidance_start[level] * self._num_inference_steps)
            skip_stop_step = int(self.guidance_stop[level] * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step

        is_close = False
        if self.use_original_formulation:
            is_close = math.isclose(self.guidance_scales[level], 0.0)
        else:
            is_close = math.isclose(self.guidance_scales[level], 1.0)

        return is_within_range and not is_close
