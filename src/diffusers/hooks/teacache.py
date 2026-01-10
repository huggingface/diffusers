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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ..models.modeling_outputs import Transformer2DModelOutput
from ..utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_TEACACHE_HOOK = "teacache"


def _handle_accelerate_hook(module: torch.nn.Module, *args, **kwargs) -> Tuple[tuple, dict]:
    """Handle compatibility with accelerate's CPU offload hooks.

    When TeaCache's new_forward replaces the forward chain, accelerate's hooks are bypassed.
    This function manually triggers accelerate's pre_forward to ensure proper device placement.

    Args:
        module: The model module that may have accelerate hooks attached.
        *args: Forward arguments to potentially move to the execution device.
        **kwargs: Forward keyword arguments to potentially move to the execution device.

    Returns:
        Tuple of (args, kwargs) potentially moved to the correct device.
    """
    if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "pre_forward"):
        # Accelerate's CpuOffload hook will move the module to GPU and return modified args/kwargs
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
    return args, kwargs


def _get_model_config():
    """Get model configuration mapping.

    Returns dict at runtime when forward functions are defined. Order matters: more specific model variants must come
    before generic ones.
    """
    return {
        "FluxKontext": {
            "forward_func": _flux_teacache_forward,
            "coefficients": [-1.04655119e03, 3.12563399e02, -1.69500694e01, 4.10995971e-01, 3.74537863e-02],
        },
        "Flux": {
            "forward_func": _flux_teacache_forward,
            "coefficients": [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
        },
        "Mochi": {
            "forward_func": _mochi_teacache_forward,
            "coefficients": [-3.51241319e03, 8.11675948e02, -6.09400215e01, 2.42429681e00, 3.05291719e-03],
        },
        "Lumina2": {
            "forward_func": _lumina2_teacache_forward,
            "coefficients": [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344],
        },
        "CogVideoX1.5-5B-I2V": {
            "forward_func": _cogvideox_teacache_forward,
            "coefficients": [1.22842302e02, -1.04088754e02, 2.62981677e01, -3.06001e-01, 3.71213220e-02],
        },
        "CogVideoX1.5-5B": {
            "forward_func": _cogvideox_teacache_forward,
            "coefficients": [2.50210439e02, -1.65061612e02, 3.57804877e01, -7.81551492e-01, 3.58559703e-02],
        },
        "CogVideoX-2b": {
            "forward_func": _cogvideox_teacache_forward,
            "coefficients": [-3.10658903e01, 2.54732368e01, -5.92380459e00, 1.75769064e00, -3.61568434e-03],
        },
        "CogVideoX-5b": {
            "forward_func": _cogvideox_teacache_forward,
            "coefficients": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
        },
        "CogVideoX": {
            "forward_func": _cogvideox_teacache_forward,
            "coefficients": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
        },
    }


def _auto_detect_model_type(module):
    """Auto-detect model type from class name and config path."""
    class_name = module.__class__.__name__
    config_path = getattr(getattr(module, "config", None), "_name_or_path", "").lower()
    model_config = _get_model_config()

    # Check config path first (for variants), then class name (ordered most specific first)
    for model_type in model_config:
        if model_type.lower() in config_path or model_type in class_name:
            return model_type

    raise ValueError(f"TeaCache: Unsupported model '{class_name}'. Supported: {', '.join(model_config.keys())}")


def _rescale_distance(coefficients, x):
    """Polynomial rescaling: c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]"""
    c = coefficients
    return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x + c[4]


@torch.compiler.disable
def _should_compute(state, modulated_inp, coefficients, rel_l1_thresh):
    """Determine if full computation is needed (single residual models)."""
    # First timestep always computes
    if state.cnt == 0:
        state.accumulated_rel_l1_distance = 0
        return True
    # Last timestep always computes
    if state.num_steps > 0 and state.cnt == state.num_steps - 1:
        state.accumulated_rel_l1_distance = 0
        return True
    # No previous state - must compute
    if state.previous_modulated_input is None:
        return True
    if state.previous_residual is None:
        return True

    # Compute L1 distance and check threshold
    # Note: .item() implicitly syncs GPU->CPU. This is necessary for the threshold comparison.
    prev_mean = state.previous_modulated_input.abs().mean()
    if prev_mean.item() > 1e-9:
        rel_distance = ((modulated_inp - state.previous_modulated_input).abs().mean() / prev_mean).item()
    else:
        # Handle near-zero previous input: if current is also near-zero, no change; otherwise force recompute
        rel_distance = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float("inf")
    rescaled = _rescale_distance(coefficients, rel_distance)
    state.accumulated_rel_l1_distance += rescaled

    if state.accumulated_rel_l1_distance < rel_l1_thresh:
        return False

    state.accumulated_rel_l1_distance = 0
    return True


@torch.compiler.disable
def _should_compute_dual(state, modulated_inp, coefficients, rel_l1_thresh):
    """Determine if full computation is needed (dual residual models like CogVideoX)."""
    # Also check encoder residual
    if state.previous_residual is None or state.previous_residual_encoder is None:
        return True
    return _should_compute(state, modulated_inp, coefficients, rel_l1_thresh)


def _update_state(state, output, original_input, modulated_inp):
    """Update cache state after full computation (single residual)."""
    state.previous_residual = output - original_input
    state.previous_modulated_input = modulated_inp
    state.cnt += 1


def _update_state_dual(state, hs_output, enc_output, hs_original, enc_original, modulated_inp):
    """Update cache state after full computation (dual residual)."""
    state.previous_residual = hs_output - hs_original
    state.previous_residual_encoder = enc_output - enc_original
    state.previous_modulated_input = modulated_inp
    state.cnt += 1


def _apply_cached_residual(state, input_tensor, modulated_inp):
    """Apply cached residual - fast path (single residual)."""
    output = input_tensor + state.previous_residual
    state.previous_modulated_input = modulated_inp
    state.cnt += 1
    return output


def _apply_cached_residual_dual(state, hs, enc, modulated_inp):
    """Apply cached residuals - fast path (dual residual)."""
    hs_out = hs + state.previous_residual
    enc_out = enc + state.previous_residual_encoder
    state.previous_modulated_input = modulated_inp
    state.cnt += 1
    return hs_out, enc_out


@dataclass
class TeaCacheConfig:
    r"""
    Configuration for [TeaCache](https://arxiv.org/abs/2411.19108) applied to transformer models.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique that speeds up diffusion model inference
    by reusing transformer block computations when consecutive timestep embeddings are similar. It uses polynomial
    rescaling of L1 distances between modulated inputs to intelligently decide when to cache.

    Reference: [TeaCache: Timestep Embedding Aware Cache for Efficient Diffusion Model Inference](https://arxiv.org/abs/2411.19108)

    Currently supports: FLUX, FLUX-Kontext, Mochi, Lumina2, and CogVideoX models. Model type is auto-detected, and
    model-specific polynomial coefficients are automatically applied.

    Args:
        rel_l1_thresh (`float`, defaults to `0.2`):
            Threshold for accumulated relative L1 distance. When the accumulated distance is below this threshold, the
            cached residual from the previous timestep is reused instead of computing the full transformer. Based on
            the original TeaCache paper, values in the range [0.1, 0.3] work best for balancing speed and quality:
            - 0.25 for ~1.5x speedup with minimal quality loss
            - 0.4 for ~1.8x speedup with slight quality loss
            - 0.6 for ~2.0x speedup with noticeable quality loss
            - 0.8 for ~2.25x speedup with significant quality loss
            Higher thresholds lead to more aggressive caching and faster inference, but may reduce output quality.
            Note: Mochi models require lower thresholds (0.06-0.09) due to different coefficient scaling.
        coefficients (`List[float]`, *optional*, defaults to polynomial coefficients from TeaCache paper):
            Polynomial coefficients used for rescaling the raw L1 distance. These coefficients transform the relative
            L1 distance into a model-specific caching signal. If not provided, defaults to the coefficients determined
            for FLUX models in the TeaCache paper: [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00,
            2.64230861e-01]. The polynomial is evaluated as: `c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]` where x
            is the relative L1 distance.
        current_timestep_callback (`Callable[[], int]`, *optional*, defaults to `None`):
            Callback function that returns the current timestep during inference. This is used internally for debugging
            and statistics tracking. If not provided, TeaCache will still function correctly.
        num_inference_steps (`int`, *optional*, defaults to `None`):
            Total number of inference steps. Required for proper state management - ensures first and last timesteps
            are always computed (never cached) and that state resets between inference runs. If not provided, TeaCache
            will attempt to detect via callback or module attribute.
        num_inference_steps_callback (`Callable[[], int]`, *optional*, defaults to `None`):
            Callback function that returns the total number of inference steps. Alternative to `num_inference_steps`
            for dynamic step counts.

    Example:
        ```python
        from diffusers import FluxPipeline
        from diffusers.hooks import TeaCacheConfig

        # Load FLUX pipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        # Enable TeaCache with auto-detection (1.5x speedup)
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        pipe.transformer.enable_cache(config)

        # Generate image with caching
        image = pipe("A cat sitting on a windowsill", num_inference_steps=4).images[0]

        # Disable caching
        pipe.transformer.disable_cache()
        ```
    """

    rel_l1_thresh: float = 0.2
    coefficients: Optional[List[float]] = None
    current_timestep_callback: Optional[Callable[[], int]] = None
    num_inference_steps: Optional[int] = None
    num_inference_steps_callback: Optional[Callable[[], int]] = None

    def __post_init__(self):
        # Validate rel_l1_thresh
        if not isinstance(self.rel_l1_thresh, (int, float)):
            raise TypeError(
                f"rel_l1_thresh must be a number, got {type(self.rel_l1_thresh).__name__}. "
                f"Please provide a float value between 0.1 and 1.0."
            )
        if self.rel_l1_thresh <= 0:
            raise ValueError(
                f"rel_l1_thresh must be positive, got {self.rel_l1_thresh}. "
                f"Based on the TeaCache paper, values between 0.1 and 0.3 work best. "
                f"Try 0.25 for 1.5x speedup or 0.6 for 2x speedup."
            )
        if self.rel_l1_thresh < 0.05:
            logger.warning(
                f"rel_l1_thresh={self.rel_l1_thresh} is very low and may result in minimal caching. "
                f"Consider using values between 0.1 and 0.3 for optimal performance."
            )
        if self.rel_l1_thresh > 1.0:
            logger.warning(
                f"rel_l1_thresh={self.rel_l1_thresh} is very high and may cause quality degradation. "
                f"Consider using values between 0.1 and 0.6 for better quality-speed tradeoff."
            )

        # Validate coefficients only if explicitly provided (None = auto-detect later)
        if self.coefficients is not None:
            if not isinstance(self.coefficients, (list, tuple)):
                raise TypeError(
                    f"coefficients must be a list or tuple, got {type(self.coefficients).__name__}. "
                    f"Please provide a list of 5 polynomial coefficients."
                )
            if len(self.coefficients) != 5:
                raise ValueError(
                    f"coefficients must contain exactly 5 elements for 4th-degree polynomial, "
                    f"got {len(self.coefficients)}. The polynomial is evaluated as: "
                    f"c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]"
                )
            if not all(isinstance(c, (int, float)) for c in self.coefficients):
                raise TypeError(
                    f"All coefficients must be numbers. Got types: {[type(c).__name__ for c in self.coefficients]}"
                )

    def __repr__(self) -> str:
        return (
            f"TeaCacheConfig(\n"
            f"  rel_l1_thresh={self.rel_l1_thresh},\n"
            f"  coefficients={self.coefficients},\n"
            f"  current_timestep_callback={self.current_timestep_callback},\n"
            f"  num_inference_steps={self.num_inference_steps},\n"
            f"  num_inference_steps_callback={self.num_inference_steps_callback}\n"
            f")"
        )


class TeaCacheState(BaseState):
    """
    State management for TeaCache hook.

    This class tracks the caching state across diffusion timesteps, managing counters, accumulated distances, and
    cached values needed for the TeaCache algorithm. The state persists across multiple forward passes during a single
    inference run and is automatically reset when a new inference begins.

    Attributes:
        cnt (int):
            Current timestep counter, incremented with each forward pass. Used to identify first/last timesteps which
            are always computed (never cached) for maximum quality.
        num_steps (int):
            Total number of inference steps for the current run. Used to identify the last timestep. Automatically
            detected from callbacks or pipeline attributes if not explicitly set.
        accumulated_rel_l1_distance (float):
            Running accumulator for rescaled L1 distances between consecutive modulated inputs. Compared against the
            threshold to make caching decisions. Reset to 0 when the decision is made to recompute.
        previous_modulated_input (torch.Tensor):
            Modulated input from the previous timestep, extracted from the first transformer block's norm1 layer. Used
            for computing L1 distance to determine similarity between consecutive timesteps.
        previous_residual (torch.Tensor):
            Cached residual (output - input) from the previous timestep's full transformer computation. Applied
            directly when caching is triggered instead of computing all transformer blocks.
        previous_residual_encoder (torch.Tensor, optional):
            Cached encoder residual for models that cache both encoder and hidden_states residuals (e.g., CogVideoX).
            None for models that only cache hidden_states residual.
    """

    def __init__(self):
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        # CogVideoX-specific: dual residual caching (encoder + hidden_states)
        self.previous_residual_encoder = None  # Only used by CogVideoX
        # Lumina2-specific: per-sequence-length caching for variable sequence lengths
        # Other models don't use these fields but they're allocated for simplicity
        self.cache_dict = {}  # Only used by Lumina2
        self.uncond_seq_len = None  # Only used by Lumina2

    def reset(self):
        """Reset all state variables to initial values for a new inference run."""
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_encoder = None
        self.cache_dict = {}
        self.uncond_seq_len = None

    def __repr__(self) -> str:
        return (
            f"TeaCacheState(\n"
            f"  cnt={self.cnt},\n"
            f"  num_steps={self.num_steps},\n"
            f"  accumulated_rel_l1_distance={self.accumulated_rel_l1_distance:.6f},\n"
            f"  previous_modulated_input={'cached' if self.previous_modulated_input is not None else 'None'},\n"
            f"  previous_residual={'cached' if self.previous_residual is not None else 'None'}\n"
            f")"
        )


class TeaCacheHook(ModelHook):
    """
    ModelHook implementing TeaCache for transformer models.

    This hook intercepts transformer forward pass and implements adaptive caching based on timestep embedding
    similarity. It extracts modulated inputs, computes L1 distances, applies polynomial rescaling, and decides whether
    to reuse cached residuals or compute full transformer blocks.

    The hook follows the original TeaCache algorithm from the paper:
    1. Extract modulated input using provided extractor function
    2. Compute relative L1 distance between current and previous modulated inputs
    3. Apply polynomial rescaling with model-specific coefficients to the distance
    4. Accumulate rescaled distances and compare to threshold
    5. If below threshold: reuse cached residual (fast path, skip transformer computation)
    6. If above threshold: compute full transformer blocks and cache new residual (slow path)

    The first and last timesteps are always computed fully (never cached) to ensure maximum quality.

    Attributes:
        config (TeaCacheConfig):
            Configuration containing threshold, polynomial coefficients, and optional callbacks.
        coefficients (List[float]):
            Polynomial coefficients for rescaling L1 distances (auto-detected or user-provided).
        state_manager (StateManager):
            Manages TeaCacheState across forward passes, maintaining counters and cached values.
    """

    _is_stateful = True

    def __init__(self, config: TeaCacheConfig):
        super().__init__()
        self.config = config
        # Coefficients will be set in initialize_hook() via auto-detection or user config
        self.coefficients: Optional[List[float]] = config.coefficients
        self.state_manager = StateManager(TeaCacheState, (), {})
        self.model_type: Optional[str] = None
        self._forward_func: Optional[Callable] = None

    def _maybe_reset_state_for_new_inference(
        self, state: TeaCacheState, module: torch.nn.Module, reset_encoder_residual: bool = False
    ) -> None:
        """Reset state if we've completed all steps (start of new inference run).

        Also initializes num_steps on first timestep if not set.

        Args:
            state: TeaCacheState instance.
            module: The transformer module.
            reset_encoder_residual: If True, also reset previous_residual_encoder (for CogVideoX).
        """
        # Reset counter if we've completed all steps (new inference run)
        if state.cnt == state.num_steps and state.num_steps > 0:
            logger.debug("TeaCache: Inference run completed, resetting state")
            state.cnt = 0
            state.accumulated_rel_l1_distance = 0.0
            state.previous_modulated_input = None
            state.previous_residual = None
            if reset_encoder_residual:
                state.previous_residual_encoder = None

        # Set num_steps on first timestep if not already set
        if state.cnt == 0 and state.num_steps == 0:
            # Priority: config value > callback > module attribute
            if self.config.num_inference_steps is not None:
                state.num_steps = self.config.num_inference_steps
            elif self.config.num_inference_steps_callback is not None:
                state.num_steps = self.config.num_inference_steps_callback()
            elif hasattr(module, "num_steps"):
                state.num_steps = module.num_steps

            if state.num_steps > 0:
                logger.debug(f"TeaCache: Using {state.num_steps} inference steps")

    def initialize_hook(self, module):
        # TODO: DN6 raised concern about context setting timing.
        # Currently set in initialize_hook(). Should this be in denoising loop instead?
        # See PR #12652 for discussion. Keeping current behavior pending clarification.
        self.state_manager.set_context("teacache")

        model_config = _get_model_config()

        # Auto-detect model type and get forward function
        self.model_type = _auto_detect_model_type(module)
        self._forward_func = model_config[self.model_type]["forward_func"]

        # Validate model has required attributes for TeaCache
        if self.model_type in ("Flux", "FluxKontext", "Mochi"):
            if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
                raise ValueError(f"TeaCache: {self.model_type} model missing transformer_blocks")
            if not hasattr(module.transformer_blocks[0], "norm1"):
                raise ValueError(f"TeaCache: {self.model_type} transformer_blocks[0] missing norm1")
        elif self.model_type == "Lumina2":
            if not hasattr(module, "layers") or len(module.layers) == 0:
                raise ValueError(f"TeaCache: Lumina2 model missing layers")
        elif "CogVideoX" in self.model_type:
            if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
                raise ValueError(f"TeaCache: {self.model_type} model missing transformer_blocks")

        # Auto-detect coefficients if not provided by user
        if self.config.coefficients is None:
            self.coefficients = model_config[self.model_type]["coefficients"]
            logger.debug(f"TeaCache: Using {self.model_type} coefficients")
        else:
            self.coefficients = self.config.coefficients
            logger.debug("TeaCache: Using user-provided coefficients")

        return module

    def new_forward(self, module, *args, **kwargs):
        return self._forward_func(self, module, *args, **kwargs)

    def reset_state(self, module):
        self.state_manager.reset()
        return module


def _flux_teacache_forward(
    hook: "TeaCacheHook",
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    return_dict: bool = True,
    **kwargs,
):
    """TeaCache forward for Flux models."""
    # Handle accelerate CPU offload compatibility - moves module and inputs to GPU if needed
    args, extra_kwargs = _handle_accelerate_hook(
        module,
        hidden_states,
        timestep,
        pooled_projections,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        return_dict=return_dict,
        **kwargs,
    )
    hidden_states, timestep, pooled_projections, encoder_hidden_states, txt_ids, img_ids = args
    return_dict = extra_kwargs.pop("return_dict", return_dict)
    kwargs = extra_kwargs

    state = hook.state_manager.get_state()
    hook._maybe_reset_state_for_new_inference(state, module)

    hidden_states = module.x_embedder(hidden_states)

    timestep_scaled = timestep.to(hidden_states.dtype) * 1000
    if kwargs.get("guidance") is not None:
        guidance = kwargs["guidance"].to(hidden_states.dtype) * 1000
        temb = module.time_text_embed(timestep_scaled, guidance, pooled_projections)
    else:
        temb = module.time_text_embed(timestep_scaled, pooled_projections)

    # Inline extractor: Flux uses transformer_blocks[0].norm1
    modulated_inp = module.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]

    # Caching decision and execution
    if _should_compute(state, modulated_inp, hook.coefficients, hook.config.rel_l1_thresh):
        # Full computation path
        ori_hs = hidden_states.clone()
        enc = module.context_embedder(encoder_hidden_states)

        txt = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
        img = img_ids[0] if img_ids.ndim == 3 else img_ids
        ids = torch.cat((txt, img), dim=0)
        image_rotary_emb = module.pos_embed(ids)

        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        for block in module.transformer_blocks:
            enc, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        for block in module.single_transformer_blocks:
            enc, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        _update_state(state, hidden_states, ori_hs, modulated_inp)
    else:
        # Cached path
        hidden_states = _apply_cached_residual(state, hidden_states, modulated_inp)

    hidden_states = module.norm_out(hidden_states, temb)
    output = module.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _mochi_teacache_forward(
    hook: "TeaCacheHook",
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    """TeaCache forward for Mochi models."""
    # Handle accelerate CPU offload compatibility - moves module and inputs to GPU if needed
    args, kwargs = _handle_accelerate_hook(
        module,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        attention_kwargs=attention_kwargs,
        return_dict=return_dict,
    )
    hidden_states, encoder_hidden_states, timestep, encoder_attention_mask = args
    attention_kwargs = kwargs.get("attention_kwargs", attention_kwargs)
    return_dict = kwargs.get("return_dict", return_dict)

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(module, lora_scale)

    state = hook.state_manager.get_state()
    hook._maybe_reset_state_for_new_inference(state, module)

    batch_size, _, num_frames, height, width = hidden_states.shape
    p = module.config.patch_size
    post_patch_height = height // p
    post_patch_width = width // p

    temb, encoder_hidden_states = module.time_embed(
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        hidden_dtype=hidden_states.dtype,
    )

    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
    hidden_states = module.patch_embed(hidden_states)
    hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

    image_rotary_emb = module.rope(
        module.pos_frequencies,
        num_frames,
        post_patch_height,
        post_patch_width,
        device=hidden_states.device,
        dtype=torch.float32,
    )

    # Inline extractor: Mochi norm1 returns tuple (modulated_inp, gate_msa, scale_mlp, gate_mlp)
    modulated_inp = module.transformer_blocks[0].norm1(hidden_states, temb)[0]

    # Caching decision and execution
    if _should_compute(state, modulated_inp, hook.coefficients, hook.config.rel_l1_thresh):
        # Full computation path
        ori_hs = hidden_states.clone()
        enc = encoder_hidden_states
        for block in module.transformer_blocks:
            if torch.is_grad_enabled() and module.gradient_checkpointing:
                hidden_states, enc = module._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    enc,
                    temb,
                    encoder_attention_mask,
                    image_rotary_emb,
                )
            else:
                hidden_states, enc = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
        # norm_out is included in residual (matches original TeaCache implementation)
        hidden_states = module.norm_out(hidden_states, temb)
        _update_state(state, hidden_states, ori_hs, modulated_inp)
    else:
        # Cached path - residual already includes norm_out effect
        hidden_states = _apply_cached_residual(state, hidden_states, modulated_inp)

    hidden_states = module.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
    hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
    output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(module, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _lumina2_teacache_forward(
    hook: "TeaCacheHook",
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    """TeaCache forward for Lumina2 models (handles variable seq lens + per-len caches)."""
    # Handle accelerate CPU offload compatibility - moves module and inputs to GPU if needed
    args, kwargs = _handle_accelerate_hook(
        module,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        attention_kwargs=attention_kwargs,
        return_dict=return_dict,
    )
    hidden_states, timestep, encoder_hidden_states, encoder_attention_mask = args
    attention_kwargs = kwargs.get("attention_kwargs", attention_kwargs)
    return_dict = kwargs.get("return_dict", return_dict)

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(module, lora_scale)

    state = hook.state_manager.get_state()
    hook._maybe_reset_state_for_new_inference(state, module)

    batch_size, _, height, width = hidden_states.shape

    temb, encoder_hidden_states_processed = module.time_caption_embed(hidden_states, timestep, encoder_hidden_states)
    (
        image_patch_embeddings,
        context_rotary_emb,
        noise_rotary_emb,
        joint_rotary_emb,
        encoder_seq_lengths,
        seq_lengths,
    ) = module.rope_embedder(hidden_states, encoder_attention_mask)
    image_patch_embeddings = module.x_embedder(image_patch_embeddings)

    for layer in module.context_refiner:
        encoder_hidden_states_processed = layer(
            encoder_hidden_states_processed, encoder_attention_mask, context_rotary_emb
        )
    for layer in module.noise_refiner:
        image_patch_embeddings = layer(image_patch_embeddings, None, noise_rotary_emb, temb)

    max_seq_len = max(seq_lengths)
    input_to_main_loop = image_patch_embeddings.new_zeros(batch_size, max_seq_len, module.config.hidden_size)
    for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        input_to_main_loop[i, :enc_len] = encoder_hidden_states_processed[i, :enc_len]
        input_to_main_loop[i, enc_len:seq_len_val] = image_patch_embeddings[i]

    use_mask = len(set(seq_lengths)) > 1
    attention_mask_for_main_loop_arg = None
    if use_mask:
        mask = input_to_main_loop.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            mask[i, :seq_len_val] = True
        attention_mask_for_main_loop_arg = mask

    # Inline extractor: Lumina2 uses layers[0].norm1
    modulated_inp = module.layers[0].norm1(input_to_main_loop, temb)[0]

    cache_key = max_seq_len
    if cache_key not in state.cache_dict:
        state.cache_dict[cache_key] = {
            "previous_modulated_input": None,
            "previous_residual": None,
            "accumulated_rel_l1_distance": 0.0,
        }
    current_cache = state.cache_dict[cache_key]

    if state.cnt == 0 or state.cnt == state.num_steps - 1:
        should_calc = True
        current_cache["accumulated_rel_l1_distance"] = 0.0
    else:
        if current_cache["previous_modulated_input"] is not None:
            prev_mod_input = current_cache["previous_modulated_input"]
            prev_mean = prev_mod_input.abs().mean()
            if prev_mean.item() > 1e-9:
                rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).item()
            else:
                rel_l1_change = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float("inf")
            rescaled_distance = _rescale_distance(hook.coefficients, rel_l1_change)
            current_cache["accumulated_rel_l1_distance"] += rescaled_distance
            if current_cache["accumulated_rel_l1_distance"] < hook.config.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0

    current_cache["previous_modulated_input"] = modulated_inp.clone()

    if state.uncond_seq_len is None:
        state.uncond_seq_len = cache_key
    if cache_key != state.uncond_seq_len:
        state.cnt += 1
        if state.cnt >= state.num_steps:
            state.cnt = 0

    if not should_calc and current_cache["previous_residual"] is not None:
        processed_hidden_states = input_to_main_loop + current_cache["previous_residual"]
    else:
        current_processing_states = input_to_main_loop
        for layer in module.layers:
            current_processing_states = layer(
                current_processing_states, attention_mask_for_main_loop_arg, joint_rotary_emb, temb
            )
        processed_hidden_states = current_processing_states
        current_cache["previous_residual"] = processed_hidden_states - input_to_main_loop

    output_after_norm = module.norm_out(processed_hidden_states, temb)
    p = module.config.patch_size
    final_output_list = []
    for i, (enc_len, seq_len_val) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
        image_part = output_after_norm[i][enc_len:seq_len_val]
        h_p, w_p = height // p, width // p
        reconstructed_image = (
            image_part.view(h_p, w_p, p, p, module.out_channels).permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)
        )
        final_output_list.append(reconstructed_image)
    final_output_tensor = torch.stack(final_output_list, dim=0)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(module, lora_scale)

    if not return_dict:
        return (final_output_tensor,)
    return Transformer2DModelOutput(sample=final_output_tensor)


def _cogvideox_teacache_forward(
    hook: "TeaCacheHook",
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    """TeaCache forward for CogVideoX models (handles dual residual caching)."""
    # Handle accelerate CPU offload compatibility - moves module and inputs to GPU if needed
    args, kwargs = _handle_accelerate_hook(
        module,
        hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_cond=timestep_cond,
        ofs=ofs,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=attention_kwargs,
        return_dict=return_dict,
    )
    hidden_states, encoder_hidden_states, timestep = args
    timestep_cond = kwargs.get("timestep_cond", timestep_cond)
    ofs = kwargs.get("ofs", ofs)
    image_rotary_emb = kwargs.get("image_rotary_emb", image_rotary_emb)
    attention_kwargs = kwargs.get("attention_kwargs", attention_kwargs)
    return_dict = kwargs.get("return_dict", return_dict)

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(module, lora_scale)

    state = hook.state_manager.get_state()
    hook._maybe_reset_state_for_new_inference(state, module, reset_encoder_residual=True)

    batch_size, num_frames, _, height, width = hidden_states.shape

    t_emb = module.time_proj(timestep)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = module.time_embedding(t_emb, timestep_cond)

    if module.ofs_embedding is not None:
        ofs_emb = module.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = module.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    hs = module.patch_embed(encoder_hidden_states, hidden_states)
    hs = module.embedding_dropout(hs)

    text_seq_length = encoder_hidden_states.shape[1]
    enc = hs[:, :text_seq_length]
    hs = hs[:, text_seq_length:]

    # Inline extractor: CogVideoX uses timestep embedding directly
    modulated_inp = emb

    # Caching decision and execution (dual residual)
    if _should_compute_dual(state, modulated_inp, hook.coefficients, hook.config.rel_l1_thresh):
        # Full computation path
        ori_hs = hs.clone()
        ori_enc = enc.clone()
        for block in module.transformer_blocks:
            if torch.is_grad_enabled() and module.gradient_checkpointing:
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hs, enc = torch.utils.checkpoint.checkpoint(
                    lambda *a: block(*a),
                    hs,
                    enc,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hs, enc = block(
                    hidden_states=hs,
                    encoder_hidden_states=enc,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
        _update_state_dual(state, hs, enc, ori_hs, ori_enc, modulated_inp)
    else:
        # Cached path
        hs, enc = _apply_cached_residual_dual(state, hs, enc, modulated_inp)

    if not module.config.use_rotary_positional_embeddings:
        hs = module.norm_final(hs)
    else:
        hs_cat = torch.cat([enc, hs], dim=1)
        hs_cat = module.norm_final(hs_cat)
        hs = hs_cat[:, text_seq_length:]

    hs = module.norm_out(hs, temb=emb)
    hs = module.proj_out(hs)

    p = module.config.patch_size
    p_t = module.config.patch_size_t
    if p_t is None:
        output = hs.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hs.reshape(batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p)
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(module, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def apply_teacache(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    """
    Apply TeaCache optimization to a transformer model.

    This function registers a TeaCacheHook on the provided transformer, enabling adaptive caching of transformer block
    computations based on timestep embedding similarity. The hook intercepts the forward pass and implements the
    TeaCache algorithm to achieve 1.5x-2x speedup with minimal quality loss.

    Reference: [TeaCache: Timestep Embedding Aware Cache for Efficient Diffusion Model Inference](https://arxiv.org/abs/2411.19108)

    Args:
        module (`torch.nn.Module`):
            The transformer model to optimize (e.g., FluxTransformer2DModel, CogVideoXTransformer3DModel).
        config (`TeaCacheConfig`):
            Configuration specifying caching threshold and optional callbacks.

    Example:
        ```python
        from diffusers import FluxPipeline
        from diffusers.hooks import TeaCacheConfig

        # Load FLUX pipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        # Enable TeaCache via CacheMixin (recommended)
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        pipe.transformer.enable_cache(config)

        # Generate with caching enabled
        image = pipe("A cat on a windowsill", num_inference_steps=4).images[0]

        # Disable caching
        pipe.transformer.disable_cache()
        ```

    Note:
        For most use cases, it's recommended to use the CacheMixin interface: `pipe.transformer.enable_cache(...)`
        which provides additional convenience methods like `disable_cache()` for easy toggling.
    """
    # Register hook on main transformer
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TeaCacheHook(config)
    registry.register_hook(hook, _TEACACHE_HOOK)
