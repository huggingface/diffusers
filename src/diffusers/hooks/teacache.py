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

from ..loaders.peft import PeftAdapterMixin
from ..models.modeling_outputs import Transformer2DModelOutput
from ..utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_TEACACHE_HOOK = "teacache"


def _handle_accelerate_hook(module: torch.nn.Module, *args, **kwargs) -> Tuple[tuple, dict]:
    """Handle accelerate CPU offload hook compatibility."""
    if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "pre_forward"):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
    return args, kwargs


def _extract_lora_scale(attention_kwargs: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
    """Extract LoRA scale from attention kwargs."""
    if attention_kwargs is None:
        return None, 1.0
    attention_kwargs = attention_kwargs.copy()
    return attention_kwargs, attention_kwargs.pop("scale", 1.0)


def _get_model_config() -> Dict[str, Dict[str, Any]]:
    """
    Model configuration mapping for TeaCache.

    Keys are actual model class names from diffusers.models.transformers.
    Variant-specific coefficients are detected via config._name_or_path.

    Polynomial coefficients rescale L1 distances for caching decisions.
    The 4th-degree polynomial: c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]

    Coefficient derivation:
    - Coefficients are calibrated empirically by fitting polynomial curves to
      L1 distance measurements during inference
    - For models with similar architectures, coefficients can be transferred
      (e.g., FLUX -> TangoFlux, CogVideoX-5B -> CogVideoX-1.5)
    - Users can provide custom coefficients via TeaCacheConfig

    Sources:
    - FLUX: https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py
    - Mochi: https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Mochi/
    - Lumina2: https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Lumina2/
    - CogVideoX: https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4CogVideoX1.5/
    """
    return {
        "FluxTransformer2DModel": {
            "forward_func": _flux_teacache_forward,
            "coefficients": [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
            "variants": {
                "kontext": [-1.04655119e03, 3.12563399e02, -1.69500694e01, 4.10995971e-01, 3.74537863e-02],
            },
        },
        "MochiTransformer3DModel": {
            "forward_func": _mochi_teacache_forward,
            "coefficients": [-3.51241319e03, 8.11675948e02, -6.09400215e01, 2.42429681e00, 3.05291719e-03],
        },
        "Lumina2Transformer2DModel": {
            "forward_func": _lumina2_teacache_forward,
            "coefficients": [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344],
        },
        "CogVideoXTransformer3DModel": {
            "forward_func": _cogvideox_teacache_forward,
            "coefficients": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
            "variants": {
                "1.5-5b-i2v": [1.22842302e02, -1.04088754e02, 2.62981677e01, -3.06001e-01, 3.71213220e-02],
                "1.5-5b": [2.50210439e02, -1.65061612e02, 3.57804877e01, -7.81551492e-01, 3.58559703e-02],
                "2b": [-3.10658903e01, 2.54732368e01, -5.92380459e00, 1.75769064e00, -3.61568434e-03],
                "5b": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
            },
        },
    }


def _auto_detect_model_type(module: torch.nn.Module) -> Tuple[str, Callable, List[float]]:
    """Auto-detect model type and coefficients from class name and config path."""
    class_name = module.__class__.__name__
    config_path = getattr(getattr(module, "config", None), "_name_or_path", "").lower()
    model_config = _get_model_config()

    # Exact match on class name (no substring matching)
    if class_name not in model_config:
        raise ValueError(
            f"TeaCache: Unsupported model '{class_name}'. Supported models: {', '.join(model_config.keys())}"
        )

    config = model_config[class_name]
    coefficients = config["coefficients"]
    forward_func = config["forward_func"]

    # Check for variant-specific coefficients via config path
    if "variants" in config:
        for variant_key, variant_coeffs in config["variants"].items():
            if variant_key in config_path:
                coefficients = variant_coeffs
                logger.debug(f"TeaCache: Using {class_name} variant '{variant_key}' coefficients")
                break

    return class_name, forward_func, coefficients


def _rescale_distance(coefficients: List[float], x: float) -> float:
    """Polynomial rescaling: c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]"""
    c = coefficients
    return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x + c[4]


def _rescale_distance_tensor(coefficients: List[float], x: torch.Tensor) -> torch.Tensor:
    """Polynomial rescaling using tensor operations (torch.compile friendly)."""
    c = coefficients
    return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x + c[4]


def _compute_rel_l1_distance(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Compute relative L1 distance between tensors."""
    prev_mean = previous.abs().mean()
    if prev_mean.item() > 1e-9:
        return ((current - previous).abs().mean() / prev_mean).item()
    return 0.0 if current.abs().mean().item() < 1e-9 else float("inf")


def _compute_rel_l1_distance_tensor(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Compute relative L1 distance as a tensor (torch.compile friendly)."""
    prev_mean = previous.abs().mean()
    curr_diff_mean = (current - previous).abs().mean()

    # Use torch.where for conditional logic instead of Python if
    rel_distance = torch.where(
        prev_mean > 1e-9,
        curr_diff_mean / prev_mean,
        torch.where(
            current.abs().mean() < 1e-9,
            torch.zeros(1, device=current.device, dtype=current.dtype),
            torch.full((1,), float("inf"), device=current.device, dtype=current.dtype),
        ),
    )
    return rel_distance.squeeze()


def _should_compute(state, modulated_inp, coefficients, rel_l1_thresh):
    """Determine if full computation is needed (single residual models).

    Uses tensor-only operations for torch.compile compatibility. One .item() call
    remains for the final branching decision since Python control flow requires a boolean.
    """
    is_first_step = state.cnt == 0
    is_last_step = state.num_steps > 0 and state.cnt == state.num_steps - 1
    missing_state = state.previous_modulated_input is None or state.previous_residual is None

    if is_first_step or is_last_step or missing_state:
        state.accumulated_rel_l1_distance = torch.zeros(1, device=modulated_inp.device, dtype=modulated_inp.dtype)
        return True

    # Initialize accumulated distance tensor if needed (first non-boundary step)
    if not isinstance(state.accumulated_rel_l1_distance, torch.Tensor):
        state.accumulated_rel_l1_distance = torch.zeros(1, device=modulated_inp.device, dtype=modulated_inp.dtype)

    rel_distance = _compute_rel_l1_distance_tensor(modulated_inp, state.previous_modulated_input)
    rescaled = _rescale_distance_tensor(coefficients, rel_distance)
    state.accumulated_rel_l1_distance = state.accumulated_rel_l1_distance + rescaled

    # Single .item() for branching (unavoidable for Python control flow)
    should_compute = (state.accumulated_rel_l1_distance >= rel_l1_thresh).item()

    if should_compute:
        state.accumulated_rel_l1_distance = torch.zeros(1, device=modulated_inp.device, dtype=modulated_inp.dtype)

    return should_compute


def _should_compute_dual(state, modulated_inp, coefficients, rel_l1_thresh):
    """Determine if full computation is needed (dual residual models like CogVideoX)."""
    if state.previous_residual is None or state.previous_residual_encoder is None:
        return True
    return _should_compute(state, modulated_inp, coefficients, rel_l1_thresh)


def _update_state(state, output, original_input, modulated_inp):
    """Update cache state after full computation."""
    state.previous_residual = output - original_input
    state.previous_modulated_input = modulated_inp
    state.cnt += 1


def _update_state_dual(state, hs_output, enc_output, hs_original, enc_original, modulated_inp):
    """Update cache state after full computation (dual residual for CogVideoX)."""
    state.previous_residual = hs_output - hs_original
    state.previous_residual_encoder = enc_output - enc_original
    state.previous_modulated_input = modulated_inp
    state.cnt += 1


def _apply_cached_residual(state, input_tensor, modulated_inp):
    """Apply cached residual (fast path)."""
    output = input_tensor + state.previous_residual
    state.previous_modulated_input = modulated_inp
    state.cnt += 1
    return output


def _apply_cached_residual_dual(state, hs, enc, modulated_inp):
    """Apply cached residuals (fast path for CogVideoX)."""
    hs_out = hs + state.previous_residual
    enc_out = enc + state.previous_residual_encoder
    state.previous_modulated_input = modulated_inp
    state.cnt += 1
    return hs_out, enc_out


@dataclass
class TeaCacheConfig:
    r"""
    Configuration for [TeaCache](https://huggingface.co/papers/2411.19108).

    TeaCache (Timestep Embedding Aware Cache) speeds up diffusion model inference by reusing transformer block
    computations when consecutive timestep embeddings are similar. It uses polynomial rescaling of L1 distances between
    modulated inputs to decide when to cache.

    Currently supports: FLUX, FLUX-Kontext, Mochi, Lumina2, and CogVideoX models. Model type is auto-detected.

    Attributes:
        rel_l1_thresh (`float`, defaults to `0.2`):
            Threshold for accumulated relative L1 distance. When below this threshold, the cached residual is reused.
            Recommended values: 0.25 for ~1.5x speedup, 0.4 for ~1.8x, 0.6 for ~2.0x. Mochi models require lower
            thresholds (0.06-0.09).
        coefficients (`List[float]`, *optional*):
            Polynomial coefficients for rescaling L1 distance. Auto-detected based on model type if not provided.
            Evaluated as: `c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]`.
        current_timestep_callback (`Callable[[], int]`, *optional*):
            Callback returning current timestep. Used for debugging/statistics.
        num_inference_steps (`int`, *optional*):
            Total inference steps. Ensures first/last timesteps are always computed. Auto-detected if not provided.
        num_inference_steps_callback (`Callable[[], int]`, *optional*):
            Callback returning total inference steps. Alternative to `num_inference_steps`.

    Example:
        ```python
        >>> import torch
        >>> from diffusers import FluxPipeline
        >>> from diffusers.hooks import TeaCacheConfig

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> config = TeaCacheConfig(rel_l1_thresh=0.2)
        >>> pipe.transformer.enable_cache(config)

        >>> image = pipe("A cat sitting on a windowsill", num_inference_steps=4).images[0]
        >>> pipe.transformer.disable_cache()
        ```
    """

    rel_l1_thresh: float = 0.2
    coefficients: Optional[List[float]] = None
    current_timestep_callback: Optional[Callable[[], int]] = None
    num_inference_steps: Optional[int] = None
    num_inference_steps_callback: Optional[Callable[[], int]] = None

    def __post_init__(self):
        self._validate_threshold()
        self._validate_coefficients()

    def _validate_threshold(self):
        """Validate rel_l1_thresh parameter."""
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

    def _validate_coefficients(self):
        """Validate coefficients parameter if provided."""
        if self.coefficients is None:
            return
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
    r"""
    State for [TeaCache](https://huggingface.co/papers/2411.19108).

    Tracks caching state across diffusion timesteps, including counters, accumulated distances, and cached residuals.
    """

    def __init__(self):
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = None  # Tensor, initialized on first use
        self.previous_modulated_input = None
        self.previous_residual = None
        # CogVideoX-specific: dual residual caching (encoder + hidden_states)
        self.previous_residual_encoder = None  # Only used by CogVideoX
        # Lumina2-specific: per-sequence-length caching for variable sequence lengths
        # Other models don't use these fields but they're allocated for simplicity
        self.cache_dict = {}  # Only used by Lumina2
        self.uncond_seq_len = None  # Only used by Lumina2 (shorter seq = uncond)
        self.cond_seq_len = None  # Only used by Lumina2 (longer seq = cond)

    def reset(self):
        """Reset all state variables to initial values for a new inference run."""
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_rel_l1_distance = None
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_residual_encoder = None
        self.cache_dict = {}
        self.uncond_seq_len = None
        self.cond_seq_len = None

    def __repr__(self) -> str:
        acc_dist = self.accumulated_rel_l1_distance
        acc_dist_str = f"{acc_dist.item():.6f}" if isinstance(acc_dist, torch.Tensor) else "None"
        return (
            f"TeaCacheState(\n"
            f"  cnt={self.cnt},\n"
            f"  num_steps={self.num_steps},\n"
            f"  accumulated_rel_l1_distance={acc_dist_str},\n"
            f"  previous_modulated_input={'cached' if self.previous_modulated_input is not None else 'None'},\n"
            f"  previous_residual={'cached' if self.previous_residual is not None else 'None'}\n"
            f")"
        )


class TeaCacheHook(ModelHook):
    r"""
    Hook implementing [TeaCache](https://huggingface.co/papers/2411.19108) for transformer models.

    Intercepts transformer forward pass and implements adaptive caching based on timestep embedding similarity. First
    and last timesteps are always computed fully (never cached) to ensure maximum quality.
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
        """Reset state if inference run completed. Initialize num_steps on first timestep if not set."""
        # Reset if we've completed all steps (new inference run)
        if state.cnt == state.num_steps and state.num_steps > 0:
            logger.debug("TeaCache: Inference run completed, resetting state")
            state.cnt = 0
            state.accumulated_rel_l1_distance = None
            state.previous_modulated_input = None
            state.previous_residual = None
            if reset_encoder_residual:
                state.previous_residual_encoder = None
            # Lumina2-specific: clear per-sequence-length cache
            state.cache_dict.clear()
            state.uncond_seq_len = None
            state.cond_seq_len = None

        # Set num_steps on first timestep (priority: config > callback > module attribute)
        if state.cnt == 0 and state.num_steps == 0:
            if self.config.num_inference_steps is not None:
                state.num_steps = self.config.num_inference_steps
            elif self.config.num_inference_steps_callback is not None:
                state.num_steps = self.config.num_inference_steps_callback()
            elif hasattr(module, "num_steps"):
                state.num_steps = module.num_steps

            if state.num_steps > 0:
                logger.debug(f"TeaCache: Using {state.num_steps} inference steps")

    def initialize_hook(self, module):
        # Context is set by pipeline's cache_context() calls in the denoising loop.
        # This enables proper state isolation between cond/uncond branches.
        # See PR #12652 for discussion on this design decision.

        # Auto-detect model type, forward function, and coefficients
        self.model_type, self._forward_func, detected_coefficients = _auto_detect_model_type(module)

        # Validate model has required attributes for TeaCache
        if self.model_type == "FluxTransformer2DModel":
            if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
                raise ValueError(f"TeaCache: {self.model_type} model missing transformer_blocks")
            if not hasattr(module.transformer_blocks[0], "norm1"):
                raise ValueError(f"TeaCache: {self.model_type} transformer_blocks[0] missing norm1")
        elif self.model_type == "MochiTransformer3DModel":
            if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
                raise ValueError(f"TeaCache: {self.model_type} model missing transformer_blocks")
            if not hasattr(module.transformer_blocks[0], "norm1"):
                raise ValueError(f"TeaCache: {self.model_type} transformer_blocks[0] missing norm1")
        elif self.model_type == "Lumina2Transformer2DModel":
            if not hasattr(module, "layers") or len(module.layers) == 0:
                raise ValueError("TeaCache: Lumina2Transformer2DModel model missing layers")
        elif self.model_type == "CogVideoXTransformer3DModel":
            if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
                raise ValueError(f"TeaCache: {self.model_type} model missing transformer_blocks")

        # Use user-provided coefficients if available, otherwise use auto-detected
        if self.config.coefficients is None:
            self.coefficients = detected_coefficients
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
    controlnet_block_samples: Optional[List[torch.Tensor]] = None,
    controlnet_single_block_samples: Optional[List[torch.Tensor]] = None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    **kwargs,
):
    """
    TeaCache-enabled forward pass for FLUX transformer models.

    This function is adapted from FluxTransformer2DModel.forward() with TeaCache
    caching logic inserted. When the original forward is updated, this function
    should be reviewed for compatibility.

    Reference:
        Source: src/diffusers/models/transformers/transformer_flux.py
        Class: FluxTransformer2DModel
        Method: forward()

    Key sections that must stay in sync:
        - Embedding computation (x_embedder, time_text_embed)
        - Position embedding (pos_embed, image_rotary_emb)
        - ControlNet integration (controlnet_block_samples handling)
        - Block iteration (transformer_blocks, single_transformer_blocks)
        - Output normalization (norm_out, proj_out)

    TeaCache-specific additions:
        - Extract modulated_inp from first block's norm1
        - Conditional computation based on _should_compute()
        - Update state after full computation / apply cached residual
    """
    args, extra_kwargs = _handle_accelerate_hook(
        module,
        hidden_states,
        timestep,
        pooled_projections,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
        return_dict=return_dict,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
        **kwargs,
    )
    hidden_states, timestep, pooled_projections, encoder_hidden_states, txt_ids, img_ids = args
    controlnet_block_samples = extra_kwargs.pop("controlnet_block_samples", controlnet_block_samples)
    controlnet_single_block_samples = extra_kwargs.pop(
        "controlnet_single_block_samples", controlnet_single_block_samples
    )
    controlnet_blocks_repeat = extra_kwargs.pop("controlnet_blocks_repeat", controlnet_blocks_repeat)
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

    modulated_inp = module.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]

    if _should_compute(state, modulated_inp, hook.coefficients, hook.config.rel_l1_thresh):
        ori_hs = hidden_states.clone()
        enc = module.context_embedder(encoder_hidden_states)

        txt = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
        img = img_ids[0] if img_ids.ndim == 3 else img_ids
        ids = torch.cat((txt, img), dim=0)
        image_rotary_emb = module.pos_embed(ids)

        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        for index_block, block in enumerate(module.transformer_blocks):
            enc, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            # ControlNet residual
            if controlnet_block_samples is not None:
                interval_control = len(module.transformer_blocks) / len(controlnet_block_samples)
                interval_control = (
                    int(interval_control) if interval_control == int(interval_control) else int(interval_control) + 1
                )
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(module.single_transformer_blocks):
            enc, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=enc,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            # ControlNet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(module.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = (
                    int(interval_control) if interval_control == int(interval_control) else int(interval_control) + 1
                )
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        _update_state(state, hidden_states, ori_hs, modulated_inp)
    else:
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
    """
    TeaCache-enabled forward pass for Mochi transformer models.

    This function is adapted from MochiTransformer3DModel.forward() with TeaCache
    caching logic inserted. When the original forward is updated, this function
    should be reviewed for compatibility.

    Reference:
        Source: src/diffusers/models/transformers/transformer_mochi.py
        Class: MochiTransformer3DModel
        Method: forward()

    Key sections that must stay in sync:
        - Time embedding (time_embed)
        - Patch embedding (patch_embed)
        - RoPE computation (rope, pos_frequencies)
        - Block iteration (transformer_blocks) with gradient checkpointing
        - Output normalization (norm_out) and projection (proj_out)
        - Output reshaping for video format

    TeaCache-specific additions:
        - Extract modulated_inp from first block's norm1
        - Conditional computation based on _should_compute()
        - Update state after full computation / apply cached residual
    """
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

    attention_kwargs, lora_scale = _extract_lora_scale(attention_kwargs)
    if USE_PEFT_BACKEND and isinstance(module, PeftAdapterMixin):
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

    modulated_inp = module.transformer_blocks[0].norm1(hidden_states, temb)[0]

    if _should_compute(state, modulated_inp, hook.coefficients, hook.config.rel_l1_thresh):
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
        hidden_states = module.norm_out(hidden_states, temb)
        _update_state(state, hidden_states, ori_hs, modulated_inp)
    else:
        hidden_states = _apply_cached_residual(state, hidden_states, modulated_inp)

    hidden_states = module.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
    hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
    output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if USE_PEFT_BACKEND and isinstance(module, PeftAdapterMixin):
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
    """
    TeaCache-enabled forward pass for Lumina2 transformer models.

    This function is adapted from Lumina2Transformer2DModel.forward() with TeaCache
    caching logic inserted. When the original forward is updated, this function
    should be reviewed for compatibility.

    Reference:
        Source: src/diffusers/models/transformers/transformer_lumina2.py
        Class: Lumina2Transformer2DModel
        Method: forward()

    Key sections that must stay in sync:
        - Time/caption embedding (time_caption_embed)
        - RoPE embedding (rope_embedder)
        - Context refiner and noise refiner loops
        - Main transformer layers loop
        - Output normalization and reconstruction

    TeaCache-specific additions:
        - Per-sequence-length caching via cache_dict (needed for variable sequence lengths in CFG)
        - Inline caching logic instead of _should_compute() due to per-sequence-length caching
        - Extract modulated_inp from first layer's norm1

    Note: Gradient checkpointing is not supported in this TeaCache implementation for Lumina2.
    """
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

    attention_kwargs, lora_scale = _extract_lora_scale(attention_kwargs)
    if USE_PEFT_BACKEND and isinstance(module, PeftAdapterMixin):
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

    modulated_inp = module.layers[0].norm1(input_to_main_loop, temb)[0]

    # Per-sequence-length caching for variable sequence lengths
    cache_key = max_seq_len
    if cache_key not in state.cache_dict:
        state.cache_dict[cache_key] = {
            "previous_modulated_input": None,
            "previous_residual": None,
            "accumulated_rel_l1_distance": 0.0,
        }
    cache = state.cache_dict[cache_key]

    # Track both cond (max) and uncond (min) sequence lengths
    # The longer sequence is always the conditional pass (more tokens from prompt)
    # The shorter sequence is always the unconditional pass
    if state.uncond_seq_len is None:
        state.uncond_seq_len = cache_key
        state.cond_seq_len = cache_key
    else:
        state.uncond_seq_len = min(state.uncond_seq_len, cache_key)
        state.cond_seq_len = max(state.cond_seq_len, cache_key)

    # Determine if we've seen both sequence lengths (CFG is active)
    has_both_lengths = state.cond_seq_len != state.uncond_seq_len
    is_cond_pass = cache_key == state.cond_seq_len

    # Increment counter BEFORE boundary check
    # For cond-first pipelines (Lumina2), increment at the START of each step (on cond pass)
    # This ensures both passes of the same step see the same cnt value
    # - Step 0: forwards 1-2, cnt=0 for both (no increment yet, has_both=False on forward 1)
    # - Step 1: forward 3 increments cnt 0→1, forwards 3-4 see cnt=1
    # - Step N-1: forward 2N-1 increments cnt to N-1, forwards 2N-1 and 2N see cnt=N-1
    if has_both_lengths and is_cond_pass:
        state.cnt += 1
        if state.cnt >= state.num_steps and state.num_steps > 0:
            state.cnt = 0

    # Boundary detection: first step (cnt=0) or last step (cnt=num_steps-1)
    is_first_step = state.cnt == 0
    is_last_step = state.num_steps > 0 and state.cnt == state.num_steps - 1
    is_boundary_step = is_first_step or is_last_step

    has_previous = cache["previous_modulated_input"] is not None

    if is_boundary_step or not has_previous:
        should_calc = True
        cache["accumulated_rel_l1_distance"] = 0.0
    else:
        rel_distance = _compute_rel_l1_distance(modulated_inp, cache["previous_modulated_input"])
        cache["accumulated_rel_l1_distance"] += _rescale_distance(hook.coefficients, rel_distance)
        if cache["accumulated_rel_l1_distance"] >= hook.config.rel_l1_thresh:
            should_calc = True
            cache["accumulated_rel_l1_distance"] = 0.0
        else:
            should_calc = False

    cache["previous_modulated_input"] = modulated_inp.clone()

    # Apply cached residual or compute full forward
    if not should_calc and cache["previous_residual"] is not None:
        processed_hidden_states = input_to_main_loop + cache["previous_residual"]
    else:
        processed_hidden_states = input_to_main_loop
        for layer in module.layers:
            processed_hidden_states = layer(
                processed_hidden_states, attention_mask_for_main_loop_arg, joint_rotary_emb, temb
            )
        cache["previous_residual"] = processed_hidden_states - input_to_main_loop

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

    if USE_PEFT_BACKEND and isinstance(module, PeftAdapterMixin):
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
    """
    TeaCache-enabled forward pass for CogVideoX transformer models.

    This function is adapted from CogVideoXTransformer3DModel.forward() with TeaCache
    caching logic inserted. When the original forward is updated, this function
    should be reviewed for compatibility.

    Reference:
        Source: src/diffusers/models/transformers/cogvideox_transformer_3d.py
        Class: CogVideoXTransformer3DModel
        Method: forward()

    Key sections that must stay in sync:
        - Time embedding (time_proj, time_embedding)
        - OFS embedding (ofs_proj, ofs_embedding) for 1.5 models
        - Patch embedding (patch_embed, embedding_dropout)
        - Block iteration (transformer_blocks) with gradient checkpointing
        - Final normalization (norm_final, norm_out)
        - Output reshaping for video format (patch_size, patch_size_t)

    TeaCache-specific additions:
        - Dual residual caching for encoder and hidden_states (CogVideoX-specific)
        - Uses _should_compute_dual() for threshold checking
        - Update state with _update_state_dual() / _apply_cached_residual_dual()
    """
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

    attention_kwargs, lora_scale = _extract_lora_scale(attention_kwargs)
    if USE_PEFT_BACKEND and isinstance(module, PeftAdapterMixin):
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

    modulated_inp = emb

    if _should_compute_dual(state, modulated_inp, hook.coefficients, hook.config.rel_l1_thresh):
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

    if USE_PEFT_BACKEND and isinstance(module, PeftAdapterMixin):
        unscale_lora_layers(module, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def apply_teacache(module: torch.nn.Module, config: TeaCacheConfig) -> None:
    r"""
    Applies [TeaCache](https://huggingface.co/papers/2411.19108) to a given module.

    TeaCache speeds up diffusion model inference (1.5x-2x) by caching transformer block computations when consecutive
    timestep embeddings are similar. Model type is auto-detected based on the module class name.

    Args:
        module (`torch.nn.Module`):
            The transformer model to optimize (e.g., FluxTransformer2DModel, CogVideoXTransformer3DModel).
        config (`TeaCacheConfig`):
            The configuration to use for TeaCache.

    Example:
        ```python
        >>> import torch
        >>> from diffusers import FluxPipeline
        >>> from diffusers.hooks import TeaCacheConfig, apply_teacache

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> apply_teacache(pipe.transformer, TeaCacheConfig(rel_l1_thresh=0.2))

        >>> image = pipe("A cat on a windowsill", num_inference_steps=4).images[0]
        ```
    """
    # Register hook on main transformer
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TeaCacheHook(config)
    registry.register_hook(hook, _TEACACHE_HOOK)
