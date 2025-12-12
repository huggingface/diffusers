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
from typing import Callable, List, Optional

import torch

from ..utils import logging
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_TEACACHE_HOOK = "teacache"

# Model-specific polynomial coefficients from TeaCache paper/reference implementations
_MODEL_COEFFICIENTS = {
    "Flux": [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
    "FluxKontext": [-1.04655119e03, 3.12563399e02, -1.69500694e01, 4.10995971e-01, 3.74537863e-02],
    "Mochi": [-3.51241319e03, 8.11675948e02, -6.09400215e01, 2.42429681e00, 3.05291719e-03],
    "Lumina2": [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344],
    "CogVideoX": [
        -1.53880483e03,
        8.43202495e02,
        -1.34363087e02,
        7.97131516e00,
        -5.23162339e-02,
    ],  # Default to 5b variant
    # CogVideoX model variants with specific coefficients
    "CogVideoX-2b": [-3.10658903e01, 2.54732368e01, -5.92380459e00, 1.75769064e00, -3.61568434e-03],
    "CogVideoX-5b": [-1.53880483e03, 8.43202495e02, -1.34363087e02, 7.97131516e00, -5.23162339e-02],
    "CogVideoX1.5-5B": [2.50210439e02, -1.65061612e02, 3.57804877e01, -7.81551492e-01, 3.58559703e-02],
    "CogVideoX1.5-5B-I2V": [1.22842302e02, -1.04088754e02, 2.62981677e01, -3.06001e-01, 3.71213220e-02],
}


def _flux_modulated_input_extractor(module, hidden_states, timestep_emb):
    """Extract modulated input for FLUX models."""
    return module.transformer_blocks[0].norm1(hidden_states, emb=timestep_emb)[0]


def _mochi_modulated_input_extractor(module, hidden_states, timestep_emb):
    """Extract modulated input for Mochi models."""
    # Mochi norm1 returns tuple: (modulated_inp, gate_msa, scale_mlp, gate_mlp)
    return module.transformer_blocks[0].norm1(hidden_states, timestep_emb)[0]


def _lumina2_modulated_input_extractor(module, hidden_states, timestep_emb):
    """Extract modulated input for Lumina2 models."""
    # Lumina2 uses 'layers' instead of 'transformer_blocks' and norm1 returns tuple
    # Note: This extractor expects input_to_main_loop as hidden_states (after preprocessing)
    return module.layers[0].norm1(hidden_states, timestep_emb)[0]


def _cogvideox_modulated_input_extractor(module, hidden_states, timestep_emb):
    """Extract modulated input for CogVideoX models."""
    # CogVideoX uses the timestep embedding directly, not from a block
    return timestep_emb


# Extractor registry - maps model types to extraction functions
# Multiple model variants can share the same extractor
# Order matters: more specific variants first (e.g., CogVideoX1.5-5B-I2V before CogVideoX)
_EXTRACTOR_REGISTRY = {
    "FluxKontext": _flux_modulated_input_extractor,
    "Flux": _flux_modulated_input_extractor,
    "Mochi": _mochi_modulated_input_extractor,
    "Lumina2": _lumina2_modulated_input_extractor,
    "CogVideoX1.5-5B-I2V": _cogvideox_modulated_input_extractor,
    "CogVideoX1.5-5B": _cogvideox_modulated_input_extractor,
    "CogVideoX-2b": _cogvideox_modulated_input_extractor,
    "CogVideoX-5b": _cogvideox_modulated_input_extractor,
    "CogVideoX": _cogvideox_modulated_input_extractor,
}


def _auto_detect_extractor(module):
    """Auto-detect and return appropriate extractor."""
    return _EXTRACTOR_REGISTRY[_auto_detect_model_type(module)]


def _auto_detect_model_type(module):
    """Auto-detect model type from class name and config path."""
    class_name = module.__class__.__name__
    config_path = getattr(getattr(module, "config", None), "_name_or_path", "").lower()
    
    # Check config path first (for variants), then class name (ordered most specific first)
    for model_type in _EXTRACTOR_REGISTRY:
        if model_type.lower() in config_path or model_type in class_name:
            if model_type not in _MODEL_COEFFICIENTS:
                raise ValueError(f"TeaCache: No coefficients for '{model_type}'")
            return model_type
    
    raise ValueError(f"TeaCache: Unsupported model '{class_name}'. Supported: {', '.join(_EXTRACTOR_REGISTRY)}")


def _get_model_coefficients(model_type):
    """Get polynomial coefficients for a specific model type.
    
    Args:
        model_type: Model type string (e.g., "Flux", "Mochi")
        
    Raises:
        ValueError: If coefficients not found for model type.
    """
    if model_type not in _MODEL_COEFFICIENTS:
        available_models = ", ".join(_MODEL_COEFFICIENTS.keys())
        raise ValueError(
            f"TeaCache: No coefficients found for model type '{model_type}'. "
            f"Available models: {available_models}. "
            f"Please provide coefficients explicitly in TeaCacheConfig."
        )
    return _MODEL_COEFFICIENTS[model_type]


@dataclass
class TeaCacheConfig:
    r"""
    Configuration for [TeaCache](https://liewfeng.github.io/TeaCache/) applied to transformer models.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique that speeds up diffusion model inference
    by reusing transformer block computations when consecutive timestep embeddings are similar. It uses polynomial
    rescaling of L1 distances between modulated inputs to intelligently decide when to cache.

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
        coefficients (`List[float]`, *optional*, defaults to polynomial coefficients from TeaCache paper):
            Polynomial coefficients used for rescaling the raw L1 distance. These coefficients transform the relative
            L1 distance into a model-specific caching signal. If not provided, defaults to the coefficients determined
            for FLUX models in the TeaCache paper: [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00,
            2.64230861e-01]. The polynomial is evaluated as: `c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]` where x
            is the relative L1 distance.
        extract_modulated_input_fn (`Callable`, *optional*, defaults to auto-detection):
            Function to extract modulated input from the transformer module. Takes (module, hidden_states,
            timestep_emb) and returns the modulated input tensor. If not provided, auto-detects based on model type.
        current_timestep_callback (`Callable[[], int]`, *optional*, defaults to `None`):
            Callback function that returns the current timestep during inference. This is used internally for debugging
            and statistics tracking. If not provided, TeaCache will still function correctly.
        num_inference_steps (`int`, *optional*, defaults to `None`):
            Total number of inference steps. Required for proper state management - ensures first and last timesteps
            are always computed (never cached) and that state resets between inference runs. If not provided,
            TeaCache will attempt to detect via callback or module attribute.
        num_inference_steps_callback (`Callable[[], int]`, *optional*, defaults to `None`):
            Callback function that returns the total number of inference steps. Alternative to `num_inference_steps`
            for dynamic step counts.

    Examples:
        ```python
        from diffusers import FluxPipeline
        from diffusers.hooks import TeaCacheConfig

        # Load FLUX pipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        # Enable TeaCache with auto-detection (1.5x speedup)
        pipe.transformer.enable_teacache(rel_l1_thresh=0.2)

        # Or with explicit config
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
    extract_modulated_input_fn: Optional[Callable] = None
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
            f"  extract_modulated_input_fn={self.extract_modulated_input_fn},\n"
            f"  current_timestep_callback={self.current_timestep_callback},\n"
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
        # For models that cache both encoder and hidden_states residuals (e.g., CogVideoX)
        self.previous_residual_encoder = None
        # For models with variable sequence lengths (e.g., Lumina2)
        self.cache_dict = {}
        self.uncond_seq_len = None

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
        rescale_func (np.poly1d):
            Polynomial function for rescaling L1 distances using model-specific coefficients.
        state_manager (StateManager):
            Manages TeaCacheState across forward passes, maintaining counters and cached values.
    """

    _is_stateful = True

    def __init__(self, config: TeaCacheConfig):
        super().__init__()
        self.config = config
        # Set default rescale_func with config coefficients (will be updated in initialize_hook if needed)
        # This ensures rescale_func is always valid, even if initialize_hook isn't called (e.g., in tests)
        default_coeffs = config.coefficients if config.coefficients else _MODEL_COEFFICIENTS["Flux"]
        self.coefficients = default_coeffs
        self.rescale_func = self._create_rescale_func(default_coeffs)
        self.state_manager = StateManager(TeaCacheState, (), {})
        self.extractor_fn = None
        self.model_type = None

    @staticmethod
    def _create_rescale_func(coefficients):
        """Create polynomial rescale function from coefficients.
        
        Evaluates: c[0]*x^4 + c[1]*x^3 + c[2]*x^2 + c[3]*x + c[4]
        """
        def rescale(x):
            return coefficients[0] * x**4 + coefficients[1] * x**3 + coefficients[2] * x**2 + coefficients[3] * x + coefficients[4]
        return rescale

    def _maybe_reset_state_for_new_inference(self, state, module, reset_encoder_residual=False):
        """Reset state if we've completed all steps (start of new inference run).
        
        Also initializes num_steps on first timestep if not set.
        
        Args:
            state: TeaCacheState instance.
            module: The transformer module.
            reset_encoder_residual: If True, also reset previous_residual_encoder (for CogVideoX).
        """
        # Reset counter if we've completed all steps (new inference run)
        if state.cnt == state.num_steps and state.num_steps > 0:
            logger.info("TeaCache inference completed")
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
                logger.info(f"TeaCache: Using {state.num_steps} inference steps")

    def initialize_hook(self, module):
        self.state_manager.set_context("teacache")

        # Strict auto-detection
        if self.config.extract_modulated_input_fn is None:
            self.extractor_fn = _auto_detect_extractor(module)  # Raises if unsupported
            self.model_type = _auto_detect_model_type(module)   # Raises if unsupported
        else:
            self.extractor_fn = self.config.extract_modulated_input_fn
            # Still try to detect model type for coefficients
            try:
                self.model_type = _auto_detect_model_type(module)
            except ValueError:
                self.model_type = None  # User provided custom extractor
                logger.warning(
                    f"TeaCache: Using custom extractor for {module.__class__.__name__}. "
                    f"Coefficients must be provided explicitly."
                )

        # Auto-detect coefficients if not provided by user
        if self.config.coefficients is None:
            if self.model_type is None:
                raise ValueError(
                    "TeaCache: Cannot auto-detect coefficients when using custom extractor. "
                    "Please provide coefficients explicitly in TeaCacheConfig."
                )
            self.coefficients = _get_model_coefficients(self.model_type)  # Raises if not found
            logger.info(f"TeaCache: Using {self.model_type} coefficients")
        else:
            self.coefficients = self.config.coefficients
            logger.info(f"TeaCache: Using user-provided coefficients")

        # Initialize rescale function with final coefficients
        self.rescale_func = self._create_rescale_func(self.coefficients)

        return module

    def new_forward(self, module, *args, **kwargs):
        """
        Route to model-specific forward handler based on detected model type.
        """
        module_class_name = module.__class__.__name__

        if "Flux" in module_class_name:
            return self._handle_flux_forward(module, *args, **kwargs)
        elif "Mochi" in module_class_name:
            return self._handle_mochi_forward(module, *args, **kwargs)
        elif "Lumina2" in module_class_name:
            return self._handle_lumina2_forward(module, *args, **kwargs)
        elif "CogVideoX" in module_class_name:
            return self._handle_cogvideox_forward(module, *args, **kwargs)
        else:
            # Default to FLUX handler for backward compatibility
            logger.warning(
                f"TeaCache: Unknown model type {module_class_name}, using FLUX handler. Results may be incorrect."
            )
            return self._handle_flux_forward(module, *args, **kwargs)

    def _handle_flux_forward(
        self,
        module,
        hidden_states,
        timestep,
        pooled_projections,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        return_dict=True,
        **kwargs,
    ):
        """
        Handle FLUX transformer forward pass with TeaCache.

        Args:
            module: The FluxTransformer2DModel instance.
            hidden_states (`torch.Tensor`): Input latent tensor.
            timestep (`torch.Tensor`): Current diffusion timestep.
            pooled_projections (`torch.Tensor`): Pooled text embeddings.
            encoder_hidden_states (`torch.Tensor`): Text encoder outputs.
            txt_ids (`torch.Tensor`): Position IDs for text tokens.
            img_ids (`torch.Tensor`): Position IDs for image tokens.
            return_dict (`bool`): Whether to return a dict.
            **kwargs: Additional arguments.

        Returns:
            `torch.Tensor` or `Transformer2DModelOutput`: Denoised output.
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        state = self.state_manager.get_state()
        self._maybe_reset_state_for_new_inference(state, module)

        # Process inputs like original TeaCache
        # Must process hidden_states through x_embedder first
        hidden_states = module.x_embedder(hidden_states)

        # Extract timestep embedding
        timestep_scaled = timestep.to(hidden_states.dtype) * 1000
        if kwargs.get("guidance") is not None:
            guidance = kwargs["guidance"].to(hidden_states.dtype) * 1000
            temb = module.time_text_embed(timestep_scaled, guidance, pooled_projections)
        else:
            temb = module.time_text_embed(timestep_scaled, pooled_projections)

        # Extract modulated input using configured extractor (extractors don't modify inputs)
        modulated_inp = self.extractor_fn(module, hidden_states, temb)

        # Make caching decision
        should_calc = self._should_compute_full_transformer(state, modulated_inp)

        if not should_calc:
            # Fast path: apply cached residual
            hidden_states = hidden_states + state.previous_residual
        else:
            # Slow path: full computation inline (like original TeaCache)
            ori_hidden_states = hidden_states.clone()

            # Process encoder_hidden_states
            encoder_hidden_states = module.context_embedder(encoder_hidden_states)

            # Process txt_ids and img_ids
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = module.pos_embed(ids)

            # Process through transformer blocks
            for block in module.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=kwargs.get("joint_attention_kwargs"),
                )

            # Process through single transformer blocks
            # Note: single blocks concatenate internally, so pass separately
            for block in module.single_transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=kwargs.get("joint_attention_kwargs"),
                )

            # Cache the residual
            state.previous_residual = hidden_states - ori_hidden_states

        state.previous_modulated_input = modulated_inp
        state.cnt += 1

        # Apply final norm and projection (always needed)
        hidden_states = module.norm_out(hidden_states, temb)
        output = module.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def _handle_mochi_forward(
        self,
        module,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        attention_kwargs=None,
        return_dict=True,
    ):
        """
        Handle Mochi transformer forward pass with TeaCache.

        Args:
            module: The MochiTransformer3DModel instance.
            hidden_states (`torch.Tensor`): Input latent tensor.
            encoder_hidden_states (`torch.Tensor`): Text encoder outputs.
            timestep (`torch.Tensor`): Current diffusion timestep.
            encoder_attention_mask (`torch.Tensor`): Attention mask for encoder.
            attention_kwargs (`dict`, optional): Additional attention arguments.
            return_dict (`bool`): Whether to return a dict.

        Returns:
            `torch.Tensor` or `Transformer2DModelOutput`: Denoised output.
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(module, lora_scale)

        state = self.state_manager.get_state()
        self._maybe_reset_state_for_new_inference(state, module)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = module.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        # Process time embedding
        temb, encoder_hidden_states = module.time_embed(
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_dtype=hidden_states.dtype,
        )

        # Process patch embedding
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = module.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        # Get rotary embeddings
        image_rotary_emb = module.rope(
            module.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            device=hidden_states.device,
            dtype=torch.float32,
        )

        # Extract modulated input (extractors don't modify inputs)
        modulated_inp = self.extractor_fn(module, hidden_states, temb)

        # Make caching decision
        should_calc = self._should_compute_full_transformer(state, modulated_inp)

        if not should_calc:
            # Fast path: apply cached residual (already includes norm_out)
            hidden_states = hidden_states + state.previous_residual
        else:
            # Slow path: full computation
            ori_hidden_states = hidden_states.clone()

            # Process through transformer blocks
            for block in module.transformer_blocks:
                if torch.is_grad_enabled() and module.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = module._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        encoder_attention_mask,
                        image_rotary_emb,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        encoder_attention_mask=encoder_attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )

            # Apply norm_out before caching residual (matches reference implementation)
            hidden_states = module.norm_out(hidden_states, temb)
            
            # Cache the residual (includes norm_out transformation)
            state.previous_residual = hidden_states - ori_hidden_states

        state.previous_modulated_input = modulated_inp
        state.cnt += 1

        # Apply projection
        hidden_states = module.proj_out(hidden_states)

        # Reshape output
        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(module, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def _handle_lumina2_forward(
        self,
        module,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        attention_kwargs=None,
        return_dict=True,
    ):
        """
        Handle Lumina2 transformer forward pass with TeaCache.

        Note: Lumina2 has complex preprocessing and uses 'layers' instead of 'transformer_blocks'.
        The modulated input extraction happens after preprocessing to input_to_main_loop.

        Args:
            module: The Lumina2Transformer2DModel instance.
            hidden_states (`torch.Tensor`): Input latent tensor.
            timestep (`torch.Tensor`): Current diffusion timestep.
            encoder_hidden_states (`torch.Tensor`): Text encoder outputs.
            encoder_attention_mask (`torch.Tensor`): Attention mask for encoder.
            attention_kwargs (`dict`, optional): Additional attention arguments.
            return_dict (`bool`): Whether to return a dict.

        Returns:
            `torch.Tensor` or `Transformer2DModelOutput`: Denoised output.
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(module, lora_scale)

        state = self.state_manager.get_state()
        self._maybe_reset_state_for_new_inference(state, module)

        batch_size, _, height, width = hidden_states.shape

        # Lumina2 preprocessing (matches original forward)
        temb, encoder_hidden_states_processed = module.time_caption_embed(
            hidden_states, timestep, encoder_hidden_states
        )
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

        # Extract modulated input (after preprocessing)
        modulated_inp = self.extractor_fn(module, input_to_main_loop, temb)

        # Per-sequence-length cache for Lumina2 (handles variable sequence lengths)
        cache_key = max_seq_len
        if cache_key not in state.cache_dict:
            state.cache_dict[cache_key] = {
                "previous_modulated_input": None,
                "previous_residual": None,
                "accumulated_rel_l1_distance": 0.0,
            }
        current_cache = state.cache_dict[cache_key]

        # Make caching decision using per-cache values
        if state.cnt == 0 or state.cnt == state.num_steps - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache["previous_modulated_input"] is not None:
                prev_mod_input = current_cache["previous_modulated_input"]
                prev_mean = prev_mod_input.abs().mean()
                
                if prev_mean.item() > 1e-9:
                    rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item()
                else:
                    rel_l1_change = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float('inf')
                
                rescaled_distance = self.rescale_func(rel_l1_change)
                current_cache["accumulated_rel_l1_distance"] += rescaled_distance
                
                if current_cache["accumulated_rel_l1_distance"] < self.config.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    current_cache["accumulated_rel_l1_distance"] = 0.0
            else:
                should_calc = True
                current_cache["accumulated_rel_l1_distance"] = 0.0

        current_cache["previous_modulated_input"] = modulated_inp.clone()

        # Track unconditional sequence length for counter management
        if state.uncond_seq_len is None:
            state.uncond_seq_len = cache_key
        # Only increment counter when not processing unconditional (different seq len)
        if cache_key != state.uncond_seq_len:
            state.cnt += 1
            if state.cnt >= state.num_steps:
                state.cnt = 0

        # Fast or slow path with per-cache residual
        if not should_calc and current_cache["previous_residual"] is not None:
            # Fast path: apply cached residual
            processed_hidden_states = input_to_main_loop + current_cache["previous_residual"]
        else:
            # Slow path: full computation
            current_processing_states = input_to_main_loop
            for layer in module.layers:
                current_processing_states = layer(
                    current_processing_states, attention_mask_for_main_loop_arg, joint_rotary_emb, temb
                )
            processed_hidden_states = current_processing_states
            # Cache the residual in per-cache storage
            current_cache["previous_residual"] = processed_hidden_states - input_to_main_loop

        # Apply final norm and reshape
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

    def _handle_cogvideox_forward(
        self,
        module,
        hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_cond=None,
        ofs=None,
        image_rotary_emb=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        """
        Handle CogVideoX transformer forward pass with TeaCache.

        Note: CogVideoX uses timestep embedding directly (not from a block) and caches
        both encoder_hidden_states and hidden_states residuals.

        Args:
            module: The CogVideoXTransformer3DModel instance.
            hidden_states (`torch.Tensor`): Input latent tensor.
            encoder_hidden_states (`torch.Tensor`): Text encoder outputs.
            timestep (`torch.Tensor`): Current diffusion timestep.
            timestep_cond (`torch.Tensor`, optional): Additional timestep conditioning.
            ofs (`torch.Tensor`, optional): Offset tensor.
            image_rotary_emb (`torch.Tensor`, optional): Rotary embeddings.
            attention_kwargs (`dict`, optional): Additional attention arguments.
            return_dict (`bool`): Whether to return a dict.

        Returns:
            `torch.Tensor` or `Transformer2DModelOutput`: Denoised output.
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(module, lora_scale)

        state = self.state_manager.get_state()
        self._maybe_reset_state_for_new_inference(state, module, reset_encoder_residual=True)

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # Process time embedding
        timesteps = timestep
        t_emb = module.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = module.time_embedding(t_emb, timestep_cond)

        if module.ofs_embedding is not None:
            ofs_emb = module.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = module.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # Process patch embedding
        hidden_states = module.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = module.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # Extract modulated input (CogVideoX uses timestep embedding directly)
        modulated_inp = self.extractor_fn(module, hidden_states, emb)

        # Make caching decision
        should_calc = self._should_compute_full_transformer(state, modulated_inp)

        # Fast or slow path based on caching decision
        if not should_calc:
            # Fast path: apply cached residuals (both encoder and hidden_states)
            hidden_states = hidden_states + state.previous_residual
            encoder_hidden_states = encoder_hidden_states + state.previous_residual_encoder
        else:
            # Slow path: full computation
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()

            # Process through transformer blocks
            for block in module.transformer_blocks:
                if torch.is_grad_enabled() and module.gradient_checkpointing:
                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        lambda *args: block(*args),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

            # Cache both residuals
            state.previous_residual = hidden_states - ori_hidden_states
            state.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states

        state.previous_modulated_input = modulated_inp
        state.cnt += 1

        # Apply final norm
        if not module.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = module.norm_final(hidden_states)
        else:
            # CogVideoX-5B and CogVideoX1.5-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = module.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # Final block
        hidden_states = module.norm_out(hidden_states, temb=emb)
        hidden_states = module.proj_out(hidden_states)

        # Unpatchify
        p = module.config.patch_size
        p_t = module.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(module, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @torch.compiler.disable
    def _should_compute_full_transformer(self, state, modulated_inp):
        """
        Determine whether to compute full transformer blocks or reuse cached residual.

        This method implements the core caching decision logic from the TeaCache paper:
        - Always compute first and last timesteps (for maximum quality)
        - For intermediate timesteps, compute relative L1 distance between current and previous modulated inputs
        - Apply polynomial rescaling to convert distance to model-specific caching signal
        - Accumulate rescaled distances and compare to threshold
        - Return True (compute) if accumulated distance exceeds threshold, False (cache) otherwise

        Args:
            state (`TeaCacheState`): Current state containing counters and cached values.
            modulated_inp (`torch.Tensor`): Modulated input extracted using configured extractor function.

        Returns:
            `bool`: True to compute full transformer, False to reuse cached residual.
        """
        # Compute first timestep
        if state.cnt == 0:
            state.accumulated_rel_l1_distance = 0
            return True

        # compute last timestep (if num_steps is set)
        if state.num_steps > 0 and state.cnt == state.num_steps - 1:
            state.accumulated_rel_l1_distance = 0
            return True

        # Need previous modulated input for comparison
        if state.previous_modulated_input is None:
            return True

        # Compute relative L1 distance
        rel_distance = (
            (
                (modulated_inp - state.previous_modulated_input).abs().mean()
                / state.previous_modulated_input.abs().mean()
            )
            .cpu()
            .item()
        )

        # Apply polynomial rescaling
        rescaled_distance = self.rescale_func(rel_distance)
        state.accumulated_rel_l1_distance += rescaled_distance

        # Debug logging (uncomment to debug)
        # logger.warning(f"Step {state.cnt}: rel_l1={rel_distance:.6f}, rescaled={rescaled_distance:.6f}, accumulated={state.accumulated_rel_l1_distance:.6f}, thresh={self.config.rel_l1_thresh}")

        # Make decision based on accumulated threshold
        if state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False
        else:
            state.accumulated_rel_l1_distance = 0  # Reset accumulator
            return True

    def reset_state(self, module):
        self.state_manager.reset()
        return module


def apply_teacache(module, config: TeaCacheConfig):
    """
    Apply TeaCache optimization to a transformer model.

    This function registers a TeaCacheHook on the provided transformer, enabling adaptive caching of transformer block
    computations based on timestep embedding similarity. The hook intercepts the forward pass and implements the
    TeaCache algorithm to achieve 1.5x-2x speedup with minimal quality loss.

    Args:
        module: The transformer model to optimize (e.g., FluxTransformer2DModel, CogVideoXTransformer3DModel).
        config (`TeaCacheConfig`): Configuration specifying caching threshold and optional callbacks.

    Examples:
        ```python
        from diffusers import FluxPipeline
        from diffusers.hooks import TeaCacheConfig, apply_teacache

        # Load FLUX pipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        # Apply TeaCache directly to transformer
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        apply_teacache(pipe.transformer, config)

        # Generate with caching enabled
        image = pipe("A cat on a windowsill", num_inference_steps=4).images[0]

        # Or use the convenience method via CacheMixin
        pipe.transformer.enable_teacache(rel_l1_thresh=0.2)
        ```

    Note:
        For most use cases, it's recommended to use the CacheMixin interface: `pipe.transformer.enable_teacache(...)`
        which provides additional convenience methods like `disable_cache()` for easy toggling.
    """
    # Register hook on main transformer
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TeaCacheHook(config)
    registry.register_hook(hook, _TEACACHE_HOOK)
