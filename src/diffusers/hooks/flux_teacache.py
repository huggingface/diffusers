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

import numpy as np
import torch

from .hooks import BaseState, HookRegistry, ModelHook, StateManager


@dataclass
class FluxTeaCacheConfig:
    """Configuration for FLUX TeaCache following original algorithm."""
    rel_l1_thresh: float = 0.2  # threshold for accumulated distance (based on paper 0.1->0.3 works best)
    coefficients: Optional[List[float]] = None  # FLUX-specific polynomial coefficients  
    current_timestep_callback: Optional[Callable[[], int]] = None
    
    def __post_init__(self):
        if self.coefficients is None:
            # original FLUX coefficients from TeaCache paper
            self.coefficients = [4.98651651e+02, -2.83781631e+02, 
                               5.58554382e+01, -3.82021401e+00, 2.64230861e-01]


class FluxTeaCacheState(BaseState):
    """State management following original TeaCache implementation."""
    def __init__(self):
        self.cnt = 0  # Current timestep counter
        self.num_steps = 0  # Total inference steps
        self.accumulated_rel_l1_distance = 0.0  # Running accumulator
        self.previous_modulated_input = None  # Previous timestep modulated features
        self.previous_residual = None  # cached transformer residual
        
    def reset(self):
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None  
        self.previous_residual = None


class FluxTeaCacheHook(ModelHook):
    """Main hook implementing FLUX TeaCache logic."""
    
    _is_stateful = True
    
    def __init__(self, config: FluxTeaCacheConfig):
        super().__init__()
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state_manager = StateManager(FluxTeaCacheState, (), {})
        
    def initialize_hook(self, module):
        self.state_manager.set_context("flux_teacache")
        return module
        
    def new_forward(self, module, hidden_states, timestep, pooled_projections, 
                   encoder_hidden_states, txt_ids, img_ids, **kwargs):
        """Replace FLUX transformer forward with TeaCache logic."""
        state = self.state_manager.get_state()
        
        # Extract timestep embedding for FLUX
        timestep = timestep.to(hidden_states.dtype) * 1000
        temb = module.time_text_embed(timestep, pooled_projections)
        
        # Extract modulated input from first transformer block
        modulated_inp, _, _, _, _ = module.transformer_blocks[0].norm1(
            hidden_states.clone(), emb=temb.clone()
        )
        
        # Make caching decision using original algorithm
        should_calc = self._should_compute_full_transformer(state, modulated_inp)
        
        if not should_calc:
            # Fast path: apply cached residual
            output = hidden_states + state.previous_residual
        else:
            # Slow path: full computation
            ori_hidden_states = hidden_states.clone()
            output = self.fn_ref.original_forward(
                hidden_states, timestep, pooled_projections, 
                encoder_hidden_states, txt_ids, img_ids, **kwargs
            )
            # Cache the residual
            state.previous_residual = output - ori_hidden_states
            state.previous_modulated_input = modulated_inp
            
        state.cnt += 1
        return output
        
    def _should_compute_full_transformer(self, state, modulated_inp):
        """Core caching decision logic from original TeaCache."""
        # Compute first and last timesteps (always compute)
        if state.cnt == 0 or state.cnt == state.num_steps - 1:
            state.accumulated_rel_l1_distance = 0
            return True
            
        # Compute relative L1 distance 
        rel_distance = ((modulated_inp - state.previous_modulated_input).abs().mean() 
                       / state.previous_modulated_input.abs().mean()).cpu().item()
        
        # Apply polynomial rescaling
        rescaled_distance = self.rescale_func(rel_distance)
        state.accumulated_rel_l1_distance += rescaled_distance
        
        # Make decision based on accumulated threshold
        if state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False  
        else:
            state.accumulated_rel_l1_distance = 0  # Reset accumulator
            return True   
    
    def reset_state(self, module):
        self.state_manager.reset()
        return module


def apply_flux_teacache(module, config: FluxTeaCacheConfig):
    """Apply TeaCache to FLUX transformer following diffusers patterns."""
    from ..models.transformers.transformer_flux import FluxTransformer2DModel
    
    # Validate FLUX model
    if not isinstance(module, FluxTransformer2DModel):
        raise ValueError("TeaCache supports only FLUX transformer model for now")
        
    # Register hook on main transformer
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = FluxTeaCacheHook(config)
    registry.register_hook(hook, "flux_teacache")
