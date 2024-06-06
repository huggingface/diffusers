# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Tuple
import torch

from ...models.attention_processor import (
    AttnProcessor2_0,
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
)

from ...utils import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class PAGMixin:
    r"""Mixin class for PAG."""
    
    @staticmethod
    def _check_input_pag_applied_layer(layer):
        layer_splits = layer.split(".")
        if len(layer_splits) > 3:
            raise ValueError(f"pag layer should only contains block_type, block_index and attention_index{layer}.")
        
        if len(layer_splits) >= 1:
            if layer_splits[0] not in ["down", "mid", "up"]:
                raise ValueError(f"Invalid block_type in pag layer {layer}. Accept 'down', 'mid', 'up', got {layer_splits[0]}")
        
        if len(layer_splits) >= 2:
            if not layer_splits[1].startswith("block_"):
                raise ValueError(f"Invalid block_index in pag layer: {layer}. Should start with 'block_'")
        
        if len(layer_splits) == 3:
            if not layer_splits[2].startswith("attentions_"):
                raise ValueError(f"Invalid attention_index in pag layer: {layer}. Should start with 'attentions_'")


    def _set_attn_processor_pag_applied_layers(self, replace_processor):
        
        def is_self_attn(name):
            return "attn1" in name and "to" not in name

        def get_block_type(name): 
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "down"
            return name.split(".")[0].split("_")[0]
        
        def get_block_index(name):
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "blocks_1"
            return f"block_{name.split('.')[1]}"
        
        def get_attn_index(name):
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "attentions_0"
            return f"attentions_{name.split('.')[3]}"
        
        
        for drop_layer in self.pag_applied_layers:

            self._check_input_pag_applied_layer(drop_layer)
            drop_layer_splits = drop_layer.split(".")

            if len(drop_layer_splits) == 1:
                # e.g. "mid"
                block_type = drop_layer_splits[0]
                target_modules = []
                for name, module in self.unet.named_modules():
                    if not is_self_attn(name):
                        continue
                    if get_block_type(name) == block_type:
                        target_modules.append(module)
             
            elif len(drop_layer_splits) == 2:
                # e.g. "down.block_1"
                block_type = drop_layer_splits[0]
                block_index = drop_layer_splits[1]
                target_modules = []
                for name, module in self.unet.named_modules():
                    if not is_self_attn(name):
                        continue
                    if get_block_type(name) == block_type and get_block_index(name) == block_index:
                        target_modules.append(module)

            elif len(drop_layer_splits) == 3:
                # e.g. "down.blocks_1.attentions_1"
                block_type = drop_layer_splits[0]
                block_index = drop_layer_splits[1]
                attn_index = drop_layer_splits[2]
                target_modules = []
                for name, module in self.unet.named_modules():
                    if not is_self_attn(name):
                        continue
                    if get_block_type(name) == block_type and get_block_index(name) == block_index and get_attn_index(name) == attn_index:
                        target_modules.append(module)
            
            if len(target_modules) == 0:
                logger.warning(f"Cannot find pag layer to set attention processor: {drop_layer}")
            
            for module in target_modules:
                module.processor = replace_processor

    def _set_pag_attn_processor(self, do_classifier_free_guidance):
        if do_classifier_free_guidance:
            self._set_attn_processor_pag_applied_layers(PAGCFGIdentitySelfAttnProcessor2_0())
        else:
            self._set_attn_processor_pag_applied_layers(PAGIdentitySelfAttnProcessor2_0())

    def _reset_attn_processor(self):
        self._set_attn_processor_pag_applied_layers(AttnProcessor2_0())

    def _get_pag_scale(self, t):
        if self.do_pag_adaptive_scaling:
            signal_scale = self.pag_scale - self.pag_adaptive_scale * (1000 - t)
            if signal_scale < 0:
                signal_scale = 0
            return signal_scale
        else:
            return self.pag_scale

    def _apply_perturbed_attention_guidance(self, noise_pred, do_classifier_free_guidance, guidance_scale, t):
        pag_scale = self._get_pag_scale(t)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
                + pag_scale * (noise_pred_text - noise_pred_perturb)
            )
        else:
            noise_pred_uncond, noise_pred_perturb = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + pag_scale * (noise_pred_uncond - noise_pred_perturb)
        return noise_pred
    
    def _prepare_perturbed_attention_guidance(self, input, uncond_input, do_classifier_free_guidance):
        input = torch.cat([input] * 2, dim=0)

        if do_classifier_free_guidance:
            input = torch.cat([uncond_input, input], dim=0)
        return input

    def _reset_attn_processor(self):
        self._set_attn_processor_pag_applied_layers(AttnProcessor2_0())
    
    @property
    def pag_scale(self):
        return self._pag_scale
    
    @property
    def pag_adaptive_scale(self):
        return self._pag_adaptive_scale

    @property
    def do_pag_adaptive_scaling(self):
        return self._is_pag_enabled and self._pag_adaptive_scale > 0

    @property
    def do_perturbed_attention_guidance(self):
        return self._is_pag_enabled and self._pag_scale > 0
