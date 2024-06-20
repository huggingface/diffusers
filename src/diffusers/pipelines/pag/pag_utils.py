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

import torch

from ...models.attention_processor import (
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
)
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PAGMixin:
    r"""Mixin class for PAG."""

    @staticmethod
    def _check_input_pag_applied_layer(layer):
        r"""
        Check if each layer input in `applied_pag_layers` is valid. It should be either one of these 3 formats:
        "{block_type}", "{block_type}.{block_index}", or "{block_type}.{block_index}.{attention_index}". `block_type`
        can be "down", "mid", "up". `block_index` should be in the format of "block_{i}". `attention_index` should be
        in the format of "attentions_{j}".
        """

        layer_splits = layer.split(".")

        if len(layer_splits) > 3:
            raise ValueError(f"pag layer should only contains block_type, block_index and attention_index{layer}.")

        if len(layer_splits) >= 1:
            if layer_splits[0] not in ["down", "mid", "up"]:
                raise ValueError(
                    f"Invalid block_type in pag layer {layer}. Accept 'down', 'mid', 'up', got {layer_splits[0]}"
                )

        if len(layer_splits) >= 2:
            if not layer_splits[1].startswith("block_"):
                raise ValueError(f"Invalid block_index in pag layer: {layer}. Should start with 'block_'")

        if len(layer_splits) == 3:
            if not layer_splits[2].startswith("attentions_"):
                raise ValueError(f"Invalid attention_index in pag layer: {layer}. Should start with 'attentions_'")

    def _set_pag_attn_processor(self, pag_applied_layers, do_classifier_free_guidance):
        r"""
        Set the attention processor for the PAG layers.
        """
        if do_classifier_free_guidance:
            pag_attn_proc = PAGCFGIdentitySelfAttnProcessor2_0()
        else:
            pag_attn_proc = PAGIdentitySelfAttnProcessor2_0()

        def is_self_attn(module_name):
            r"""
            Check if the module is self-attention module based on its name.
            """
            return "attn1" in module_name and "to" not in name

        def get_block_type(module_name):
            r"""
            Get the block type from the module name. can be "down", "mid", "up".
            """
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "down"
            return module_name.split(".")[0].split("_")[0]

        def get_block_index(module_name):
            r"""
            Get the block index from the module name. can be "block_0", "block_1", ... If there is only one block (e.g.
            mid_block) and index is ommited from the name, it will be "block_0".
            """
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "block_1"
            # mid_block.attentions.0.transformer_blocks.0.attn1 -> "block_0"
            if "attentions" in module_name.split(".")[1]:
                return "block_0"
            else:
                return f"block_{module_name.split('.')[1]}"

        def get_attn_index(module_name):
            r"""
            Get the attention index from the module name. can be "attentions_0", "attentions_1", ...
            """
            # down_blocks.1.attentions.0.transformer_blocks.0.attn1 -> "attentions_0"
            # mid_block.attentions.0.transformer_blocks.0.attn1 -> "attentions_0"
            if "attentions" in module_name.split(".")[2]:
                return f"attentions_{module_name.split('.')[3]}"
            elif "attentions" in module_name.split(".")[1]:
                return f"attentions_{module_name.split('.')[2]}"

        for pag_layer_input in pag_applied_layers:
            # for each PAG layer input, we find corresponding self-attention layers in the unet model
            target_modules = []

            pag_layer_input_splits = pag_layer_input.split(".")

            if len(pag_layer_input_splits) == 1:
                # when the layer input only contains block_type. e.g. "mid", "down", "up"
                block_type = pag_layer_input_splits[0]
                for name, module in self.unet.named_modules():
                    if is_self_attn(name) and get_block_type(name) == block_type:
                        target_modules.append(module)

            elif len(pag_layer_input_splits) == 2:
                # when the layer inpput contains both block_type and block_index. e.g. "down.block_1", "mid.block_0"
                block_type = pag_layer_input_splits[0]
                block_index = pag_layer_input_splits[1]
                for name, module in self.unet.named_modules():
                    if (
                        is_self_attn(name)
                        and get_block_type(name) == block_type
                        and get_block_index(name) == block_index
                    ):
                        target_modules.append(module)

            elif len(pag_layer_input_splits) == 3:
                # when the layer input contains block_type, block_index and attention_index. e.g. "down.blocks_1.attentions_1"
                block_type = pag_layer_input_splits[0]
                block_index = pag_layer_input_splits[1]
                attn_index = pag_layer_input_splits[2]

                for name, module in self.unet.named_modules():
                    if (
                        is_self_attn(name)
                        and get_block_type(name) == block_type
                        and get_block_index(name) == block_index
                        and get_attn_index(name) == attn_index
                    ):
                        target_modules.append(module)

            if len(target_modules) == 0:
                raise ValueError(f"Cannot find pag layer to set attention processor for: {pag_layer_input}")

            for module in target_modules:
                module.processor = pag_attn_proc

    def _get_pag_scale(self, t):
        r"""
        Get the scale factor for the perturbed attention guidance at timestep `t`.
        """

        if self.do_pag_adaptive_scaling:
            signal_scale = self.pag_scale - self.pag_adaptive_scale * (1000 - t)
            if signal_scale < 0:
                signal_scale = 0
            return signal_scale
        else:
            return self.pag_scale

    def _apply_perturbed_attention_guidance(self, noise_pred, do_classifier_free_guidance, guidance_scale, t):
        r"""
        Apply perturbed attention guidance to the noise prediction.

        Args:
            noise_pred (torch.Tensor): The noise prediction tensor.
            do_classifier_free_guidance (bool): Whether to apply classifier-free guidance.
            guidance_scale (float): The scale factor for the guidance term.
            t (int): The current time step.

        Returns:
            torch.Tensor: The updated noise prediction tensor after applying perturbed attention guidance.
        """
        pag_scale = self._get_pag_scale(t)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_uncond)
                + pag_scale * (noise_pred_text - noise_pred_perturb)
            )
        else:
            noise_pred_text, noise_pred_perturb = noise_pred.chunk(2)
            noise_pred = noise_pred_text + pag_scale * (noise_pred_text - noise_pred_perturb)
        return noise_pred

    def _prepare_perturbed_attention_guidance(self, cond, uncond, do_classifier_free_guidance):
        """
        Prepares the perturbed attention guidance for the PAG model.

        Args:
            cond (torch.Tensor): The conditional input tensor.
            uncond (torch.Tensor): The unconditional input tensor.
            do_classifier_free_guidance (bool): Flag indicating whether to perform classifier-free guidance.

        Returns:
            torch.Tensor: The prepared perturbed attention guidance tensor.
        """

        cond = torch.cat([cond] * 2, dim=0)

        if do_classifier_free_guidance:
            cond = torch.cat([uncond, cond], dim=0)
        return cond

    def set_pag_applied_layers(self, pag_applied_layers):
        r"""
        set the the self-attention layers to apply PAG. Raise ValueError if the input is invalid.
        """

        if not isinstance(pag_applied_layers, list):
            pag_applied_layers = [pag_applied_layers]

        for pag_layer in pag_applied_layers:
            self._check_input_pag_applied_layer(pag_layer)

        self.pag_applied_layers = pag_applied_layers

    @property
    def pag_scale(self):
        """
        Get the scale factor for the perturbed attention guidance.
        """
        return self._pag_scale

    @property
    def pag_adaptive_scale(self):
        """
        Get the adaptive scale factor for the perturbed attention guidance.
        """
        return self._pag_adaptive_scale

    @property
    def do_pag_adaptive_scaling(self):
        """
        Check if the adaptive scaling is enabled for the perturbed attention guidance.
        """
        return self._pag_adaptive_scale > 0 and self._pag_scale > 0 and len(self.pag_applied_layers) > 0

    @property
    def do_perturbed_attention_guidance(self):
        """
        Check if the perturbed attention guidance is enabled.
        """
        return self._pag_scale > 0 and len(self.pag_applied_layers) > 0

    @property
    def pag_attn_processors(self):
        r"""
        Returns:
            `dict` of PAG attention processors: A dictionary contains all PAG attention processors used in the model
            with the key as the name of the layer.
        """

        processors = {}
        for name, proc in self.unet.attn_processors.items():
            if proc.__class__ in (PAGCFGIdentitySelfAttnProcessor2_0, PAGIdentitySelfAttnProcessor2_0):
                processors[name] = proc
        return processors
