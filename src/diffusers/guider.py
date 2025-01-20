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

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .models.attention_processor import (
    Attention,
    AttentionProcessor,
    PAGCFGIdentitySelfAttnProcessor2_0,
    PAGIdentitySelfAttnProcessor2_0,
)
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class CFGGuider:
    """
    This class is used to guide the pipeline with CFG (Classifier-Free Guidance).
    """

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0 and not self._disable_guidance

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def batch_size(self):
        return self._batch_size

    def set_guider(self, pipeline, guider_kwargs: Dict[str, Any]):
        # a flag to disable CFG, e.g. we disable it for LCM and use a guidance scale embedding instead
        disable_guidance = guider_kwargs.get("disable_guidance", False)
        guidance_scale = guider_kwargs.get("guidance_scale", None)
        if guidance_scale is None:
            raise ValueError("guidance_scale is not provided in guider_kwargs")
        guidance_rescale = guider_kwargs.get("guidance_rescale", 0.0)
        batch_size = guider_kwargs.get("batch_size", None)
        if batch_size is None:
            raise ValueError("batch_size is not provided in guider_kwargs")
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._batch_size = batch_size
        self._disable_guidance = disable_guidance

    def reset_guider(self, pipeline):
        pass

    def maybe_update_guider(self, pipeline, timestep):
        pass

    def maybe_update_input(self, pipeline, cond_input):
        pass

    def _maybe_split_prepared_input(self, cond):
        """
        Process and potentially split the conditional input for Classifier-Free Guidance (CFG).

        This method handles inputs that may already have CFG applied (i.e. when `cond` is output of `prepare_input`).
        It determines whether to split the input based on its batch size relative to the expected batch size.

        Args:
            cond (torch.Tensor): The conditional input tensor to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The negative conditional input (uncond_input)
                - The positive conditional input (cond_input)
        """
        if cond.shape[0] == self.batch_size * 2:
            neg_cond = cond[0 : self.batch_size]
            cond = cond[self.batch_size :]
            return neg_cond, cond
        elif cond.shape[0] == self.batch_size:
            return cond, cond
        else:
            raise ValueError(f"Unsupported input shape: {cond.shape}")

    def _is_prepared_input(self, cond):
        """
        Check if the input is already prepared for Classifier-Free Guidance (CFG).

        Args:
            cond (torch.Tensor): The conditional input tensor to check.

        Returns:
            bool: True if the input is already prepared, False otherwise.
        """
        cond_tensor = cond[0] if isinstance(cond, (list, tuple)) else cond

        return cond_tensor.shape[0] == self.batch_size * 2

    def prepare_input(
        self,
        cond_input: Union[torch.Tensor, List[torch.Tensor]],
        negative_cond_input: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Prepare the input for CFG.

        Args:
            cond_input (Union[torch.Tensor, List[torch.Tensor]]):
                The conditional input. It can be a single tensor or a
            list of tensors. It must have the same length as `negative_cond_input`.
            negative_cond_input (Union[torch.Tensor, List[torch.Tensor]]): The negative conditional input. It can be a
                single tensor or a list of tensors. It must have the same length as `cond_input`.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The prepared input.
        """

        # we check if cond_input already has CFG applied, and split if it is the case.
        if self._is_prepared_input(cond_input) and self.do_classifier_free_guidance:
            return cond_input

        if self._is_prepared_input(cond_input) and not self.do_classifier_free_guidance:
            if isinstance(cond_input, list):
                negative_cond_input, cond_input = zip(*[self._maybe_split_prepared_input(cond) for cond in cond_input])
            else:
                negative_cond_input, cond_input = self._maybe_split_prepared_input(cond_input)

        if not self._is_prepared_input(cond_input) and self.do_classifier_free_guidance and negative_cond_input is None:
            raise ValueError(
                "`negative_cond_input` is required when cond_input does not already contains negative conditional input"
            )

        if isinstance(cond_input, (list, tuple)):
            if not self.do_classifier_free_guidance:
                return cond_input

            if len(negative_cond_input) != len(cond_input):
                raise ValueError("The length of negative_cond_input and cond_input must be the same.")
            prepared_input = []
            for neg_cond, cond in zip(negative_cond_input, cond_input):
                if neg_cond.shape[0] != cond.shape[0]:
                    raise ValueError("The batch size of negative_cond_input and cond_input must be the same.")
                prepared_input.append(torch.cat([neg_cond, cond], dim=0))
            return prepared_input

        elif isinstance(cond_input, torch.Tensor):
            if not self.do_classifier_free_guidance:
                return cond_input
            else:
                return torch.cat([negative_cond_input, cond_input], dim=0)

        else:
            raise ValueError(f"Unsupported input type: {type(cond_input)}")

    def apply_guidance(
        self,
        model_output: torch.Tensor,
        timestep: int = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.do_classifier_free_guidance:
            return model_output

        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
        return noise_pred


class PAGGuider:
    """
    This class is used to guide the pipeline with CFG (Classifier-Free Guidance).
    """

    def __init__(
        self,
        pag_applied_layers: Union[str, List[str]],
        pag_attn_processors: Tuple[AttentionProcessor, AttentionProcessor] = (
            PAGCFGIdentitySelfAttnProcessor2_0(),
            PAGIdentitySelfAttnProcessor2_0(),
        ),
    ):
        r"""
        Set the the self-attention layers to apply PAG. Raise ValueError if the input is invalid.

        Args:
            pag_applied_layers (`str` or `List[str]`):
                One or more strings identifying the layer names, or a simple regex for matching multiple layers, where
                PAG is to be applied. A few ways of expected usage are as follows:
                  - Single layers specified as - "blocks.{layer_index}"
                  - Multiple layers as a list - ["blocks.{layers_index_1}", "blocks.{layer_index_2}", ...]
                  - Multiple layers as a block name - "mid"
                  - Multiple layers as regex - "blocks.({layer_index_1}|{layer_index_2})"
            pag_attn_processors:
                (`Tuple[AttentionProcessor, AttentionProcessor]`, defaults to `(PAGCFGIdentitySelfAttnProcessor2_0(),
                PAGIdentitySelfAttnProcessor2_0())`): A tuple of two attention processors. The first attention
                processor is for PAG with Classifier-free guidance enabled (conditional and unconditional). The second
                attention processor is for PAG with CFG disabled (unconditional only).
        """

        if not isinstance(pag_applied_layers, list):
            pag_applied_layers = [pag_applied_layers]
        if pag_attn_processors is not None:
            if not isinstance(pag_attn_processors, tuple) or len(pag_attn_processors) != 2:
                raise ValueError("Expected a tuple of two attention processors")

        for i in range(len(pag_applied_layers)):
            if not isinstance(pag_applied_layers[i], str):
                raise ValueError(
                    f"Expected either a string or a list of string but got type {type(pag_applied_layers[i])}"
                )

        self.pag_applied_layers = pag_applied_layers
        self._pag_attn_processors = pag_attn_processors

    def _set_pag_attn_processor(self, model, pag_applied_layers, do_classifier_free_guidance):
        r"""
        Set the attention processor for the PAG layers.
        """
        pag_attn_processors = self._pag_attn_processors
        pag_attn_proc = pag_attn_processors[0] if do_classifier_free_guidance else pag_attn_processors[1]

        def is_self_attn(module: nn.Module) -> bool:
            r"""
            Check if the module is self-attention module based on its name.
            """
            return isinstance(module, Attention) and not module.is_cross_attention

        def is_fake_integral_match(layer_id, name):
            layer_id = layer_id.split(".")[-1]
            name = name.split(".")[-1]
            return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        for layer_id in pag_applied_layers:
            # for each PAG layer input, we find corresponding self-attention layers in the unet model
            target_modules = []

            for name, module in model.named_modules():
                # Identify the following simple cases:
                #   (1) Self Attention layer existing
                #   (2) Whether the module name matches pag layer id even partially
                #   (3) Make sure it's not a fake integral match if the layer_id ends with a number
                #       For example, blocks.1, blocks.10 should be differentiable if layer_id="blocks.1"
                if (
                    is_self_attn(module)
                    and re.search(layer_id, name) is not None
                    and not is_fake_integral_match(layer_id, name)
                ):
                    logger.debug(f"Applying PAG to layer: {name}")
                    target_modules.append(module)

            if len(target_modules) == 0:
                raise ValueError(f"Cannot find PAG layer to set attention processor for: {layer_id}")

            for module in target_modules:
                module.processor = pag_attn_proc

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and not self._disable_guidance

    @property
    def do_perturbed_attention_guidance(self):
        return self._pag_scale > 0 and not self._disable_guidance

    @property
    def do_pag_adaptive_scaling(self):
        return self._pag_adaptive_scale > 0 and self._pag_scale > 0 and not self._disable_guidance

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def pag_scale(self):
        return self._pag_scale

    @property
    def pag_adaptive_scale(self):
        return self._pag_adaptive_scale

    def set_guider(self, pipeline, guider_kwargs: Dict[str, Any]):
        pag_scale = guider_kwargs.get("pag_scale", 3.0)
        pag_adaptive_scale = guider_kwargs.get("pag_adaptive_scale", 0.0)

        batch_size = guider_kwargs.get("batch_size", None)
        if batch_size is None:
            raise ValueError("batch_size is a required argument for PAGGuider")

        guidance_scale = guider_kwargs.get("guidance_scale", None)
        guidance_rescale = guider_kwargs.get("guidance_rescale", 0.0)
        disable_guidance = guider_kwargs.get("disable_guidance", False)

        if guidance_scale is None:
            raise ValueError("guidance_scale is a required argument for PAGGuider")

        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale
        self._guidance_scale = guidance_scale
        self._disable_guidance = disable_guidance
        self._guidance_rescale = guidance_rescale
        self._batch_size = batch_size
        if not hasattr(pipeline, "original_attn_proc") or pipeline.original_attn_proc is None:
            pipeline.original_attn_proc = pipeline.unet.attn_processors
            self._set_pag_attn_processor(
                model=pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer,
                pag_applied_layers=self.pag_applied_layers,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

    def reset_guider(self, pipeline):
        if (
            self.do_perturbed_attention_guidance
            and hasattr(pipeline, "original_attn_proc")
            and pipeline.original_attn_proc is not None
        ):
            pipeline.unet.set_attn_processor(pipeline.original_attn_proc)
            pipeline.original_attn_proc = None

    def maybe_update_guider(self, pipeline, timestep):
        pass

    def maybe_update_input(self, pipeline, cond_input):
        pass

    def _is_prepared_input(self, cond):
        """
        Check if the input is already prepared for Perturbed Attention Guidance (PAG).

        Args:
            cond (torch.Tensor): The conditional input tensor to check.

        Returns:
            bool: True if the input is already prepared, False otherwise.
        """
        cond_tensor = cond[0] if isinstance(cond, (list, tuple)) else cond

        return cond_tensor.shape[0] == self.batch_size * 3

    def _maybe_split_prepared_input(self, cond):
        """
        Process and potentially split the conditional input for Classifier-Free Guidance (CFG).

        This method handles inputs that may already have CFG applied (i.e. when `cond` is output of `prepare_input`).
        It determines whether to split the input based on its batch size relative to the expected batch size.

        Args:
            cond (torch.Tensor): The conditional input tensor to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The negative conditional input (uncond_input)
                - The positive conditional input (cond_input)
        """
        if cond.shape[0] == self.batch_size * 3:
            neg_cond = cond[0 : self.batch_size]
            cond = cond[self.batch_size : self.batch_size * 2]
            return neg_cond, cond
        elif cond.shape[0] == self.batch_size:
            return cond, cond
        else:
            raise ValueError(f"Unsupported input shape: {cond.shape}")

    def prepare_input(
        self,
        cond_input: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        negative_cond_input: Optional[Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Prepare the input for CFG.

        Args:
            cond_input (Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
                The conditional input. It can be a single tensor or a
            list of tensors. It must have the same length as `negative_cond_input`.
            negative_cond_input (Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
                The negative conditional input. It can be a single tensor or a list of tensors. It must have the same
                length as `cond_input`.

        Returns:
            Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]: The prepared input.
        """

        # we check if cond_input already has CFG applied, and split if it is the case.

        if self._is_prepared_input(cond_input) and self.do_perturbed_attention_guidance:
            return cond_input

        if self._is_prepared_input(cond_input) and not self.do_perturbed_attention_guidance:
            if isinstance(cond_input, list):
                negative_cond_input, cond_input = zip(*[self._maybe_split_prepared_input(cond) for cond in cond_input])
            else:
                negative_cond_input, cond_input = self._maybe_split_prepared_input(cond_input)

        if not self._is_prepared_input(cond_input) and self.do_perturbed_attention_guidance and negative_cond_input is None:
            raise ValueError(
                "`negative_cond_input` is required when cond_input does not already contains negative conditional input"
            )

        if isinstance(cond_input, (list, tuple)):
            if not self.do_perturbed_attention_guidance:
                return cond_input

            if len(negative_cond_input) != len(cond_input):
                raise ValueError("The length of negative_cond_input and cond_input must be the same.")

            prepared_input = []
            for neg_cond, cond in zip(negative_cond_input, cond_input):
                if neg_cond.shape[0] != cond.shape[0]:
                    raise ValueError("The batch size of negative_cond_input and cond_input must be the same.")

                cond = torch.cat([cond] * 2, dim=0)
                if self.do_classifier_free_guidance:
                    prepared_input.append(torch.cat([neg_cond, cond], dim=0))
                else:
                    prepared_input.append(cond)

            return prepared_input

        elif isinstance(cond_input, torch.Tensor):
            if not self.do_perturbed_attention_guidance:
                return cond_input

            cond_input = torch.cat([cond_input] * 2, dim=0)
            if self.do_classifier_free_guidance:
                return torch.cat([negative_cond_input, cond_input], dim=0)
            else:
                return cond_input

        else:
            raise ValueError(f"Unsupported input type: {type(negative_cond_input)} and {type(cond_input)}")

    def apply_guidance(
        self,
        model_output: torch.Tensor,
        timestep: int,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.do_perturbed_attention_guidance:
            return model_output

        if self.do_pag_adaptive_scaling:
            pag_scale = max(self._pag_scale - self._pag_adaptive_scale * (1000 - timestep), 0)
        else:
            pag_scale = self._pag_scale

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text, noise_pred_perturb = model_output.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                + pag_scale * (noise_pred_text - noise_pred_perturb)
            )
        else:
            noise_pred_text, noise_pred_perturb = model_output.chunk(2)
            noise_pred = noise_pred_text + pag_scale * (noise_pred_text - noise_pred_perturb)

        if self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

        return noise_pred


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


class APGGuider:
    """
    This class is used to guide the pipeline with APG (Adaptive Projected Guidance).
    """

    def normalized_guidance(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: torch.Tensor,
        guidance_scale: float,
        momentum_buffer: MomentumBuffer = None,
        norm_threshold: float = 0.0,
        eta: float = 1.0,
    ):
        """
        Based on the findings of [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion
        Models](https://arxiv.org/pdf/2410.02416)
        """
        diff = pred_cond - pred_uncond
        if momentum_buffer is not None:
            momentum_buffer.update(diff)
            diff = momentum_buffer.running_average
        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor
        v0, v1 = diff.double(), pred_cond.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        diff_parallel, diff_orthogonal = v0_parallel.to(diff.dtype), v0_orthogonal.to(diff.dtype)
        normalized_update = diff_orthogonal + eta * diff_parallel
        pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
        return pred_guided

    @property
    def adaptive_projected_guidance_momentum(self):
        return self._adaptive_projected_guidance_momentum

    @property
    def adaptive_projected_guidance_rescale_factor(self):
        return self._adaptive_projected_guidance_rescale_factor

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0 and not self._disable_guidance

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def batch_size(self):
        return self._batch_size

    def set_guider(self, pipeline, guider_kwargs: Dict[str, Any]):
        disable_guidance = guider_kwargs.get("disable_guidance", False)
        guidance_scale = guider_kwargs.get("guidance_scale", None)
        if guidance_scale is None:
            raise ValueError("guidance_scale is not provided in guider_kwargs")
        adaptive_projected_guidance_momentum = guider_kwargs.get("adaptive_projected_guidance_momentum", None)
        adaptive_projected_guidance_rescale_factor = guider_kwargs.get(
            "adaptive_projected_guidance_rescale_factor", 15.0
        )
        guidance_rescale = guider_kwargs.get("guidance_rescale", 0.0)
        batch_size = guider_kwargs.get("batch_size", None)
        if batch_size is None:
            raise ValueError("batch_size is not provided in guider_kwargs")
        self._adaptive_projected_guidance_momentum = adaptive_projected_guidance_momentum
        self._adaptive_projected_guidance_rescale_factor = adaptive_projected_guidance_rescale_factor
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._batch_size = batch_size
        self._disable_guidance = disable_guidance
        if adaptive_projected_guidance_momentum is not None:
            self.momentum_buffer = MomentumBuffer(adaptive_projected_guidance_momentum)
        else:
            self.momentum_buffer = None
        self.scheduler = pipeline.scheduler

    def reset_guider(self, pipeline):
        pass

    def maybe_update_guider(self, pipeline, timestep):
        pass

    def maybe_update_input(self, pipeline, cond_input):
        pass

    def _maybe_split_prepared_input(self, cond):
        """
        Process and potentially split the conditional input for Classifier-Free Guidance (CFG).

        This method handles inputs that may already have CFG applied (i.e. when `cond` is output of `prepare_input`).
        It determines whether to split the input based on its batch size relative to the expected batch size.

        Args:
            cond (torch.Tensor): The conditional input tensor to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The negative conditional input (uncond_input)
                - The positive conditional input (cond_input)
        """
        if cond.shape[0] == self.batch_size * 2:
            neg_cond = cond[0 : self.batch_size]
            cond = cond[self.batch_size :]
            return neg_cond, cond
        elif cond.shape[0] == self.batch_size:
            return cond, cond
        else:
            raise ValueError(f"Unsupported input shape: {cond.shape}")

    def _is_prepared_input(self, cond):
        """
        Check if the input is already prepared for Classifier-Free Guidance (CFG).

        Args:
            cond (torch.Tensor): The conditional input tensor to check.

        Returns:
            bool: True if the input is already prepared, False otherwise.
        """
        cond_tensor = cond[0] if isinstance(cond, (list, tuple)) else cond

        return cond_tensor.shape[0] == self.batch_size * 2

    def prepare_input(
        self,
        cond_input: Union[torch.Tensor, List[torch.Tensor]],
        negative_cond_input: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Prepare the input for CFG.

        Args:
            cond_input (Union[torch.Tensor, List[torch.Tensor]]):
                The conditional input. It can be a single tensor or a
            list of tensors. It must have the same length as `negative_cond_input`.
            negative_cond_input (Union[torch.Tensor, List[torch.Tensor]]): The negative conditional input. It can be a
                single tensor or a list of tensors. It must have the same length as `cond_input`.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The prepared input.
        """

        # we check if cond_input already has CFG applied, and split if it is the case.
        if self._is_prepared_input(cond_input) and self.do_classifier_free_guidance:
            return cond_input

        if self._is_prepared_input(cond_input) and not self.do_classifier_free_guidance:
            if isinstance(cond_input, list):
                negative_cond_input, cond_input = zip(*[self._maybe_split_prepared_input(cond) for cond in cond_input])
            else:
                negative_cond_input, cond_input = self._maybe_split_prepared_input(cond_input)

        if not self._is_prepared_input(cond_input) and self.do_classifier_free_guidance and negative_cond_input is None:
            raise ValueError(
                "`negative_cond_input` is required when cond_input does not already contains negative conditional input"
            )

        if isinstance(cond_input, (list, tuple)):
            if not self.do_classifier_free_guidance:
                return cond_input

            if len(negative_cond_input) != len(cond_input):
                raise ValueError("The length of negative_cond_input and cond_input must be the same.")
            prepared_input = []
            for neg_cond, cond in zip(negative_cond_input, cond_input):
                if neg_cond.shape[0] != cond.shape[0]:
                    raise ValueError("The batch size of negative_cond_input and cond_input must be the same.")
                prepared_input.append(torch.cat([neg_cond, cond], dim=0))
            return prepared_input

        elif isinstance(cond_input, torch.Tensor):
            if not self.do_classifier_free_guidance:
                return cond_input
            else:
                return torch.cat([negative_cond_input, cond_input], dim=0)

        else:
            raise ValueError(f"Unsupported input type: {type(cond_input)}")

    def apply_guidance(
        self,
        model_output: torch.Tensor,
        timestep: int = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.do_classifier_free_guidance:
            return model_output

        if latents is None:
            raise ValueError("APG requires `latents` to convert model output to denoised prediction (x0).")

        sigma = self.scheduler.sigmas[self.scheduler.step_index]
        noise_pred = latents - sigma * model_output
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = self.normalized_guidance(
            noise_pred_text,
            noise_pred_uncond,
            self.guidance_scale,
            self.momentum_buffer,
            self.adaptive_projected_guidance_rescale_factor,
        )
        noise_pred = (latents - noise_pred) / sigma

        if self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
        return noise_pred
