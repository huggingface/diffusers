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
from typing import Optional, Union, List
import torch
from torch import nn
from torch.nn import functional as F
import inspect
import safetensors
from accelerate.utils import set_module_tensor_to_device
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline, StableDiffusionPipeline


class SVDiffModule(nn.Module):
    def __init__(self, weight: torch.Tensor, weight_type: Optional[str] = None, init_delta: Optional[torch.Tensor] = None, scale: float = 1.0):
        """
        SVDiff module to be registered as hook 

        Parameters:
            weight (`torch.Tensor`, *required*): pre-trained weight 
            weight_type (`string`, *optional*): choose from [None, 'conv', '1d']
            init_delta (`string, *optional*`): init values for delta
            scale (`float`): spectral shifts scale. This is intended to use during inference. 
        """
        super().__init__()
        self.weight_type = weight_type
        weight = self._reshape_weight_for_svd(weight)
        # perform SVD
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        self.register_buffer("U", U.detach())
        self.U.requires_grad = False
        self.register_buffer("S", S.detach())
        self.S.requires_grad = False
        self.register_buffer("Vh", Vh.detach())
        self.Vh.requires_grad = False
        # initialize to 0 for smooth tuning 
        self.delta = nn.Parameter(torch.zeros_like(S)) if init_delta is None else nn.Parameter(init_delta)
        self.scale = scale

    def _reshape_weight_for_svd(self, updated_weight: torch.Tensor, original_weight: torch.Tensor = None, reverse: bool = False, **kwargs):
        if self.weight_type is not None:
            if self.weight_type == "conv":
                if not reverse:
                    shape = updated_weight.shape
                    return torch.reshape(updated_weight, (shape[0], shape[1:].numel()))
                else:
                    return torch.reshape(updated_weight, original_weight.shape)
            elif self.weight_type == "1d":
                if not reverse:
                    return updated_weight.unsqueeze(0)
                else:
                    return updated_weight.squeeze(0)
            else:
                raise ValueError(f"`weight_type`={self.weight_type} is invalid!")
        else:
            return updated_weight
        
    def forward(self, weight: torch.Tensor):
        updated_weight = self.U @ torch.diag(F.relu(self.S + self.scale * self.delta)) @ self.Vh
        updated_weight = self._reshape_weight_for_svd(updated_weight, weight, reverse=True)
        return updated_weight


def set_spectral_shifts(model: nn.Module, spectral_shifts_ckpt: str = None, **kwargs):
    """
    Perform SVD and register parametrization to model 

    Args:
        model ([`torch.nn.Module`]):
            The model to perform SVD
        spectral_shifts_ckpt (`str` *optional*):
            The checkpoint path of spectral shifts 

    Return:
        `torch.nn.Module` for a model added the spectral shifts 
        `Dict[str, torch.nn.Module]` as a collection of svdiff modules 
    """
    # key to module
    svdiff_modules = {}
    for name, module in model.named_modules():
        svdiff_module = None
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            svdiff_module = SVDiffModule(module.weight, weight_type="conv")
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
            svdiff_module = SVDiffModule(module.weight, weight_type="1d")
        elif isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            svdiff_module = SVDiffModule(module.weight)
        if svdiff_module:
            svdiff_modules[name+".delta"] = svdiff_module
            # register parametrization 
            nn.utils.parametrize.register_parametrization(module, "weight", svdiff_module)
            
    if spectral_shifts_ckpt:
        param_device = "cpu"
        torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
        with safetensors.safe_open(spectral_shifts_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(svdiff_modules[key], "delta", param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(svdiff_modules[key], "delta", param_device, value=f.get_tensor(key))
        print(f"Resumed from {spectral_shifts_ckpt}")
    return model, svdiff_modules


def slerp(val, low, high):
    """ taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)


def ddim_invert(pipe, prompt=None, image=None, num_inference_steps=50, guidance_scale=1.0, **kwargs):
    assert isinstance(pipe, StableDiffusionPipeline), f"{pipe.__class__.__name__} is not supported!"
    # use only DDIMInversion part
    pipe = StableDiffusionPix2PixZeroPipeline(
        vae=pipe.vae, 
        text_encoder=pipe.text_encoder, 
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=DDIMScheduler.from_config(pipe.scheduler.config),
        inverse_scheduler=DDIMInverseScheduler.from_config(pipe.scheduler.config),
        feature_extractor=pipe.feature_extractor,
        safety_checker=pipe.safety_checker,
        caption_generator=None,
        caption_processor=None,
    )
    # in SVDiff, they use guidance scale=1 in ddim inversion
    inv_latents = pipe.invert(
        prompt=prompt, 
        image=image, 
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_reg_steps=0, # disabled 
        **kwargs
    ).latents
    return inv_latents
