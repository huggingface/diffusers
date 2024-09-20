# Copyright 2024 AuraFlow Authors and The HuggingFace Team. All rights reserved.
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
import inspect 
from typing import List, Optional, Tuple, Union 

import torch 
from transformers import T5Tokenizer, UMT5EncoderModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AuraFlowTransformer2DModel, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0, FusedAttnProcessor2_0, XFormersAttnProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler 
from diffusers.utils import logging, replace_example_docstring 
from diffusers.utils.torch_utils import randn_tensor 
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput 


logger = logging.get_logger(__name__) #pylint: disable=invalid-name 


EXAMPLE_DOC_STRING  = """
    Examples:
    ```py
    >>> import torch
    >>> from diffusers.utils import load_image 
    >>> from pipeline_aura_flow_differential_img2img import AuraFlowDifferentialImg2ImgPipeline
    >>> pipe = AuraFlowDifferentialImg2ImgPipeline.from_pretrained(
            "fal/AuraFlow", torch_dtype=torch.float16
        ).to("cuda")
    >>> source_image = load_image(
            "https://huggingface.co/datasets/OzzyGT/testing-
resources/resolve/main/differential/20240329211129_4024911930.png"
    >>> )
    >>> map = load_image(
            "https://huggingface.co/datasets/OzzyGT/testing-
resources/resolve/main/differential/gradient_mask_2.png"
    >>> )
    >>> prompt = "a green pear"
    >>> negative_prompt = "blurry"
    >>> image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_image,
            num_inference_steps=28,
            guidance_scale=4.5,
            strength=1.0,
            map=map,
    >>> ).images[0]
    ```
""" 


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps 
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    
    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedulers. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps 
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps 
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps 
    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents 
def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents 
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    

class AuraFlowDifferentialImg2ImgPipeline(DiffusionPipeline):
    r"""
    Args:
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. AuraFlow uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [EleutherAI/pile-t5-xl](https://huggingface.co/EleutherAI/pile-t5-xl) variant.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        transformer ([`AuraFlowTransformer2DModel`]):
            Conditional Transformer (MMDiT and DiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    