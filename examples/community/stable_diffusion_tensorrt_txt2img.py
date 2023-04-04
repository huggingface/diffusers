#
# Copyright 2023 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from typing import List, Optional, Union

import torch
import tensorrt as trt
from polygraphy import cuda
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from tensorrt_utils import TRT_LOGGER, build_engines, device_view, runEngine, Optimizer, BaseModel
from huggingface_hub import snapshot_download
from diffusers.utils import DIFFUSERS_CACHE, logging
from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput, StableDiffusionSafetyChecker

'''
Installation instructions
python3 -m pip install --upgrade tensorrt polygraphy onnx-graphsurgeon -i https://pypi.ngc.nvidia.com
python3 -m pip install onnxruntime
'''

EXAMPLE_DOC_STRING = """
    Examples:
        ```python3
        >>> import torch
        >>> from diffusers import DDIMScheduler

        >>> # Use the DDIMScheduler scheduler here instead
        >>> scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                        subfolder="scheduler")

        >>> pipe = TensorRTStableDiffusionPipeline("stabilityai/stable-diffusion-2-1",
                                                    revision='fp16',
                                                    torch_dtype=torch.float16,
                                                    scheduler=scheduler
                                                    )

        >>> pipe = pipe.to("cuda")

        >>> prompt = "a beautiful photograph of Mt. Fuji during cherry blossom"
        >>> image = pipe(prompt).images[0]
        >>> image.save('tensorrt_mt_fuji.png')
        ```
"""

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class TensorRTStableDiffusionPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using TensorRT accelerated Stable Diffusion.

    This model inherits from [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        stages=['clip','unet','vae'],
        image_height: int = 768,
        image_width: int = 768,
        max_batch_size: int = 16,
        # ONNX export parameters
        onnx_opset: int = 17,
        onnx_dir: str = 'onnx',
        # TensorRT engine build parameters
        engine_dir: str = 'engine',
        force_engine_rebuild: bool = False,
        timing_cache: str = 'timing_cache'
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)

        self.vae.forward = self.vae.decode

        self.stages = stages
        self.image_height, self.image_width = image_height, image_width
        self.inpaint = False
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.timing_cache = timing_cache
        self.build_static_batch = False
        self.build_dynamic_shape = False
        self.build_preview_features = False

        self.max_batch_size = max_batch_size
        # TODO: Restrict batch size to 4 for larger image dimensions as a WAR for TensorRT limitation.
        if self.build_dynamic_shape or self.image_height > 512 or self.image_width > 512:
            self.max_batch_size = 4

        self.stream = None # loaded in loadResources()
        self.models = {} # loaded in __loadModels()
        self.engine = {} # loaded in build_engines()

    def __loadModels(self):
        # Load pipeline models
        self.embedding_dim = self.text_encoder.config.hidden_size
        models_args = {'device': self.torch_device, 'max_batch_size': self.max_batch_size, \
                       'embedding_dim': self.embedding_dim, 'inpaint': self.inpaint}
        if 'clip' in self.stages:
            self.models['clip'] = make_CLIP(self.text_encoder, **models_args)
        if 'unet' in self.stages:
            self.models['unet'] = make_UNet(self.unet, **models_args)
        if 'vae' in self.stages:
            self.models['vae'] = make_VAE(self.vae, **models_args)

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings: bool = False):
        super().to(torch_device, silence_dtype_warnings)

        self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)
        self.timing_cache = os.path.join(self.cached_folder, self.timing_cache)

        # set device
        self.torch_device = self._execution_device
        logger.warning(f"Running inference on device: {self.torch_device}")

        # load models
        self.__loadModels()

        # build engines
        self.engine = build_engines(self.models, self.engine_dir, self.onnx_dir, self.onnx_opset, \
            opt_image_height=self.image_height, opt_image_width=self.image_width, \
            force_engine_rebuild=self.force_engine_rebuild, static_batch=self.build_static_batch, \
            static_shape=not self.build_dynamic_shape, enable_preview=self.build_preview_features, \
            timing_cache=self.timing_cache)

        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)

        cls.cached_folder = (
                pretrained_model_name_or_path
                if os.path.isdir(pretrained_model_name_or_path)
                else snapshot_download(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                )
            )
        return pipe

    def __encode_prompt(
        self,
        prompt,
        negative_prompt
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        # Tokenize prompt
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.torch_device)

        text_input_ids_inp = device_view(text_input_ids)
        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = runEngine(self.engine['clip'], {"input_ids": text_input_ids_inp}, self.stream)['text_embeddings'].clone()

        # Tokenize negative prompt
        uncond_input_ids = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.torch_device)
        uncond_input_ids_inp = device_view(uncond_input_ids)
        uncond_embeddings = runEngine(self.engine['clip'], {"input_ids": uncond_input_ids_inp}, self.stream)['text_embeddings']

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings


    def __denoise_latent(self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            embeddings_dtype = np.float16
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = device_view(latent_model_input)
            timestep_inp = device_view(timestep_float)
            embeddings_inp = device_view(text_embeddings)
            noise_pred = runEngine(self.engine['unet'], \
                                   {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp}, \
                                    self.stream)['latent']

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        latents = 1. / 0.18215 * latents
        return latents

    def __decode_latent(self, latents):
        images = runEngine(self.engine['vae'], {"latent": device_view(latents)}, self.stream)['images']
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()

    def __loadResources(self, image_height, image_width, batch_size):
        self.stream = cuda.Stream()

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.torch_device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.

        """
        self.generator = generator
        self.denoising_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps, device=self.torch_device)

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Expected prompt to be of type list or str but got {type(prompt)}")

        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {len(prompt)} is larger than allowed {self.max_batch_size}. If dynamic shape is used, then maximum batch size is 4")
        
        # load resources
        self.__loadResources(self.image_height, self.image_width, batch_size)
    
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # CLIP text encoder
            text_embeddings = self.__encode_prompt(prompt, negative_prompt)

            # Pre-initialize latents
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(batch_size,
                                           num_channels_latents,
                                           self.image_height,
                                           self.image_width,
                                           torch.float32,
                                           self.torch_device,
                                           generator)

            # UNet denoiser
            latents = self.__denoise_latent(latents, text_embeddings)

            # VAE decode latent
            images = self.__decode_latent(latents)

        images, has_nsfw_concept = self.run_safety_checker(images, self.torch_device, text_embeddings.dtype)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


class CLIP(BaseModel):
    def __init__(self,
        model,
        device,
        max_batch_size,
        embedding_dim
    ):
        super(CLIP, self).__init__(model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "CLIP"

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings', 'pooler_output']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        return opt_onnx_graph

def make_CLIP(model, device, max_batch_size, embedding_dim, inpaint=False):
    return CLIP(model, device=device, max_batch_size=max_batch_size, \
                embedding_dim=embedding_dim)

class UNet(BaseModel):
    def __init__(self,
        model,
        fp16=False,
        device='cuda',
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UNet, self).__init__(model=model, fp16=fp16, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim, text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
        )

def make_UNet(model, device, max_batch_size, embedding_dim, inpaint=False):
    return UNet(model, fp16=True, device=device, max_batch_size=max_batch_size, \
                embedding_dim=embedding_dim, unet_dim=(9 if inpaint else 4))

class VAE(BaseModel):
    def __init__(self,
        model,
        device,
        max_batch_size,
        embedding_dim
    ):
        super(VAE, self).__init__(model=model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "VAE decoder"

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)

def make_VAE(model, device, max_batch_size, embedding_dim, inpaint=False):
    return VAE(model, device=device, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
