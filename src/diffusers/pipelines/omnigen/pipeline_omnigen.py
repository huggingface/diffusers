from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import numpy as np
import torch
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.models.transformers import OmniGenTransformer
from diffusers.utils.omnigen_processor import OmniGenProcessor
from diffusers.loaders import TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import OmniGenPipeline
        >>> pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt, num_inference_steps=50, guidance_scale=3.0).images[0]
        >>> image.save("omnigen.png")
        ```
"""

class OmniGenPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using OmniGen.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model for encoding and decoding images
        model ([`OmniGenTransformer`]):
            OmniGen transformer model for generating images
        processor ([`OmniGenProcessor`]):
            Processor for handling text and image inputs
    """
    model_cpu_offload_seq = "model->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGenTransformer,
        processor: OmniGenProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            model=model,
            processor=processor,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.model_cpu_offload = False

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None):
        """
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance.
        """
        self.model_cpu_offload = True
        if gpu_id is None:
            gpu_id = 0

        self._move_text_encoder_to_cpu()
        self.model.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()

    def disable_model_cpu_offload(self):
        """
        Disable CPU offloading.
        """
        self.model_cpu_offload = False
        device = self._execution_device
        self.model.to(device)
        self.vae.to(device)

    def enable_vae_slicing(self):
        """
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def prepare_latents(
        self, 
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Prepare latents for diffusion.
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        
        if latents is None:
            shape = (batch_size, 4, latent_height, latent_width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                Optional input images to condition generation on.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 3.0):
                Guidance scale for text conditioning.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Whether to use image guidance when input images are provided.
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Guidance scale for image conditioning.
            max_input_image_size (`int`, *optional*, defaults to 1024):
                Maximum size for input images.
            separate_cfg_infer (`bool`, *optional*, defaults to True):
                Whether to separate classifier-free guidance inference.
            use_kv_cache (`bool`, *optional*, defaults to True):
                Whether to use key-value cache for transformer.
            offload_kv_cache (`bool`, *optional*, defaults to True):
                Whether to offload key-value cache to CPU.
            generator (`torch.Generator`, *optional*):
                A torch generator to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents.
            output_type (`str`, *optional*, defaults to "pil"):
                The output format of the generated image.
            return_dict (`bool`, *optional*, defaults to True):
                Whether or not to return a [`~pipelines.stable_diffusion.OmniGenPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function calls.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.OmniGenPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.OmniGenPipelineOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to model dimensions if not specified
        height = height or 1024
        width = width or 1024

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if input_images is not None:
                input_images = [input_images]
        else:
            batch_size = len(prompt)

        device = self._execution_device
        
        # 1. Process inputs
        input_data = self.processor(
            prompt, 
            input_images, 
            height=height, 
            width=width,
            use_img_cfg=use_img_guidance,
            separate_cfg_input=separate_cfg_infer
        )

        # 2. Define number of train timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=self.model.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 4. Prepare image latents if using image guidance
        input_img_latents = []
        if input_images is not None:
            if self.model_cpu_offload:
                self.vae.to(device)
            
            if separate_cfg_infer:
                for temp_pixel_values in input_data['input_pixel_values']:
                    temp_input_latents = []
                    for img in temp_pixel_values:
                        img = self.vae.encode(img.to(device)).latent_dist.sample()
                        temp_input_latents.append(img)
                    input_img_latents.append(temp_input_latents)
            else:
                for img in input_data['input_pixel_values']:
                    img = self.vae.encode(img.to(device)).latent_dist.sample()
                    input_img_latents.append(img)

            if self.model_cpu_offload:
                self.vae.to('cpu')
                torch.cuda.empty_cache()

        # 5. Set model kwargs for inference
        model_kwargs = {
            'input_ids': input_data['input_ids'].to(device),
            'input_img_latents': input_img_latents,
            'input_image_sizes': input_data['input_image_sizes'],
            'attention_mask': input_data['attention_mask'].to(device),
            'position_ids': input_data['position_ids'].to(device),
            'cfg_scale': guidance_scale,
            'img_cfg_scale': img_guidance_scale,
            'use_img_cfg': use_img_guidance,
            'use_kv_cache': use_kv_cache,
            'offload_model': self.model_cpu_offload,
        }

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict the noise residual
                if separate_cfg_infer:
                    noise_pred = self.model.forward_with_separate_cfg(
                        latent_model_input, t, **model_kwargs
                    )
                else:
                    noise_pred = self.model.forward_with_cfg(
                        latent_model_input, t, **model_kwargs
                    )

                # Perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if self.model_cpu_offload:
            self.vae.to(device)

        # 8. Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        if self.model_cpu_offload:
            self.vae.to('cpu')
            torch.cuda.empty_cache()

        # 9. Convert to PIL
        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        elif output_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return OmniGenPipelineOutput(images=image)

class OmniGenPipelineOutput:
    """
    Output class for OmniGen pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`):
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]

    def __init__(self, images: Union[List[PIL.Image.Image], np.ndarray]):
        self.images = images