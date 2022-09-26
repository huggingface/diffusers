import inspect
import warnings
from typing import List, Optional, Union
import random

import torch
import numpy as np
from tqdm import tqdm

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker



def dynamic_thresholding_(img, quantile):
    # Dynamic thresholding from Imagen paper (May 2022)
    s = np.quantile(np.abs(img.cpu()), quantile, axis=tuple(range(1,img.ndim)))
    s = np.max(np.append(s, 1.0))
    torch.clamp_(img, -1*s, s)
    torch.FloatTensor.div_(img, s)


def dynamic_thresholding_torch(imgs, quantile):
    # Dynamic thresholding from Imagen paper (May 2022)
    #s = torch.quantile(torch.abs(imgs), quantile, dim=tuple(range(1, imgs.ndim)))
    #s = torch.max(torch.cat((s, torch.ones(1, device=imgs.device)), dim=0), dim=0)[0]
    #return torch.clamp(imgs, -s, s) / s
    quant = torch.quantile(imgs.float().abs(), quantile)
    clipped_latent = torch.clip(imgs, -quant, quant) / quant
    return clipped_latent


def minmax(a):
    max_val = a.max()
    min_val = a.min()
    a = (a - min_val) / (max_val - min_val)
    return a


class StableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
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
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            warnings.warn(
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file",
                DeprecationWarning,
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        use_safety: bool = False,
        weights: Optional[List[float]] = None,
        start_img: Optional[torch.Tensor] = None,
        noise_strength: Optional[float] = None,
        noise_step: Optional[int] = None,
        img2img_strength: Optional[float] = None,
        seed=None,
        verbose=True,
        dynamic_thresholding_quant: float = 0.0,
        dynamic_thresholding_steps: int = 0,
        t_start: Optional[int] = 0,
        noise_strength_before_encode = None,
        loss_callbacks: Optional[List] = None,
        noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # set seed 
        if seed is not None:
            self.set_seed(seed)
            
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )
            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # determine batch size
        if prompt is None:
            batch_size = len(text_embeddings)
        else:
            if isinstance(prompt, str):
                batch_size = 1
                prompt = [prompt]
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # get prompt text embeddings if not given already
        if text_embeddings is None:
            text_embeddings = self.embed_prompts(prompt, weights=weights)
            
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale != 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeddings = self.embed_prompts([""] * batch_size, weights=weights)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = offset
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        init_timestep = num_inference_steps
        offset = 0
        # create starting image/noise
        if noise is None:
            noise = self.sample_noise(width, height, batch_size=batch_size, generator=generator)
            
        if latents is None:
            if start_img is None:
                latents = noise
                
            else:
                # encode start img with vae
                # make it torch tensor first
                latents = self.encode_image(start_img, noise_strength_before_encode=noise_strength_before_encode)
                # add noise
                if noise_strength is not None and noise_strength != 0:
                    # old method to add noise that ignores noise schedule:
                    latents = latents * (1 - noise_strength) + noise * noise_strength
                elif noise_step is not None and noise_step > 0:
                    # now we use the scheduler to add noise
                    latents = self.scheduler.add_noise(latents, noise, noise_step)
                elif img2img_strength is not None and img2img_strength != 0:
                    # with img2img we skip the first (1-strength) * num_inference_steps steps, so we increase the total step count
                    num_inference_steps = int(np.ceil(num_inference_steps / img2img_strength))
                    # set schedule again according to updated steps
                    self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
                    # now we calculate how many skeps are skipped (t_start) and at what point the noise is initialized (init_timestep)
                    init_timestep = int(num_inference_steps * img2img_strength) + offset
                    t_start = num_inference_steps - init_timestep + offset
                    timesteps = int(self.scheduler.timesteps[-init_timestep])
                    timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)
                    # add noise to latents using the timesteps       
                    #if isinstance(self.scheduler, LMSDiscreteScheduler):
                    #    start_timestep = torch.tensor([t_start] * batch_size, dtype=torch.long, device=self.device)
                    #else:
                    start_timestep = timesteps
                    
                    
                    latents = self.scheduler.add_noise(latents, noise, start_timestep)
                    
            #self.scheduler.sigmas[t_start]

            #print("t_start:", t_start)
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = latents * ((self.scheduler.sigmas[t_start]**2 + 1) ** 0.5) 


            
            
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latent_list = [latents]
        
        #print("Num timesteps: ", len(self.scheduler.timesteps))

        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:]), disable=not verbose):
            latent_list.append(latents.cpu())
            
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                count = i + t_start
            else:
                count = t
                
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[count]

                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)            
            
                            
            # add dynamic thresholding
            if dynamic_thresholding_steps > 0 and dynamic_thresholding_quant > 0:
                if i % dynamic_thresholding_steps == 0:
                    denoised_images = self.vae.decode(latents / 0.18215) / 2 + 0.5 # (0, 1)
                    #denoised_images_clipped = dynamic_thresholding_torch(denoised_images, dynamic_thresholding_quant)
                    
                    dynamic_thresholding_(denoised_images, dynamic_thresholding_quant)
                    
                    latents = self.vae.encode((minmax(denoised_images) - 0.5) * 2).mean * 0.18215 
                #dynamic_thresholding_(noise_pred, dynamic_thresholding_quant * 100)
            
            
            # TODO: dynamic and static thresholding could simply be applied to decoded latent (==img)
            
            
                
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, count, latents, **extra_step_kwargs)["prev_sample"]
            
            #### ADDITIONAL GUIDANCE ###
            if loss_callbacks is not None:
                with torch.enable_grad():
                    grads = torch.zeros_like(latents)
                    #denoised_images = None
                    for callback_dict in loss_callbacks:
                        if callback_dict["frequency"] is not None and i % callback_dict["frequency"] == 0:
                            # Requires grad on the latents
                            latents = latents.detach().requires_grad_()
                            if callback_dict["apply_to_image"]:
                                # Get the predicted x0:
                                #latents_x0 = latents - sigma * noise_pred
                                #latents_x0 = self.scheduler.step(noise_pred, count, latents, **extra_step_kwargs)["prev_sample"]
                                # Decode to image space
                                #denoised_images = self.vae.decode((1 / 0.18215) * latents_x0) / 2 + 0.5  # (0, 1)
                                #if denoised_images is None:  
                                denoised_images = self.vae.decode(latents / 0.18215)["sample"] / 2 + 0.5  # (0, 1)


                                # Calculate loss
                                loss = callback_dict["loss_function"](denoised_images)
                            else:
                                loss = callback_dict["loss_function"](latents)
                            # Get gradient
                            cond_grad = -torch.autograd.grad(loss * callback_dict["weight"], latents)[0] 
                            # Modify the latents based on this gradient
                            grads += cond_grad * callback_dict["lr"]
                    
                    
                    latents = latents.detach() + grads * sigma**2
                            
        
        # scale and decode the image latents with vae
        image = self.decode_image(latents, output_type=output_type)


        if output_type == "pil" and use_safety:
            # run safety checker
            check_img = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
            safety_checker_input = self.feature_extractor(check_img, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values)
        else:
            has_nsfw_concept = False

            

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
            
    
    def sample_noise(self, width=512, height=512, batch_size=1, generator=None):
        latents = torch.randn(batch_size, self.unet.in_channels,
                                  height // 8, width // 8, generator=generator, device=self.device)
        return latents
    
    def embed_prompts(self, prompts, weights=None, device=None):
        if device is None:
            device = self.device
        if not isinstance(prompts, list):
            prompts = [prompts]
        text_embeddings = []
        for prompt in prompts:
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input = text_input.input_ids.to(device)
            text_embedding = self.text_encoder(text_input)[0]
            
            text_embeddings.append(text_embedding)
        if weights is None:
            text_embeddings = torch.mean(torch.stack(text_embeddings), 0)
        else:
            weights = torch.tensor(weights, device=text_embedding.device)
            normed_weights = weights / torch.sum(weights)
            text_embeddings = torch.sum(torch.stack(text_embeddings) * normed_weights, 0)
        return text_embeddings
    
    def encode_image(self, image, torch_device=None, noise_strength_before_encode=None):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.is_tensor(image):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            image = torch.from_numpy(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # reverse channel dimension
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        # make it float tensor between 0 and 1
        image = minmax(image).float()
        # add noise
        if noise_strength_before_encode is not None and noise_strength_before_encode > 0:
            image = (1 - noise_strength_before_encode) * image + noise_strength_before_encode * torch.randn_like(image)
        # clip to [0, 1]
        image = torch.clamp(image, 0, 1)
        # move to -1 to 1 for vae
        image = (image - 0.5) * 2
        # encode image
        latents = self.vae.encode(image.to(torch_device))["latent_dist"]
        # encoded img is DiagonalGaussianDistribution, need to sample from it or we take the mean instead
        #latents = latents.sample()
        latents = latents.mean
        # norm latents
        latents = latents * 0.18215
        return latents

    def decode_image(self, latents, output_type="pil", device=None):
        if device is None:
            device = self.device
        if output_type == "latent":
            return latents.detach().cpu()
        latents = latents / 0.18215
        image = self.vae.decode(latents.to(device))["sample"]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu()
        
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            image = image.permute(0, 2, 3, 1).numpy()
        return image
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

