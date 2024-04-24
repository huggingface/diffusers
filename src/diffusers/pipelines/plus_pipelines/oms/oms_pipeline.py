from ...stable_diffusion.pipeline_stable_diffusion import *
import copy
import torch
from safetensors import safe_open
from .process import load_seg_model, generate_mask
from ....utils import is_torch2_available
from ... import UNet2DConditionModel
import numpy as np
import PIL

if is_torch2_available():
    from .attention_processor import REFAttnProcessor2_0 as REFAttnProcessor
    from .attention_processor import AttnProcessor2_0 as AttnProcessor
    from .attention_processor import REFAnimateDiffAttnProcessor2_0 as REFAnimateDiffAttnProcessor
else:
    from .attention_processor import REFAttnProcessor, AttnProcessor

def prepare_image(image, height, width):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Image as float32
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


def prepare_mask(image, height, width):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 1, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)
        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.NEAREST) for i in image]
            image = [np.array(i.convert("L"))[..., None] for i in image]
            image = np.stack(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.stack([i[..., None] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.
        image[image > 0.5] = 1
        image[image <= 0.5] = 0

    return image

class ClothAdapter:
    def __init__(self, sd_pipe, ref_path, device, enable_cloth_guidance, set_seg_model=True):
        self.enable_cloth_guidance = enable_cloth_guidance
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.set_adapter(self.pipe.unet, "write")

        ref_unet = copy.deepcopy(sd_pipe.unet)
        if ref_unet.config.in_channels == 9:
            ref_unet.conv_in = torch.nn.Conv2d(4, 320, ref_unet.conv_in.kernel_size, ref_unet.conv_in.stride, ref_unet.conv_in.padding)
            ref_unet.register_to_config(in_channels=4)
        state_dict = {}
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        ref_unet.load_state_dict(state_dict, strict=False)

        self.ref_unet = ref_unet.to(self.device, dtype=self.pipe.dtype)
        self.set_adapter(self.ref_unet, "read")
        if set_seg_model:
            self.set_seg_model()
        self.attn_store = {}

    def set_seg_model(self, ):
        checkpoint_path = 'checkpoints/cloth_segm.pth'
        self.seg_net = load_seg_model(checkpoint_path, device=self.device)

    def set_adapter(self, unet, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
            self,
            cloth_image,
            cloth_mask_image=None,
            prompt=None,
            a_prompt="best quality, high quality",
            num_images_per_prompt=4,
            negative_prompt=None,
            seed=-1,
            guidance_scale=7.5,
            cloth_guidance_scale=2.5,
            num_inference_steps=20,
            height=512,
            width=384,
            **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = self.pipe.encode_prompt([""], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        if self.enable_cloth_guidance:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                cloth_guidance_scale=cloth_guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0, "enable_cloth_guidance": self.enable_cloth_guidance},
                **kwargs,
            ).images
        else:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0, "enable_cloth_guidance": self.enable_cloth_guidance},
                **kwargs,
            ).images

        return images, cloth_mask_image

    def generate_inpainting(
            self,
            cloth_image,
            cloth_mask_image=None,
            num_images_per_prompt=4,
            seed=-1,
            cloth_guidance_scale=2.5,
            num_inference_steps=20,
            height=512,
            width=384,
            **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        with torch.inference_mode():
            prompt_embeds_null = self.pipe.encode_prompt([""], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds_null,
            cloth_guidance_scale=cloth_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": cloth_guidance_scale > 1.0, "enable_cloth_guidance": False},
            **kwargs,
        ).images

        return images, cloth_mask_image


class ClothAdapter_AnimateDiff:
    def __init__(self, sd_pipe, pipe_path, ref_path, device, set_seg_model=True):
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.set_adapter(self.pipe.unet, "write")

        ref_unet = UNet2DConditionModel.from_pretrained(pipe_path, subfolder='unet', torch_dtype=sd_pipe.dtype)
        state_dict = {}
        with safe_open(ref_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        ref_unet.load_state_dict(state_dict, strict=False)

        self.ref_unet = ref_unet.to(self.device)
        self.set_adapter(self.ref_unet, "read")
        if set_seg_model:
            self.set_seg_model()
        self.attn_store = {}

    def set_seg_model(self, ):
        checkpoint_path = 'checkpoints/cloth_segm.pth'
        self.seg_net = load_seg_model(checkpoint_path, device=self.device)

    def set_adapter(self, unet, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name and "motion_modules" not in name:
                attn_procs[name] = REFAnimateDiffAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
            self,
            cloth_image,
            cloth_mask_image=None,
            prompt=None,
            a_prompt="best quality, high quality",
            num_images_per_prompt=4,
            negative_prompt=None,
            seed=-1,
            guidance_scale=7.5,
            cloth_guidance_scale=3.,
            num_inference_steps=20,
            height=512,
            width=384,
            **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = "bare, naked, nude, undressed, monochrome, lowres, bad anatomy, worst quality, low quality"

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = self.pipe.encode_prompt([""], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        frames = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            cloth_guidance_scale=cloth_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0},
            **kwargs,
        ).frames

        return frames, cloth_mask_image

class OmsDiffusionPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            guidance_scale: float = 5.,
            cloth_guidance_scale: float = 2.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, device, batch_size * num_images_per_prompt
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cloth, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_cloth)
                        + cloth_guidance_scale * (noise_pred_cloth - noise_pred_uncond)
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)