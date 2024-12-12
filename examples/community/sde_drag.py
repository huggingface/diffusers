import math
import tempfile
from typing import List, Optional

import numpy as np
import PIL.Image
import torch
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers, StableDiffusionLoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler


class SdeDragPipeline(DiffusionPipeline):
    r"""
    Pipeline for image drag-and-drop editing using stochastic differential equations: https://arxiv.org/abs/2311.01410.
    Please refer to the [official repository](https://github.com/ML-GSAI/SDE-Drag) for more information.

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Please use
            [`DDIMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        source_points: List[List[int]],
        target_points: List[List[int]],
        t0: Optional[float] = 0.6,
        steps: Optional[int] = 200,
        step_size: Optional[int] = 2,
        image_scale: Optional[float] = 0.3,
        adapt_radius: Optional[int] = 5,
        min_lora_scale: Optional[float] = 0.5,
        generator: Optional[torch.Generator] = None,
    ):
        r"""
        Function invoked when calling the pipeline for image editing.
        Args:
            prompt (`str`, *required*):
                The prompt to guide the image editing.
            image (`PIL.Image.Image`, *required*):
                Which will be edited, parts of the image will be masked out with `mask_image` and edited
                according to `prompt`.
            mask_image (`PIL.Image.Image`, *required*):
                To mask `image`. White pixels in the mask will be edited, while black pixels will be preserved.
            source_points (`List[List[int]]`, *required*):
                Used to mark the starting positions of drag editing in the image, with each pixel represented as a
                `List[int]` of length 2.
            target_points (`List[List[int]]`, *required*):
                Used to mark the target positions of drag editing in the image, with each pixel represented as a
                `List[int]` of length 2.
            t0 (`float`, *optional*, defaults to 0.6):
                The time parameter. Higher t0 improves the fidelity while lowering the faithfulness of the edited images
                and vice versa.
            steps (`int`, *optional*, defaults to 200):
                The number of sampling iterations.
            step_size (`int`, *optional*, defaults to 2):
                The drag diatance of each drag step.
            image_scale (`float`, *optional*, defaults to 0.3):
                To avoid duplicating the content, use image_scale to perturbs the source.
            adapt_radius (`int`, *optional*, defaults to 5):
                The size of the region for copy and paste operations during each step of the drag process.
            min_lora_scale (`float`, *optional*, defaults to 0.5):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                min_lora_scale specifies the minimum LoRA scale during the image drag-editing process.
            generator ('torch.Generator', *optional*, defaults to None):
                To make generation deterministic(https://pytorch.org/docs/stable/generated/torch.Generator.html).
        Examples:
        ```py
        >>> import PIL
        >>> import torch
        >>> from diffusers import DDIMScheduler, DiffusionPipeline

        >>> # Load the pipeline
        >>> model_path = "runwayml/stable-diffusion-v1-5"
        >>> scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        >>> pipe = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, custom_pipeline="sde_drag")
        >>> pipe.to('cuda')

        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> # If not training LoRA, please avoid using torch.float16
        >>> # pipe.to(torch.float16)

        >>> # Provide prompt, image, mask image, and the starting and target points for drag editing.
        >>> prompt = "prompt of the image"
        >>> image = PIL.Image.open('/path/to/image')
        >>> mask_image = PIL.Image.open('/path/to/mask_image')
        >>> source_points = [[123, 456]]
        >>> target_points = [[234, 567]]

        >>> # train_lora is optional, and in most cases, using train_lora can better preserve consistency with the original image.
        >>> pipe.train_lora(prompt, image)

        >>> output = pipe(prompt, image, mask_image, source_points, target_points)
        >>> output_image = PIL.Image.fromarray(output)
        >>> output_image.save("./output.png")
        ```
        """

        self.scheduler.set_timesteps(steps)

        noise_scale = (1 - image_scale**2) ** (0.5)

        text_embeddings = self._get_text_embed(prompt)
        uncond_embeddings = self._get_text_embed([""])
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latent = self._get_img_latent(image)

        mask = mask_image.resize((latent.shape[3], latent.shape[2]))
        mask = torch.tensor(np.array(mask))
        mask = mask.unsqueeze(0).expand_as(latent).to(self.device)

        source_points = torch.tensor(source_points).div(torch.tensor([8]), rounding_mode="trunc")
        target_points = torch.tensor(target_points).div(torch.tensor([8]), rounding_mode="trunc")

        distance = target_points - source_points
        distance_norm_max = torch.norm(distance.float(), dim=1, keepdim=True).max()

        if distance_norm_max <= step_size:
            drag_num = 1
        else:
            drag_num = distance_norm_max.div(torch.tensor([step_size]), rounding_mode="trunc")
            if (distance_norm_max / drag_num - step_size).abs() > (
                distance_norm_max / (drag_num + 1) - step_size
            ).abs():
                drag_num += 1

        latents = []
        for i in tqdm(range(int(drag_num)), desc="SDE Drag"):
            source_new = source_points + (i / drag_num * distance).to(torch.int)
            target_new = source_points + ((i + 1) / drag_num * distance).to(torch.int)

            latent, noises, hook_latents, lora_scales, cfg_scales = self._forward(
                latent, steps, t0, min_lora_scale, text_embeddings, generator
            )
            latent = self._copy_and_paste(
                latent,
                source_new,
                target_new,
                adapt_radius,
                latent.shape[2] - 1,
                latent.shape[3] - 1,
                image_scale,
                noise_scale,
                generator,
            )
            latent = self._backward(
                latent, mask, steps, t0, noises, hook_latents, lora_scales, cfg_scales, text_embeddings, generator
            )

            latents.append(latent)

        result_image = 1 / 0.18215 * latents[-1]

        with torch.no_grad():
            result_image = self.vae.decode(result_image).sample

        result_image = (result_image / 2 + 0.5).clamp(0, 1)
        result_image = result_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        result_image = (result_image * 255).astype(np.uint8)

        return result_image

    def train_lora(self, prompt, image, lora_step=100, lora_rank=16, generator=None):
        accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0
                    if hasattr(torch.nn.functional, "scaled_dot_product_attention")
                    else LoRAAttnProcessor
                )
            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
            )

        self.unet.set_attn_processor(unet_lora_attn_procs)
        unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)
        params_to_optimize = unet_lora_layers.parameters()

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=2e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )

        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=lora_step,
            num_cycles=1,
            power=1.0,
        )

        unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
        optimizer = accelerator.prepare_optimizer(optimizer)
        lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

        with torch.no_grad():
            text_inputs = self._tokenize_prompt(prompt, tokenizer_max_length=None)
            text_embedding = self._encode_prompt(
                text_inputs.input_ids, text_inputs.attention_mask, text_encoder_use_attention_mask=False
            )

        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        image = image_transforms(image).to(self.device, dtype=self.vae.dtype)
        image = image.unsqueeze(dim=0)
        latents_dist = self.vae.encode(image).latent_dist

        for _ in tqdm(range(lora_step), desc="Train LoRA"):
            self.unet.train()
            model_input = latents_dist.sample() * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn(
                model_input.size(),
                dtype=model_input.dtype,
                layout=model_input.layout,
                device=model_input.device,
                generator=generator,
            )
            bsz, channels, height, width = model_input.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=model_input.device, generator=generator
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)

            # Predict the noise residual
            model_pred = self.unet(noisy_model_input, timesteps, text_embedding).sample

            # Get the target for loss depending on the prediction type
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as save_lora_dir:
            StableDiffusionLoraLoaderMixin.save_lora_weights(
                save_directory=save_lora_dir,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=None,
            )

            self.unet.load_attn_procs(save_lora_dir)

    def _tokenize_prompt(self, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            max_length = self.tokenizer.model_max_length

        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return text_inputs

    def _encode_prompt(self, input_ids, attention_mask, text_encoder_use_attention_mask=False):
        text_input_ids = input_ids.to(self.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    @torch.no_grad()
    def _get_text_embed(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def _copy_and_paste(
        self, latent, source_new, target_new, adapt_radius, max_height, max_width, image_scale, noise_scale, generator
    ):
        def adaption_r(source, target, adapt_radius, max_height, max_width):
            r_x_lower = min(adapt_radius, source[0], target[0])
            r_x_upper = min(adapt_radius, max_width - source[0], max_width - target[0])
            r_y_lower = min(adapt_radius, source[1], target[1])
            r_y_upper = min(adapt_radius, max_height - source[1], max_height - target[1])
            return r_x_lower, r_x_upper, r_y_lower, r_y_upper

        for source_, target_ in zip(source_new, target_new):
            r_x_lower, r_x_upper, r_y_lower, r_y_upper = adaption_r(
                source_, target_, adapt_radius, max_height, max_width
            )

            source_feature = latent[
                :, :, source_[1] - r_y_lower : source_[1] + r_y_upper, source_[0] - r_x_lower : source_[0] + r_x_upper
            ].clone()

            latent[
                :, :, source_[1] - r_y_lower : source_[1] + r_y_upper, source_[0] - r_x_lower : source_[0] + r_x_upper
            ] = image_scale * source_feature + noise_scale * torch.randn(
                latent.shape[0],
                4,
                r_y_lower + r_y_upper,
                r_x_lower + r_x_upper,
                device=self.device,
                generator=generator,
            )

            latent[
                :, :, target_[1] - r_y_lower : target_[1] + r_y_upper, target_[0] - r_x_lower : target_[0] + r_x_upper
            ] = source_feature * 1.1
        return latent

    @torch.no_grad()
    def _get_img_latent(self, image, height=None, weight=None):
        data = image.convert("RGB")
        if height is not None:
            data = data.resize((weight, height))
        transform = transforms.ToTensor()
        data = transform(data).unsqueeze(0)
        data = (data * 2.0) - 1.0
        data = data.to(self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(data).latent_dist.sample()
        latent = 0.18215 * latent
        return latent

    @torch.no_grad()
    def _get_eps(self, latent, timestep, guidance_scale, text_embeddings, lora_scale=None):
        latent_model_input = torch.cat([latent] * 2) if guidance_scale > 1.0 else latent
        text_embeddings = text_embeddings if guidance_scale > 1.0 else text_embeddings.chunk(2)[1]

        cross_attention_kwargs = None if lora_scale is None else {"scale": lora_scale}

        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        elif guidance_scale == 1.0:
            noise_pred_text = noise_pred
            noise_pred_uncond = 0.0
        else:
            raise NotImplementedError(guidance_scale)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def _forward_sde(
        self, timestep, sample, guidance_scale, text_embeddings, steps, eta=1.0, lora_scale=None, generator=None
    ):
        num_train_timesteps = len(self.scheduler)
        alphas_cumprod = self.scheduler.alphas_cumprod
        initial_alpha_cumprod = torch.tensor(1.0)

        prev_timestep = timestep + num_train_timesteps // steps

        alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else initial_alpha_cumprod
        alpha_prod_t_prev = alphas_cumprod[prev_timestep]

        beta_prod_t_prev = 1 - alpha_prod_t_prev

        x_prev = (alpha_prod_t_prev / alpha_prod_t) ** (0.5) * sample + (1 - alpha_prod_t_prev / alpha_prod_t) ** (
            0.5
        ) * torch.randn(
            sample.size(), dtype=sample.dtype, layout=sample.layout, device=self.device, generator=generator
        )
        eps = self._get_eps(x_prev, prev_timestep, guidance_scale, text_embeddings, lora_scale)

        sigma_t_prev = (
            eta
            * (1 - alpha_prod_t) ** (0.5)
            * (1 - alpha_prod_t_prev / (1 - alpha_prod_t_prev) * (1 - alpha_prod_t) / alpha_prod_t) ** (0.5)
        )

        pred_original_sample = (x_prev - beta_prod_t_prev ** (0.5) * eps) / alpha_prod_t_prev ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t - sigma_t_prev**2) ** (0.5)

        noise = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample - pred_sample_direction_coeff * eps
        ) / sigma_t_prev

        return x_prev, noise

    def _sample(
        self,
        timestep,
        sample,
        guidance_scale,
        text_embeddings,
        steps,
        sde=False,
        noise=None,
        eta=1.0,
        lora_scale=None,
        generator=None,
    ):
        num_train_timesteps = len(self.scheduler)
        alphas_cumprod = self.scheduler.alphas_cumprod
        final_alpha_cumprod = torch.tensor(1.0)

        eps = self._get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale)

        prev_timestep = timestep - num_train_timesteps // steps

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        sigma_t = (
            eta
            * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** (0.5)
            * (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5)
            if sde
            else 0
        )

        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t_prev - sigma_t**2) ** (0.5)

        noise = (
            torch.randn(
                sample.size(), dtype=sample.dtype, layout=sample.layout, device=self.device, generator=generator
            )
            if noise is None
            else noise
        )
        latent = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_coeff * eps + sigma_t * noise
        )

        return latent

    def _forward(self, latent, steps, t0, lora_scale_min, text_embeddings, generator):
        def scale_schedule(begin, end, n, length, type="linear"):
            if type == "constant":
                return end
            elif type == "linear":
                return begin + (end - begin) * n / length
            elif type == "cos":
                factor = (1 - math.cos(n * math.pi / length)) / 2
                return (1 - factor) * begin + factor * end
            else:
                raise NotImplementedError(type)

        noises = []
        latents = []
        lora_scales = []
        cfg_scales = []
        latents.append(latent)
        t0 = int(t0 * steps)
        t_begin = steps - t0

        length = len(self.scheduler.timesteps[t_begin - 1 : -1]) - 1
        index = 1
        for t in self.scheduler.timesteps[t_begin:].flip(dims=[0]):
            lora_scale = scale_schedule(1, lora_scale_min, index, length, type="cos")
            cfg_scale = scale_schedule(1, 3.0, index, length, type="linear")
            latent, noise = self._forward_sde(
                t, latent, cfg_scale, text_embeddings, steps, lora_scale=lora_scale, generator=generator
            )

            noises.append(noise)
            latents.append(latent)
            lora_scales.append(lora_scale)
            cfg_scales.append(cfg_scale)
            index += 1
        return latent, noises, latents, lora_scales, cfg_scales

    def _backward(
        self, latent, mask, steps, t0, noises, hook_latents, lora_scales, cfg_scales, text_embeddings, generator
    ):
        t0 = int(t0 * steps)
        t_begin = steps - t0

        hook_latent = hook_latents.pop()
        latent = torch.where(mask > 128, latent, hook_latent)
        for t in self.scheduler.timesteps[t_begin - 1 : -1]:
            latent = self._sample(
                t,
                latent,
                cfg_scales.pop(),
                text_embeddings,
                steps,
                sde=True,
                noise=noises.pop(),
                lora_scale=lora_scales.pop(),
                generator=generator,
            )
            hook_latent = hook_latents.pop()
            latent = torch.where(mask > 128, latent, hook_latent)
        return latent
