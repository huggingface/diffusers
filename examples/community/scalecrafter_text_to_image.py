from diffusers import StableDiffusionPipeline
import torch
import torch.nn.functional as F
import math
from diffusers.models.lora import LoRACompatibleConv
from torch import Tensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from typing import Optional
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
import copy
import os 
import requests
import yaml


def read_dilate_settings(path):
    print(f"Reading dilation settings")
    dilate_settings = dict()
    url = os.path.join("https://huggingface.co/datasets/ayushtues/scalecrafter/resolve/main/", path)
    response = requests.get(url, allow_redirects=True)
    content = response.content.decode("utf-8")
    content = content.strip()
    content = content.split('\n')
    for raw_line in content:
        name, dilate = raw_line.split(':')
        dilate_settings[name] = float(dilate)
    return dilate_settings


def read_module_list(path):
    url = os.path.join("https://huggingface.co/datasets/ayushtues/scalecrafter/resolve/main/", path)
    response = requests.get(url, allow_redirects=True)
    content = response.content.decode("utf-8")
    content = content.strip()
    content = content.split('\n')
    module_list = [name.strip() for name in content]
    return module_list


def inflate_kernels(unet, inflate_conv_list, inflation_transform):
    def replace_module(module, name, index=None, value=None):
        if len(name) == 1 and len(index) == 0:
            setattr(module, name[0], value)
            return module

        current_name, next_name = name[0], name[1:]
        current_index, next_index = int(index[0]), index[1:]
        replace = getattr(module, current_name)
        replace[current_index] = replace_module(replace[current_index], next_name, next_index, value)
        setattr(module, current_name, replace)
        return module

    for name, module in unet.named_modules():
        if name in inflate_conv_list:
            weight, bias = module.weight.detach(), module.bias.detach()
            (i, o, *_), kernel_size = (
                weight.shape, int(math.sqrt(inflation_transform.shape[0]))
            )
            transformed_weight = torch.einsum(
                "mn, ion -> iom", inflation_transform.to(dtype=weight.dtype), weight.view(i, o, -1))
            conv = LoRACompatibleConv(
                o, i, (kernel_size, kernel_size),
                stride=module.stride, padding=module.padding, device=weight.device, dtype=weight.dtype
            )
            conv.weight.detach().copy_(transformed_weight.view(i, o, kernel_size, kernel_size))
            conv.bias.detach().copy_(bias)

            sub_names = name.split('.')
            if name.startswith('mid_block'):
                names, indexes = sub_names[1::2], sub_names[2::2]
                unet.mid_block = replace_module(unet.mid_block, names, indexes, conv)
            else:
                names, indexes = sub_names[0::2], sub_names[1::2]
                replace_module(unet, names, indexes, conv)


class ReDilateConvProcessor:
    def __init__(self, module, pf_factor=1.0, mode='bilinear', activate=True):
        self.dilation = math.ceil(pf_factor)
        self.factor = float(self.dilation / pf_factor)
        self.module = module
        self.mode = mode
        self.activate = activate

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            ori_dilation, ori_padding = self.module.dilation, self.module.padding
            inflation_kernel_size = (self.module.weight.shape[-1] - 3) // 2
            self.module.dilation, self.module.padding = self.dilation, (
                self.dilation * (1 + inflation_kernel_size), self.dilation * (1 + inflation_kernel_size)
            )
            ori_size, new_size = (
                (int(input.shape[-2] / self.module.stride[0]), int(input.shape[-1] / self.module.stride[1])),
                (round(input.shape[-2] * self.factor), round(input.shape[-1] * self.factor))
            )
            input = F.interpolate(input, size=new_size, mode=self.mode)
            input = self.module._conv_forward(input, self.module.weight, self.module.bias)
            self.module.dilation, self.module.padding = ori_dilation, ori_padding
            result = F.interpolate(input, size=ori_size, mode=self.mode)
            return result
        else:
            return self.module._conv_forward(input, self.module.weight, self.module.bias)


class ScaledAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, processor, test_res, train_res):
        self.processor = processor
        self.test_res = test_res
        self.train_res = train_res

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        input_ndim = hidden_states.ndim
        if encoder_hidden_states is None:
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                sequence_length = height * width
            else:
                batch_size, sequence_length, _ = hidden_states.shape

            test_train_ratio = float(self.test_res / self.train_res)
            train_sequence_length = sequence_length / test_train_ratio
            scale_factor = math.log(sequence_length, train_sequence_length) ** 0.5
        else:
            scale_factor = 1

        original_scale = attn.scale
        attn.scale = attn.scale * scale_factor
        hidden_states = self.processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        attn.scale = original_scale
        return hidden_states


class ScaleCrafterTexttoImagePipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        safety_checker: StableDiffusionSafetyChecker,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, feature_extractor=feature_extractor, safety_checker=safety_checker, requires_safety_checker=requires_safety_checker)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def __call__(
            self, 
            prompt=None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt=None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 1.0,
            generator=None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback=None,
            callback_steps: int = 1,
            cross_attention_kwargs=None,
            guidance_rescale: float = 0.0,
            config_path: str = "sd1.5_1024x1024.yaml",
    ):
        url = os.path.join("https://huggingface.co/datasets/ayushtues/scalecrafter/resolve/main/configs/", config_path)
        response = requests.get(url, allow_redirects=True)
        content = response.content.decode("utf-8")
        config = yaml.safe_load(content)
        # 0. Default height and width to unet
        height = config['latent_height'] or self.unet.config.sample_size * self.vae_scale_factor
        width = config['latent_width'] or self.unet.config.sample_size * self.vae_scale_factor
        inflate_tau = config['inflate_tau']
        ndcfg_tau = config['ndcfg_tau']
        dilate_tau = config['dilate_tau']
        progressive = config['progressive']

        dilate_settings = read_dilate_settings(config['dilate_settings']) \
            if config['dilate_settings'] is not None else dict()
        ndcfg_dilate_settings = read_dilate_settings(config['ndcfg_dilate_settings']) \
            if config['ndcfg_dilate_settings'] is not None else dict()
        inflate_settings = read_module_list(config['inflate_settings']) \
            if config['inflate_settings'] is not None else list()

        transform = None


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
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

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        unet_inflate, unet_inflate_vanilla = None, None

        # We have 3 unets, the original, the inflated, and the inflated vanilla
        if transform is not None:
            unet_inflate = copy.deepcopy(self.unet)
            if inflate_settings is not None:
                inflate_kernels(unet_inflate, inflate_settings, transform)

        if transform is not None and ndcfg_tau > 0:
            unet_inflate_vanilla = copy.deepcopy(self.unet)
            if inflate_settings is not None:
                inflate_kernels(unet_inflate_vanilla, inflate_settings, transform)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # use  inflated unet initially and the normal unet after few steps
                unet = unet_inflate if i < inflate_tau and transform is not None else self.unet
                backup_forwards = dict()
                for name, module in unet.named_modules():
                    if name in dilate_settings.keys():
                        backup_forwards[name] = module.forward
                        dilate = dilate_settings[name]
                        if progressive:
                            dilate = max(math.ceil(dilate * ((dilate_tau - i) / dilate_tau)), 2)
                        if i < inflate_tau and name in inflate_settings:
                            dilate = dilate / 2
                        # print(f"{name}: {dilate} {i < dilate_tau}")
                        module.forward = ReDilateConvProcessor(
                            module, dilate, mode='bilinear', activate=i < dilate_tau
                        )

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                for name, module in unet.named_modules():
                    if name in backup_forwards.keys():
                        module.forward = backup_forwards[name]

                if i < ndcfg_tau:
                    unet = unet_inflate_vanilla if i < inflate_tau and transform is not None else self.unet
                    backup_forwards = dict()
                    for name, module in unet.named_modules():
                        if name in ndcfg_dilate_settings.keys():
                            backup_forwards[name] = module.forward
                            dilate = ndcfg_dilate_settings[name]
                            if progressive:
                                dilate = max(math.ceil(dilate * ((ndcfg_tau - i) / ndcfg_tau)), 2)
                            if i < inflate_tau and name in inflate_settings:
                                dilate = dilate / 2
                            # print(f"{name}: {dilate} {i < dilate_tau}")
                            module.forward = ReDilateConvProcessor(
                                module, dilate, mode='bilinear', activate=i < ndcfg_tau
                            )

                    noise_pred_vanilla = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    for name, module in unet.named_modules():
                        if name in backup_forwards.keys():
                            module.forward = backup_forwards[name]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if i < ndcfg_tau:
                        noise_pred_vanilla, _ = noise_pred_vanilla.chunk(2)
                        noise_pred = noise_pred_vanilla + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

