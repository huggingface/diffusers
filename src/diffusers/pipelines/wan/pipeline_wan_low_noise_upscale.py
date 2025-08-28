import math
from typing import Tuple, Optional, Union, List

import torch
from PIL import Image, ImageFilter, ImageEnhance
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import DiffusionPipeline, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin


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


def _round_to_valid_dims(width: int, height: int, vae_scale_factor_spatial: int, patch_size_hw: Tuple[int, int]) -> Tuple[int, int]:
    mod_value = vae_scale_factor_spatial * patch_size_hw[1]
    width = max(mod_value, math.ceil(width / mod_value) * mod_value)
    height = max(mod_value, math.ceil(height / mod_value) * mod_value)
    return int(width), int(height)

def latent_to_image(latents, vae, latents_std, latents_mean, video_processor):
    # Keep computation on the latents' current device so accelerate can onload the VAE
    latents = latents.to(dtype=vae.dtype)
    latents_std = latents_std.to(latents.device)
    latents_mean = latents_mean.to(latents.device)
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    frames = video_processor.postprocess_video(video, output_type="pil")
    return frames[0][0]
    
class WanLowNoiseUpscalePipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """Image-to-Image upscaler using WAN 2.2 T2V low-noise transformer for detail enhancement.
    Uses only the low-noise transformer to add details and enhance image quality based on text prompts.
    Optimized for fast upscaling and detail synthesis.
    """
    
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        scheduler: UniPCMultistepScheduler,
        text_encoder: UMT5EncoderModel,
        tokenizer: AutoTokenizer,
        model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        vae_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        
        self._target_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._target_dtype = dtype if dtype is not None else (torch.float16 if self._target_device.type == "cuda" else torch.float32)

        if vae is None:
            vae = AutoencoderKLWan.from_pretrained(vae_id, subfolder="vae", torch_dtype=torch.float32)
        if transformer is None:
            transformer = WanTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer_2", torch_dtype=self._target_dtype
            )

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            
        if text_encoder is None:
            text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
        
        if scheduler is None:
            scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
            scheduler = UniPCMultistepScheduler.from_config(scheduler.config, flow_shift=8.0)
        
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=int(getattr(self.vae.config, "scale_factor_spatial", 8)))
        
        self.vae_scale_factor_spatial = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        self.vae_scale_factor_temporal = int(getattr(self.vae.config, "scale_factor_temporal", 4))

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        vae_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        **kwargs,
    ):
        """Create pipeline with automatic component loading."""
        device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype if dtype is not None else (torch.float16 if device.type == "cuda" else torch.float32)
        
        vae = AutoencoderKLWan.from_pretrained(vae_id, subfolder="vae", torch_dtype=torch.float32)
        
        scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        scheduler = UniPCMultistepScheduler.from_config(scheduler.config, flow_shift=8.0)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
        
        if transformer is None:
            transformer = WanTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer_2", torch_dtype=dtype
            )
        
        pipeline = cls(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
        return pipeline

    def prompt_clean(self, text: str) -> str:
        """Clean and normalize prompt text."""
        return text.strip()

    def _encode_text_batch(self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]] = "", max_sequence_length: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both positive and negative prompts simultaneously for efficiency."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self.prompt_clean(u) for u in prompt]
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt = [self.prompt_clean(u) for u in negative_prompt]
        
        batch_size = len(prompt)
        if len(negative_prompt) == 1 and batch_size > 1:
            negative_prompt = negative_prompt * batch_size
        elif len(negative_prompt) != batch_size:
            raise ValueError(f"Prompt and negative_prompt must have same length, got {batch_size} and {len(negative_prompt)}")
        
        all_prompts = prompt + negative_prompt
        
        text_inputs = self.tokenizer(
            all_prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        device = self.device  # This will access the DiffusionPipeline property
        dtype = self._target_dtype
        
        with torch.no_grad():
            # Let offloading place text encoder; use its current device
            text_encoder_device = next(self.text_encoder.parameters()).device
            prompt_embeds = self.text_encoder(
                text_input_ids.to(text_encoder_device), 
                mask.to(text_encoder_device)
            ).last_hidden_state
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
            )
        
        # Split back into positive and negative embeddings
        positive_embeds = prompt_embeds[:batch_size]
        negative_embeds = prompt_embeds[batch_size:]
        
        return positive_embeds, negative_embeds

    def _encode_text(self, texts: Union[str, List[str]], max_len: int = 512) -> torch.Tensor:
        """Single text encoding method for backwards compatibility."""
        prompt_embeds, _ = self._encode_text_batch(texts, "", max_len)
        return prompt_embeds

    def _compute_target_size(
        self,
        image: Image.Image,
        scale: Optional[float],
        width: Optional[int],
        height: Optional[int],
    ) -> Tuple[int, int]:
        if width and height:
            return int(width), int(height)
        if scale and scale > 0:
            w = max(16, int(round(image.width * float(scale))))
            h = max(16, int(round(image.height * float(scale))))
            return w, h
        if width and not height:
            h = int(round(image.height * (width / float(image.width))))
            return int(width), max(16, h)
        if height and not width:
            w = int(round(image.width * (height / float(image.height))))
            return max(16, w), int(height)
        # Default: identity
        return image.width, image.height

    def __call__(
        self,
        image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.2,
        strength: float = 0.8,
        sharpen_input: float = 0.0,
        desaturate_input: float = 0.0,
        pre_downscale_factor: float = 1.0,
    ) -> Image.Image:
        """Run img2img upscaling using WAN 2.2 low-noise transformer for detail enhancement.
        
        Uses only the low-noise transformer to add details and enhance image quality based on text prompts.
        """
        device = self.device  # Access DiffusionPipeline's device property
        dtype = self._target_dtype
        exec_device = getattr(self, "_execution_device", self.device)

        vae = self.vae
        transformer = self.transformer
        scheduler = self.scheduler
        video_processor = self.video_processor

        # Compute target size and round to valid dims
        target_w, target_h = self._compute_target_size(image, scale, width, height)
        vae_sf = int(getattr(vae.config, "scale_factor_spatial", 8))
        target_w, target_h = _round_to_valid_dims(target_w, target_h, vae_sf, transformer.config.patch_size)
        
        # Precompute scheduler timesteps early to avoid later delays
        desired_steps = int(max(1, num_inference_steps))
        s = float(max(1e-4, min(1.0, strength)))
        effective_total_steps = int(max(desired_steps, math.ceil(desired_steps / s)))
        effective_total_steps = int(min(2000, effective_total_steps))
        scheduler.set_timesteps(effective_total_steps, device=exec_device)
        timesteps = scheduler.timesteps[-desired_steps:]
        
        # Pre-encode text embeddings to reduce latency later
        prompt_embeds, negative_embeds = self._encode_text_batch(prompt, negative_prompt if negative_prompt else "")
        do_cfg = guidance_scale is not None and guidance_scale > 1.0

        # Handle pre-downscaling if requested
        original_w, original_h = image.size
        if pre_downscale_factor > 0 and pre_downscale_factor < 1.0:
            # Downscale the image first, then upscale back to target with latent noise
            intermediate_w = max(16, int(original_w * pre_downscale_factor))
            intermediate_h = max(16, int(original_h * pre_downscale_factor))
            image = image.resize((intermediate_w, intermediate_h), Image.LANCZOS)

        # Preprocess input with latent-space upscaling strategy
        image = image.convert("RGB")
        
        # Apply input conditioning (sharpening/desaturation)
        if sharpen_input > 0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.0 + sharpen_input)
        
        if desaturate_input > 0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.0 - desaturate_input)
        
        # Check if we need latent upsampling (only if there's actual scaling happening)
        current_w, current_h = image.size
        needs_latent_upsampling = (target_w != current_w or target_h != current_h)
        
        if needs_latent_upsampling:
            # Encode current image to latents, then upsample latents instead of pixels
            image_tensor_small = video_processor.preprocess(image, height=current_h, width=current_w).to(exec_device, vae.dtype)
            
            with torch.no_grad():
                # Convert to proper 5D video tensor for VAE
                video_condition = image_tensor_small.unsqueeze(2)  # [B, C, 1, H, W]
                small_latents = vae.encode(video_condition).latent_dist.mode()
            
            # Calculate target latent dimensions
            target_latent_h = target_h // vae_sf
            target_latent_w = target_w // vae_sf
            
            # Upsample latents using nearest neighbor (preserves sharp boundaries) + add noise to missing info
            import torch.nn.functional as F
            # small_latents is [B, C, T, H, W], we need to upsample the spatial dims (H, W)
            # Reshape to [B*C*T, H, W] for interpolation, then reshape back
            b, c, t, h, w = small_latents.shape
            small_latents_2d = small_latents.view(b * c * t, h, w).unsqueeze(1)  # [B*C*T, 1, H, W]
            
            upsampled_2d = F.interpolate(
                small_latents_2d, 
                size=(target_latent_h, target_latent_w), 
                mode='nearest-exact',
                #align_corners=False
            )
            
            # Reshape back to 5D: [B*C*T, 1, H', W'] -> [B, C, T, H', W']
            upsampled_latents = upsampled_2d.squeeze(1).view(b, c, t, target_latent_h, target_latent_w)
            
            # Add structured noise to the upsampled regions to give the transformer something to work with
            upsampled_noise = torch.randn_like(upsampled_latents) * strength  # Use strength parameter directly
            upsampled_latents = upsampled_latents + upsampled_noise
            
            # Use the upsampled latents as our starting point
            latents_from_upsampling = upsampled_latents.to(device).to(torch.float32)
        else:
            # No scaling needed, skip latent upsampling
            latents_from_upsampling = None
        
        # For image conditioning, we need the target-size image tensor
        image_tensor = video_processor.preprocess(
            image.resize((target_w, target_h), Image.LANCZOS) if needs_latent_upsampling else image, 
            height=target_h, width=target_w
        ).to(exec_device, vae.dtype)
        # Always use standard prepare_latents for consistency (handles both single and multi-frame)
        z_dim = int(getattr(vae.config, "z_dim", 16))
        latent_h = target_h // vae_sf
        latent_w = target_w // vae_sf
        
        # Stats for standardization/destandardization
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device, torch.float32)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device, torch.float32)


        # Single frame latent handling (simplified)
        if needs_latent_upsampling and latents_from_upsampling is not None:
            # Use upsampled latents directly
            latents = latents_from_upsampling
            latents = (latents - latents_mean) * latents_std
            
            # For image conditioning, encode the target-size image
            with torch.no_grad():
                # Convert to 5D video tensor for VAE
                video_condition = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
                cond_latents = vae.encode(video_condition).latent_dist.mode()
        else:
            # Encode the image once and reuse for both latents and conditioning
            with torch.no_grad():
                # Convert to 5D video tensor for VAE
                video_condition = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
                encoded_single = vae.encode(video_condition).latent_dist.mode()
            
            # Use encoded latents directly (single frame only)
            latents = encoded_single
            cond_latents = encoded_single.clone()
                
            latents = latents.to(device=device, dtype=torch.float32)
            latents = (latents - latents_mean) * latents_std
        
        # Build image conditioning from the already encoded latents
        cond_latents = (cond_latents.to(device=latents_mean.device, dtype=latents_mean.dtype) - latents_mean) * latents_std
        mask_channels = torch.ones(
            1,
            int(getattr(self, "vae_scale_factor_temporal", 4)),
            1,  # Always single frame: T=1
            latents.shape[-2],
            latents.shape[-1],
            dtype=cond_latents.dtype,
            device=cond_latents.device,
        )
        image_condition = torch.cat([mask_channels, cond_latents], dim=1).to(device=device, dtype=dtype)

        # Keep a copy of standardized original latents for consistency
        original_latents = latents.clone()
        original_latents_exec = original_latents.to(device=exec_device, dtype=torch.float32)

        # Noise injection based on strength
        if s > 1e-4:
            t0 = timesteps[0]
            latents = latents.to(device=exec_device, dtype=torch.float32)
            noise = torch.randn_like(latents, dtype=torch.float32, device=exec_device)
            if hasattr(scheduler, "add_noise"):
                if not isinstance(t0, torch.Tensor):
                    t0 = torch.tensor(t0, device=exec_device)
                if t0.ndim == 0:
                    t0 = t0.expand(latents.shape[0])
                latents = scheduler.add_noise(latents, noise, t0)
            else:
                latents = latents + noise
        else:
            # Micro-perturbation helps recover micro-detail without drifting far
            latents = latents.to(device=exec_device, dtype=torch.float32)
            latents = latents + torch.randn_like(latents, device=exec_device) * 0.003


        # Denoising loop
        iterator = tqdm(range(desired_steps), total=desired_steps, desc="Denoising", dynamic_ncols=True) if tqdm is not None else range(desired_steps)
        with torch.inference_mode():
            for i in iterator:
                t = timesteps[i]

                # Use pipeline execution device for offloading compatibility
                exec_device = getattr(self, "_execution_device", self.device)
                target_dtype = self._target_dtype
                
                latents_for_model = latents.to(device=exec_device, dtype=target_dtype)
                if hasattr(scheduler, "scale_model_input"):
                    latents_for_model = scheduler.scale_model_input(latents_for_model, t)
                    # Enforce dtype/device again in case scheduler returned float32 or moved tensors
                    latents_for_model = latents_for_model.to(device=exec_device, dtype=target_dtype)
                timestep = t.expand(latents.shape[0]).to(device=exec_device)

                with transformer.cache_context("cond"):
                    # Check for image conditioning availability
                    can_image_cond = hasattr(transformer, "condition_embedder") and getattr(
                        transformer.condition_embedder, "image_embedder", None
                    ) is not None

                    # All tensors should already be on correct device/dtype from above
                    forward_kwargs = dict(
                        hidden_states=latents_for_model,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds.to(device=exec_device, dtype=target_dtype),
                        return_dict=False,
                    )
                    if can_image_cond:
                        forward_kwargs["encoder_hidden_states_image"] = image_condition.to(device=exec_device, dtype=target_dtype)
                    
                    noise_pred = transformer(**forward_kwargs)[0]
                    
                    if do_cfg:
                        uncond_kwargs = dict(
                            hidden_states=latents_for_model,
                            timestep=timestep,
                            encoder_hidden_states=negative_embeds.to(device=exec_device, dtype=target_dtype),
                            return_dict=False,
                        )
                        if can_image_cond:
                            uncond_kwargs["encoder_hidden_states_image"] = image_condition.to(device=exec_device, dtype=target_dtype)
                        noise_uncond = transformer(**uncond_kwargs)[0]
                        noise_pred = noise_uncond + float(guidance_scale) * (noise_pred - noise_uncond)

                latents = scheduler.step(noise_pred, t, latents_for_model, return_dict=False)[0]

                # Light pull-back to prevent washout (keep on execution device)
                latents = latents.to(dtype=torch.float32, device=exec_device)
                latents = latents - 0.02 * (latents - original_latents_exec)

        # Decode to image (destandardize latents). Use execution device so accelerate can onload the VAE.
        latents = latents.to(device=exec_device, dtype=vae.dtype)
        latents_std = latents_std.to(device=exec_device)
        latents_mean = latents_mean.to(device=exec_device)
        latents = latents / latents_std + latents_mean
        
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]
        frames = video_processor.postprocess_video(video, output_type="pil")
        
        return frames[0][0]  # Always return single image


def upscale_image(
    image: Image.Image,
    scale: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    prompt: str = "",
    negative_prompt: str = "",
    num_inference_steps: int = 40,
    guidance_scale: float = 1.2,
    strength: float = 0.8,
    sharpen_input: float = 0.0,
    desaturate_input: float = 0.0,
    pre_downscale_factor: float = 1.0,
) -> Image.Image:
    pipe = WanLowNoiseUpscalePipeline.from_pretrained()
    return pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        scale=scale,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        sharpen_input=sharpen_input,
        desaturate_input=desaturate_input,
        pre_downscale_factor=pre_downscale_factor,
    )


if __name__ == "__main__":
    pass
