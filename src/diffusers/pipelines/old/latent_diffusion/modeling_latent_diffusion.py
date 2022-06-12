import tqdm
import torch

from diffusers import DiffusionPipeline

# add these relative imports here, so we can load from hub
from .modeling_vae import AutoencoderKL # NOQA
from .configuration_ldmbert import LDMBertConfig # NOQA
from .modeling_ldmbert import LDMBertModel # NOQA

class LatentDiffusion(DiffusionPipeline):
    def __init__(self, vqvae, bert, tokenizer, unet, noise_scheduler):
        super().__init__()
        self.register_modules(vqvae=vqvae, bert=bert, tokenizer=tokenizer, unet=unet, noise_scheduler=noise_scheduler)

    @torch.no_grad()
    def __call__(self, prompt, batch_size=1, generator=None, torch_device=None, eta=0.0, guidance_scale=1.0, num_inference_steps=50):
        # eta corresponds to η in paper and should be between [0, 1]

        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)
        self.vqvae.to(torch_device)
        self.bert.to(torch_device)
        
        # get unconditional embeddings for classifier free guidence
        if guidance_scale != 1.0:
            uncond_input = self.tokenizer([""], padding="max_length", max_length=77, return_tensors='pt').to(torch_device)
            uncond_embeddings = self.bert(uncond_input.input_ids)[0]
        
        # get text embedding
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors='pt').to(torch_device)
        text_embedding = self.bert(text_input.input_ids)[0]
        
        num_trained_timesteps = self.noise_scheduler.timesteps
        inference_step_times = range(0, num_trained_timesteps, num_trained_timesteps // num_inference_steps)

        image = self.noise_scheduler.sample_noise(
            (batch_size, self.unet.in_channels, self.unet.image_size, self.unet.image_size),
            device=torch_device,
            generator=generator,
        )
        
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_image -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_image_direction -> "direction pointingc to x_t"
        # - pred_prev_image -> "x_t-1"
        for t in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
            # guidance_scale of 1 means no guidance
            if guidance_scale == 1.0:
                image_in = image
                context = text_embedding
                timesteps = torch.tensor([inference_step_times[t]] * image.shape[0], device=torch_device)
            else:
                # for classifier free guidance, we need to do two forward passes
                # here we concanate embedding and unconditioned embedding in a single batch 
                # to avoid doing two forward passes
                image_in = torch.cat([image] * 2)
                context = torch.cat([uncond_embeddings, text_embedding])
                timesteps = torch.tensor([inference_step_times[t]] * image.shape[0], device=torch_device)

            # 1. predict noise residual
            pred_noise_t = self.unet(image_in, timesteps, context=context)
            
            # perform guidance
            if guidance_scale != 1.0:
                pred_noise_t_uncond, pred_noise_t = pred_noise_t.chunk(2)
                pred_noise_t = pred_noise_t_uncond + guidance_scale * (pred_noise_t - pred_noise_t_uncond)
                    
            # 2. predict previous mean of image x_t-1
            pred_prev_image = self.noise_scheduler.step(pred_noise_t, image, t, num_inference_steps, eta)

            # 3. optionally sample variance
            variance = 0
            if eta > 0:
                noise = self.noise_scheduler.sample_noise(image.shape, device=image.device, generator=generator)
                variance = self.noise_scheduler.get_variance(t, num_inference_steps).sqrt() * eta * noise

            # 4. set current image to prev_image: x_t -> x_t-1
            image = pred_prev_image + variance

        # scale and decode image with vae
        image = 1 /  0.18215 * image
        image = self.vqvae.decode(image)
        image = torch.clamp((image+1.0)/2.0, min=0.0, max=1.0)

        return image
