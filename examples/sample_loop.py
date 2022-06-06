#!/usr/bin/env python3
from diffusers import UNetModel, GaussianDDPMScheduler
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
import tqdm

#torch_device = "cuda"
#
#unet = UNetModel.from_pretrained("/home/patrick/ddpm-lsun-church")
#unet.to(torch_device)
#
#TIME_STEPS = 10
#
#scheduler = GaussianDDPMScheduler.from_config("/home/patrick/ddpm-lsun-church", timesteps=TIME_STEPS)
#
#diffusion_config = {
#    "beta_start": 0.0001,
#    "beta_end": 0.02,
#    "num_diffusion_timesteps": TIME_STEPS,
#}
#
# 2. Do one denoising step with model
#batch_size, num_channels, height, width = 1, 3, 256, 256
#
#torch.manual_seed(0)
#noise_image = torch.randn(batch_size, num_channels, height, width, device="cuda")
#
#
# Helper
#def noise_like(shape, device, repeat=False):
#    def repeat_noise():
#        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
#
#    def noise():
#        return torch.randn(shape, device=device)
#
#    return repeat_noise() if repeat else noise()
#
#
#betas = np.linspace(diffusion_config["beta_start"], diffusion_config["beta_end"], diffusion_config["num_diffusion_timesteps"], dtype=np.float64)
#betas = torch.tensor(betas, device=torch_device)
#alphas = 1.0 - betas
#
#alphas_cumprod = torch.cumprod(alphas, axis=0)
#alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
#
#posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
#posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
#
#posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
#posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
#
#
#sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
#sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
#
#
#noise_coeff = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
#coeff = 1 / torch.sqrt(alphas)


def real_fn():
    # Compare the following to Algorithm 2 Sampling of paper: https://arxiv.org/pdf/2006.11239.pdf
    # 1: x_t ~ N(0,1)
    x_t = noise_image
    # 2: for t = T, ...., 1 do
    for i in reversed(range(TIME_STEPS)):
        t = torch.tensor([i]).to(torch_device)
        # 3: z ~ N(0, 1)
        noise = noise_like(x_t.shape, torch_device)

        # 4:  √1αtxt − √1−αt1−α¯tθ(xt, t) + σtz
        # ------------------------- MODEL ------------------------------------#
        with torch.no_grad():
            pred_noise = unet(x_t, t)  # pred epsilon_theta

    #    pred_x = sqrt_recip_alphas_cumprod[t] * x_t - sqrt_recipm1_alphas_cumprod[t] * pred_noise
    #    pred_x.clamp_(-1.0, 1.0)
        # pred mean
    #    posterior_mean = posterior_mean_coef1[t] * pred_x + posterior_mean_coef2[t] * x_t
        # --------------------------------------------------------------------#

        posterior_mean = coeff[t] * (x_t - noise_coeff[t] * pred_noise)

        # ------------------------- Variance Scheduler -----------------------#
        # pred variance
        posterior_log_variance = posterior_log_variance_clipped[t]

        b, *_, device = *x_t.shape, x_t.device
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        posterior_variance = nonzero_mask * (0.5 * posterior_log_variance).exp()
        # --------------------------------------------------------------------#

        x_t_1 = (posterior_mean + posterior_variance * noise).to(torch.float32)
        x_t = x_t_1

        print(x_t.abs().sum())


def post_process_to_image(x_t):
    image = x_t.cpu().permute(0, 2, 3, 1)
    image = (image + 1.0) * 127.5
    image = image.numpy().astype(np.uint8)

    return PIL.Image.fromarray(image[0])


from pytorch_diffusion import Diffusion

#diffusion = Diffusion.from_pretrained("lsun_church")
#samples = diffusion.denoise(1)
#
#image = post_process_to_image(samples)
#image.save("check.png")
#import ipdb; ipdb.set_trace()


device = "cuda"
scheduler = GaussianDDPMScheduler.from_config("/home/patrick/ddpm-lsun-church", timesteps=10)

import ipdb; ipdb.set_trace()

model = UNetModel.from_pretrained("/home/patrick/ddpm-lsun-church").to(device)


torch.manual_seed(0)
next_image = scheduler.sample_noise((1, model.in_channels, model.resolution, model.resolution), device=device)

for t in tqdm.tqdm(reversed(range(len(scheduler))), total=len(scheduler)):
    # define coefficients for time step t
    clip_image_coeff = 1 / torch.sqrt(scheduler.get_alpha_prod(t))
    clip_noise_coeff = torch.sqrt(1 / scheduler.get_alpha_prod(t) - 1)
    image_coeff = (1 - scheduler.get_alpha_prod(t - 1)) * torch.sqrt(scheduler.get_alpha(t)) / (1 - scheduler.get_alpha_prod(t))
    clip_coeff = torch.sqrt(scheduler.get_alpha_prod(t - 1)) * scheduler.get_beta(t) / (1 - scheduler.get_alpha_prod(t))

    # predict noise residual
    with torch.no_grad():
        noise_residual = model(next_image, t)

    # compute prev image from noise
    pred_mean = clip_image_coeff * next_image - clip_noise_coeff * noise_residual
    pred_mean = torch.clamp(pred_mean, -1, 1)
    image = clip_coeff * pred_mean + image_coeff * next_image

    # sample variance
    variance = scheduler.sample_variance(t, image.shape, device=device)

    # sample previous image
    sampled_image = image + variance

    next_image = sampled_image


image = post_process_to_image(next_image)
image.save("example_new.png")
