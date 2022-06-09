# Copyright 2022 The HuggingFace Team. All rights reserved.
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


import numpy as np
import torch

import tqdm
from diffusers import ClassifierFreeGuidanceScheduler, GlideDDIMScheduler, CLIPTextModel, DiffusionPipeline, GLIDETextToImageUNetModel, GLIDESuperResUNetModel
from transformers import GPT2Tokenizer


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


class GLIDE(DiffusionPipeline):
    def __init__(
        self,
        text_unet: GLIDETextToImageUNetModel,
        text_noise_scheduler: ClassifierFreeGuidanceScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: GPT2Tokenizer,
        upscale_unet: GLIDESuperResUNetModel,
        upscale_noise_scheduler: GlideDDIMScheduler
    ):
        super().__init__()
        self.register_modules(
            text_unet=text_unet, text_noise_scheduler=text_noise_scheduler, text_encoder=text_encoder, tokenizer=tokenizer,
            upscale_unet=upscale_unet, upscale_noise_scheduler=upscale_noise_scheduler
        )

    def q_posterior_mean_variance(self, scheduler, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(scheduler.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(scheduler.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(scheduler.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            scheduler.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, scheduler, x, t, transformer_out=None, low_res=None, clip_denoised=True):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B, C = x.shape[:2]
        assert t.shape == (B,)
        if transformer_out is None:
            # super-res model
            model_output = model(x, t, low_res)
        else:
            # text2image model
            model_output = model(x, t, transformer_out)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = _extract_into_tensor(scheduler.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(scheduler.betas), t, x.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        pred_xstart = self._predict_xstart_from_eps(scheduler, x_t=x, t=t, eps=model_output)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(scheduler, x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return model_mean, model_variance, model_log_variance, pred_xstart

    def _predict_xstart_from_eps(self, scheduler, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(scheduler.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(scheduler.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, scheduler, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(scheduler.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(scheduler.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def __call__(self, prompt, generator=None, torch_device=None):
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_unet.to(torch_device)
        self.text_encoder.to(torch_device)
        self.upscale_unet.to(torch_device)

        # Create a classifier-free guidance sampling function
        guidance_scale = 3.0

        def text_model_fn(x_t, ts, transformer_out, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.text_unet(combined, ts, transformer_out, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        # 1. Sample gaussian noise
        batch_size = 2  # second image is empty for classifier-free guidance
        image = self.text_noise_scheduler.sample_noise(
            (batch_size, self.text_unet.in_channels, 64, 64), device=torch_device, generator=generator
        )

        # 2. Encode tokens
        # an empty input is needed to guide the model away from (
        inputs = self.tokenizer([prompt, ""], padding="max_length", max_length=128, return_tensors="pt")
        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)
        transformer_out = self.text_encoder(input_ids, attention_mask).last_hidden_state

        # 3. Run the text2image generation step
        num_timesteps = len(self.text_noise_scheduler)
        for i in tqdm.tqdm(reversed(range(num_timesteps)), total=num_timesteps):
            t = torch.tensor([i] * image.shape[0], device=torch_device)
            mean, variance, log_variance, pred_xstart = self.p_mean_variance(
                text_model_fn, self.text_noise_scheduler, image, t, transformer_out=transformer_out
            )
            noise = self.text_noise_scheduler.sample_noise(image.shape, device=torch_device, generator=generator)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(image.shape) - 1)))  # no noise when t == 0
            image = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

        # 4. Run the upscaling step
        batch_size = 1
        image = image[:1]
        low_res = ((image + 1) * 127.5).round() / 127.5 - 1
        eta = 0.0

        # Tune this parameter to control the sharpness of 256x256 images.
        # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
        upsample_temp = 0.997

        image = self.upscale_noise_scheduler.sample_noise(
            (batch_size, 3, 256, 256), device=torch_device, generator=generator
        ) * upsample_temp

        num_timesteps = len(self.upscale_noise_scheduler)
        for t in tqdm.tqdm(reversed(range(len(self.upscale_noise_scheduler))), total=len(self.upscale_noise_scheduler)):
            # i) define coefficients for time step t
            clipped_image_coeff = 1 / torch.sqrt(self.upscale_noise_scheduler.get_alpha_prod(t))
            clipped_noise_coeff = torch.sqrt(1 / self.upscale_noise_scheduler.get_alpha_prod(t) - 1)
            image_coeff = (1 - self.upscale_noise_scheduler.get_alpha_prod(t - 1)) * torch.sqrt(
                self.upscale_noise_scheduler.get_alpha(t)) / (1 - self.upscale_noise_scheduler.get_alpha_prod(t))
            clipped_coeff = torch.sqrt(self.upscale_noise_scheduler.get_alpha_prod(t - 1)) * self.upscale_noise_scheduler.get_beta(
                t) / (1 - self.upscale_noise_scheduler.get_alpha_prod(t))

            # ii) predict noise residual
            time_input = torch.tensor([t] * image.shape[0], device=torch_device)
            model_output = self.upscale_unet(image, time_input, low_res)
            noise_residual, pred_variance = torch.split(model_output, 3, dim=1)

            # iii) compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            pred_mean = clipped_image_coeff * image - clipped_noise_coeff * noise_residual
            pred_mean = torch.clamp(pred_mean, -1, 1)
            prev_image = clipped_coeff * pred_mean + image_coeff * image

            # iv) sample variance
            prev_variance = self.upscale_noise_scheduler.sample_variance(t, prev_image.shape, device=torch_device,
                                                                 generator=generator)

            # v) sample  x_{t-1} ~ N(prev_image, prev_variance)
            sampled_prev_image = prev_image + prev_variance
            image = sampled_prev_image

        image = image[0].permute(1, 2, 0)

        return image
