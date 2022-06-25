#!/usr/bin/env python3
import numpy as np
import torch

import PIL
from diffusers import DiffusionPipeline


# from configs.ve import ffhq_ncsnpp_continuous as configs
#  from configs.ve import cifar10_ncsnpp_continuous as configs

#  ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
# ckpt_filename = "exp/ve/ffhq_1024_ncsnpp_continuous/checkpoint_60.pth"
# Note usually we need to restore ema etc...
# ema restored checkpoint used from below
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)


class NCSNppPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def __call__(self, generator=None):
        N = self.scheduler.config.N
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_size = self.model.config.image_size
        channels = self.model.config.num_channels
        shape = (1, channels, img_size, img_size)

        model = torch.nn.DataParallel(self.model.to(device))

        centered = False
        n_steps = 1

        # Initial sample
        x = torch.randn(*shape) * self.scheduler.config.sigma_max
        x = x.to(device)

        for i in range(N):
            sigma_t = self.scheduler.get_sigma_t(i) * torch.ones(shape[0], device=device)

            for _ in range(n_steps):
                with torch.no_grad():
                    result = model(x, sigma_t)
                x = self.scheduler.step_correct(result, x)

            with torch.no_grad():
                result = model(x, sigma_t)

            x, x_mean = self.scheduler.step_pred(result, x, i)

        x = x_mean

        if centered:
            x = (x + 1.0) / 2.0

        return x


pipeline = NCSNppPipeline.from_pretrained("/home/patrick/ffhq_ncsnpp")
x = pipeline()


# for 5 cifar10
# x_sum = 106071.9922
# x_mean = 34.52864456176758

# for 1000 cifar10
# x_sum = 461.9700
# x_mean = 0.1504

# for N=2 for 1024
x_sum = 3382810112.0
x_mean = 1075.366455078125


def check_x_sum_x_mean(x, x_sum, x_mean):
    assert (x.abs().sum() - x_sum).abs().cpu().item() < 1e-2, f"sum wrong {x.abs().sum()}"
    assert (x.abs().mean() - x_mean).abs().cpu().item() < 1e-4, f"mean wrong {x.abs().mean()}"


check_x_sum_x_mean(x, x_sum, x_mean)


def save_image(x):
    image_processed = np.clip(x.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    image_pil = PIL.Image.fromarray(image_processed[0])
    image_pil.save("../images/hey.png")


# save_image(x)
