import math
import os
import urllib
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.utils import insecure_hashlib
from safetensors.torch import load_file as stl
from tqdm import tqdm

from diffusers import AutoencoderKL, ConsistencyDecoderVAE, DiffusionPipeline, StableDiffusionPipeline, UNet2DModel
from diffusers.models.autoencoders.vae import Encoder
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.unets.unet_2d_blocks import ResnetDownsampleBlock2D, ResnetUpsampleBlock2D, UNetMidBlock2D


args = ArgumentParser()
args.add_argument("--save_pretrained", required=False, default=None, type=str)
args.add_argument("--test_image", required=True, type=str)
args = args.parse_args()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    # from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L895    """
    res = arr[timesteps].float()
    dims_to_append = len(broadcast_shape) - len(res.shape)
    return res[(...,) + (None,) * dims_to_append]


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    # from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L45
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if insecure_hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if insecure_hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


class ConsistencyDecoder:
    def __init__(self, device="cuda:0", download_root=os.path.expanduser("~/.cache/clip")):
        self.n_distilled_steps = 64
        download_target = _download(
            "https://openaipublic.azureedge.net/diff-vae/c9cebd3132dd9c42936d803e33424145a748843c8f716c0814838bdc8a2fe7cb/decoder.pt",
            download_root,
        )
        self.ckpt = torch.jit.load(download_target).to(device)
        self.device = device
        sigma_data = 0.5
        betas = betas_for_alpha_bar(1024, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)
        self.c_skip = sqrt_recip_alphas_cumprod * sigma_data**2 / (sigmas**2 + sigma_data**2)
        self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
        self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

    @staticmethod
    def round_timesteps(timesteps, total_timesteps, n_distilled_steps, truncate_start=True):
        with torch.no_grad():
            space = torch.div(total_timesteps, n_distilled_steps, rounding_mode="floor")
            rounded_timesteps = (torch.div(timesteps, space, rounding_mode="floor") + 1) * space
            if truncate_start:
                rounded_timesteps[rounded_timesteps == total_timesteps] -= space
            else:
                rounded_timesteps[rounded_timesteps == total_timesteps] -= space
                rounded_timesteps[rounded_timesteps == 0] += space
            return rounded_timesteps

    @staticmethod
    def ldm_transform_latent(z, extra_scale_factor=1):
        channel_means = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
        channel_stds = [0.9654121, 1.0440036, 0.76147926, 0.77022034]

        if len(z.shape) != 4:
            raise ValueError()

        z = z * 0.18215
        channels = [z[:, i] for i in range(z.shape[1])]

        channels = [extra_scale_factor * (c - channel_means[i]) / channel_stds[i] for i, c in enumerate(channels)]
        return torch.stack(channels, dim=1)

    @torch.no_grad()
    def __call__(
        self,
        features: torch.Tensor,
        schedule=[1.0, 0.5],
        generator=None,
    ):
        features = self.ldm_transform_latent(features)
        ts = self.round_timesteps(
            torch.arange(0, 1024),
            1024,
            self.n_distilled_steps,
            truncate_start=False,
        )
        shape = (
            features.size(0),
            3,
            8 * features.size(2),
            8 * features.size(3),
        )
        x_start = torch.zeros(shape, device=features.device, dtype=features.dtype)
        schedule_timesteps = [int((1024 - 1) * s) for s in schedule]
        for i in schedule_timesteps:
            t = ts[i].item()
            t_ = torch.tensor([t] * features.shape[0]).to(self.device)
            # noise = torch.randn_like(x_start)
            noise = torch.randn(x_start.shape, dtype=x_start.dtype, generator=generator).to(device=x_start.device)
            x_start = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t_, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_, x_start.shape) * noise
            )
            c_in = _extract_into_tensor(self.c_in, t_, x_start.shape)

            import torch.nn.functional as F

            from diffusers import UNet2DModel

            if isinstance(self.ckpt, UNet2DModel):
                input = torch.concat([c_in * x_start, F.upsample_nearest(features, scale_factor=8)], dim=1)
                model_output = self.ckpt(input, t_).sample
            else:
                model_output = self.ckpt(c_in * x_start, t_, features=features)

            B, C = x_start.shape[:2]
            model_output, _ = torch.split(model_output, C, dim=1)
            pred_xstart = (
                _extract_into_tensor(self.c_out, t_, x_start.shape) * model_output
                + _extract_into_tensor(self.c_skip, t_, x_start.shape) * x_start
            ).clamp(-1, 1)
            x_start = pred_xstart
        return x_start


def save_image(image, name):
    import numpy as np
    from PIL import Image

    image = image[0].cpu().numpy()
    image = (image + 1.0) * 127.5
    image = image.clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    image.save(name)


def load_image(uri, size=None, center_crop=False):
    import numpy as np
    from PIL import Image

    image = Image.open(uri)
    if center_crop:
        image = image.crop(
            (
                (image.width - min(image.width, image.height)) // 2,
                (image.height - min(image.width, image.height)) // 2,
                (image.width + min(image.width, image.height)) // 2,
                (image.height + min(image.width, image.height)) // 2,
            )
        )
    if size is not None:
        image = image.resize(size)
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0).float()
    image = image / 127.5 - 1.0
    return image


class TimestepEmbedding_(nn.Module):
    def __init__(self, n_time=1024, n_emb=320, n_out=1280) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_time, n_emb)
        self.f_1 = nn.Linear(n_emb, n_out)
        self.f_2 = nn.Linear(n_out, n_out)

    def forward(self, x) -> torch.Tensor:
        x = self.emb(x)
        x = self.f_1(x)
        x = F.silu(x)
        return self.f_2(x)


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels=7, out_channels=320) -> None:
        super().__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)


class ImageUnembedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=6) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))


class ConvResblock(nn.Module):
    def __init__(self, in_features=320, out_features=320) -> None:
        super().__init__()
        self.f_t = nn.Linear(1280, out_features * 2)

        self.gn_1 = nn.GroupNorm(32, in_features)
        self.f_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)

        self.gn_2 = nn.GroupNorm(32, out_features)
        self.f_2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)

        skip_conv = in_features != out_features
        self.f_s = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0) if skip_conv else nn.Identity()

    def forward(self, x, t):
        x_skip = x
        t = self.f_t(F.silu(t))
        t = t.chunk(2, dim=1)
        t_1 = t[0].unsqueeze(dim=2).unsqueeze(dim=3) + 1
        t_2 = t[1].unsqueeze(dim=2).unsqueeze(dim=3)

        gn_1 = F.silu(self.gn_1(x))
        f_1 = self.f_1(gn_1)

        gn_2 = self.gn_2(f_1)

        return self.f_s(x_skip) + self.f_2(F.silu(gn_2 * t_1 + t_2))


# Also ConvResblock
class Downsample(nn.Module):
    def __init__(self, in_channels=320) -> None:
        super().__init__()
        self.f_t = nn.Linear(1280, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        avg_pool2d = F.avg_pool2d(gn_1, kernel_size=(2, 2), stride=None)

        f_1 = self.f_1(avg_pool2d)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.avg_pool2d(x_skip, kernel_size=(2, 2), stride=None)


# Also ConvResblock
class Upsample(nn.Module):
    def __init__(self, in_channels=1024) -> None:
        super().__init__()
        self.f_t = nn.Linear(1280, in_channels * 2)

        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.f_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gn_2 = nn.GroupNorm(32, in_channels)

        self.f_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        x_skip = x

        t = self.f_t(F.silu(t))
        t_1, t_2 = t.chunk(2, dim=1)
        t_1 = t_1.unsqueeze(2).unsqueeze(3) + 1
        t_2 = t_2.unsqueeze(2).unsqueeze(3)

        gn_1 = F.silu(self.gn_1(x))
        upsample = F.upsample_nearest(gn_1, scale_factor=2)
        f_1 = self.f_1(upsample)
        gn_2 = self.gn_2(f_1)

        f_2 = self.f_2(F.silu(t_2 + (t_1 * gn_2)))

        return f_2 + F.upsample_nearest(x_skip, scale_factor=2)


class ConvUNetVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_image = ImageEmbedding()
        self.embed_time = TimestepEmbedding_()

        down_0 = nn.ModuleList(
            [
                ConvResblock(320, 320),
                ConvResblock(320, 320),
                ConvResblock(320, 320),
                Downsample(320),
            ]
        )
        down_1 = nn.ModuleList(
            [
                ConvResblock(320, 640),
                ConvResblock(640, 640),
                ConvResblock(640, 640),
                Downsample(640),
            ]
        )
        down_2 = nn.ModuleList(
            [
                ConvResblock(640, 1024),
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
                Downsample(1024),
            ]
        )
        down_3 = nn.ModuleList(
            [
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
            ]
        )
        self.down = nn.ModuleList(
            [
                down_0,
                down_1,
                down_2,
                down_3,
            ]
        )

        self.mid = nn.ModuleList(
            [
                ConvResblock(1024, 1024),
                ConvResblock(1024, 1024),
            ]
        )

        up_3 = nn.ModuleList(
            [
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                Upsample(1024),
            ]
        )
        up_2 = nn.ModuleList(
            [
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 * 2, 1024),
                ConvResblock(1024 + 640, 1024),
                Upsample(1024),
            ]
        )
        up_1 = nn.ModuleList(
            [
                ConvResblock(1024 + 640, 640),
                ConvResblock(640 * 2, 640),
                ConvResblock(640 * 2, 640),
                ConvResblock(320 + 640, 640),
                Upsample(640),
            ]
        )
        up_0 = nn.ModuleList(
            [
                ConvResblock(320 + 640, 320),
                ConvResblock(320 * 2, 320),
                ConvResblock(320 * 2, 320),
                ConvResblock(320 * 2, 320),
            ]
        )
        self.up = nn.ModuleList(
            [
                up_0,
                up_1,
                up_2,
                up_3,
            ]
        )

        self.output = ImageUnembedding()

    def forward(self, x, t, features) -> torch.Tensor:
        converted = hasattr(self, "converted") and self.converted

        x = torch.cat([x, F.upsample_nearest(features, scale_factor=8)], dim=1)

        if converted:
            t = self.time_embedding(self.time_proj(t))
        else:
            t = self.embed_time(t)

        x = self.embed_image(x)

        skips = [x]
        for i, down in enumerate(self.down):
            if converted and i in [0, 1, 2, 3]:
                x, skips_ = down(x, t)
                for skip in skips_:
                    skips.append(skip)
            else:
                for block in down:
                    x = block(x, t)
                    skips.append(x)
            print(x.float().abs().sum())

        if converted:
            x = self.mid(x, t)
        else:
            for i in range(2):
                x = self.mid[i](x, t)
        print(x.float().abs().sum())

        for i, up in enumerate(self.up[::-1]):
            if converted and i in [0, 1, 2, 3]:
                skip_4 = skips.pop()
                skip_3 = skips.pop()
                skip_2 = skips.pop()
                skip_1 = skips.pop()
                skips_ = (skip_1, skip_2, skip_3, skip_4)
                x = up(x, skips_, t)
            else:
                for block in up:
                    if isinstance(block, ConvResblock):
                        x = torch.concat([x, skips.pop()], dim=1)
                    x = block(x, t)

        return self.output(x)


def rename_state_dict_key(k):
    k = k.replace("blocks.", "")
    for i in range(5):
        k = k.replace(f"down_{i}_", f"down.{i}.")
        k = k.replace(f"conv_{i}.", f"{i}.")
        k = k.replace(f"up_{i}_", f"up.{i}.")
        k = k.replace(f"mid_{i}", f"mid.{i}")
    k = k.replace("upsamp.", "4.")
    k = k.replace("downsamp.", "3.")
    k = k.replace("f_t.w", "f_t.weight").replace("f_t.b", "f_t.bias")
    k = k.replace("f_1.w", "f_1.weight").replace("f_1.b", "f_1.bias")
    k = k.replace("f_2.w", "f_2.weight").replace("f_2.b", "f_2.bias")
    k = k.replace("f_s.w", "f_s.weight").replace("f_s.b", "f_s.bias")
    k = k.replace("f.w", "f.weight").replace("f.b", "f.bias")
    k = k.replace("gn_1.g", "gn_1.weight").replace("gn_1.b", "gn_1.bias")
    k = k.replace("gn_2.g", "gn_2.weight").replace("gn_2.b", "gn_2.bias")
    k = k.replace("gn.g", "gn.weight").replace("gn.b", "gn.bias")
    return k


def rename_state_dict(sd, embedding):
    sd = {rename_state_dict_key(k): v for k, v in sd.items()}
    sd["embed_time.emb.weight"] = embedding["weight"]
    return sd


# encode with stable diffusion vae
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.vae.cuda()

# construct original decoder with jitted model
decoder_consistency = ConsistencyDecoder(device="cuda:0")

# construct UNet code, overwrite the decoder with conv_unet_vae
model = ConvUNetVAE()
model.load_state_dict(
    rename_state_dict(
        stl("consistency_decoder.safetensors"),
        stl("embedding.safetensors"),
    )
)
model = model.cuda()

decoder_consistency.ckpt = model

image = load_image(args.test_image, size=(256, 256), center_crop=True)
latent = pipe.vae.encode(image.half().cuda()).latent_dist.sample()

# decode with gan
sample_gan = pipe.vae.decode(latent).sample.detach()
save_image(sample_gan, "gan.png")

# decode with conv_unet_vae
sample_consistency_orig = decoder_consistency(latent, generator=torch.Generator("cpu").manual_seed(0))
save_image(sample_consistency_orig, "con_orig.png")


########### conversion

print("CONVERSION")

print("DOWN BLOCK ONE")

block_one_sd_orig = model.down[0].state_dict()
block_one_sd_new = {}

for i in range(3):
    block_one_sd_new[f"resnets.{i}.norm1.weight"] = block_one_sd_orig.pop(f"{i}.gn_1.weight")
    block_one_sd_new[f"resnets.{i}.norm1.bias"] = block_one_sd_orig.pop(f"{i}.gn_1.bias")
    block_one_sd_new[f"resnets.{i}.conv1.weight"] = block_one_sd_orig.pop(f"{i}.f_1.weight")
    block_one_sd_new[f"resnets.{i}.conv1.bias"] = block_one_sd_orig.pop(f"{i}.f_1.bias")
    block_one_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_one_sd_orig.pop(f"{i}.f_t.weight")
    block_one_sd_new[f"resnets.{i}.time_emb_proj.bias"] = block_one_sd_orig.pop(f"{i}.f_t.bias")
    block_one_sd_new[f"resnets.{i}.norm2.weight"] = block_one_sd_orig.pop(f"{i}.gn_2.weight")
    block_one_sd_new[f"resnets.{i}.norm2.bias"] = block_one_sd_orig.pop(f"{i}.gn_2.bias")
    block_one_sd_new[f"resnets.{i}.conv2.weight"] = block_one_sd_orig.pop(f"{i}.f_2.weight")
    block_one_sd_new[f"resnets.{i}.conv2.bias"] = block_one_sd_orig.pop(f"{i}.f_2.bias")

block_one_sd_new["downsamplers.0.norm1.weight"] = block_one_sd_orig.pop("3.gn_1.weight")
block_one_sd_new["downsamplers.0.norm1.bias"] = block_one_sd_orig.pop("3.gn_1.bias")
block_one_sd_new["downsamplers.0.conv1.weight"] = block_one_sd_orig.pop("3.f_1.weight")
block_one_sd_new["downsamplers.0.conv1.bias"] = block_one_sd_orig.pop("3.f_1.bias")
block_one_sd_new["downsamplers.0.time_emb_proj.weight"] = block_one_sd_orig.pop("3.f_t.weight")
block_one_sd_new["downsamplers.0.time_emb_proj.bias"] = block_one_sd_orig.pop("3.f_t.bias")
block_one_sd_new["downsamplers.0.norm2.weight"] = block_one_sd_orig.pop("3.gn_2.weight")
block_one_sd_new["downsamplers.0.norm2.bias"] = block_one_sd_orig.pop("3.gn_2.bias")
block_one_sd_new["downsamplers.0.conv2.weight"] = block_one_sd_orig.pop("3.f_2.weight")
block_one_sd_new["downsamplers.0.conv2.bias"] = block_one_sd_orig.pop("3.f_2.bias")

assert len(block_one_sd_orig) == 0

block_one = ResnetDownsampleBlock2D(
    in_channels=320,
    out_channels=320,
    temb_channels=1280,
    num_layers=3,
    add_downsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

block_one.load_state_dict(block_one_sd_new)

print("DOWN BLOCK TWO")

block_two_sd_orig = model.down[1].state_dict()
block_two_sd_new = {}

for i in range(3):
    block_two_sd_new[f"resnets.{i}.norm1.weight"] = block_two_sd_orig.pop(f"{i}.gn_1.weight")
    block_two_sd_new[f"resnets.{i}.norm1.bias"] = block_two_sd_orig.pop(f"{i}.gn_1.bias")
    block_two_sd_new[f"resnets.{i}.conv1.weight"] = block_two_sd_orig.pop(f"{i}.f_1.weight")
    block_two_sd_new[f"resnets.{i}.conv1.bias"] = block_two_sd_orig.pop(f"{i}.f_1.bias")
    block_two_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_two_sd_orig.pop(f"{i}.f_t.weight")
    block_two_sd_new[f"resnets.{i}.time_emb_proj.bias"] = block_two_sd_orig.pop(f"{i}.f_t.bias")
    block_two_sd_new[f"resnets.{i}.norm2.weight"] = block_two_sd_orig.pop(f"{i}.gn_2.weight")
    block_two_sd_new[f"resnets.{i}.norm2.bias"] = block_two_sd_orig.pop(f"{i}.gn_2.bias")
    block_two_sd_new[f"resnets.{i}.conv2.weight"] = block_two_sd_orig.pop(f"{i}.f_2.weight")
    block_two_sd_new[f"resnets.{i}.conv2.bias"] = block_two_sd_orig.pop(f"{i}.f_2.bias")

    if i == 0:
        block_two_sd_new[f"resnets.{i}.conv_shortcut.weight"] = block_two_sd_orig.pop(f"{i}.f_s.weight")
        block_two_sd_new[f"resnets.{i}.conv_shortcut.bias"] = block_two_sd_orig.pop(f"{i}.f_s.bias")

block_two_sd_new["downsamplers.0.norm1.weight"] = block_two_sd_orig.pop("3.gn_1.weight")
block_two_sd_new["downsamplers.0.norm1.bias"] = block_two_sd_orig.pop("3.gn_1.bias")
block_two_sd_new["downsamplers.0.conv1.weight"] = block_two_sd_orig.pop("3.f_1.weight")
block_two_sd_new["downsamplers.0.conv1.bias"] = block_two_sd_orig.pop("3.f_1.bias")
block_two_sd_new["downsamplers.0.time_emb_proj.weight"] = block_two_sd_orig.pop("3.f_t.weight")
block_two_sd_new["downsamplers.0.time_emb_proj.bias"] = block_two_sd_orig.pop("3.f_t.bias")
block_two_sd_new["downsamplers.0.norm2.weight"] = block_two_sd_orig.pop("3.gn_2.weight")
block_two_sd_new["downsamplers.0.norm2.bias"] = block_two_sd_orig.pop("3.gn_2.bias")
block_two_sd_new["downsamplers.0.conv2.weight"] = block_two_sd_orig.pop("3.f_2.weight")
block_two_sd_new["downsamplers.0.conv2.bias"] = block_two_sd_orig.pop("3.f_2.bias")

assert len(block_two_sd_orig) == 0

block_two = ResnetDownsampleBlock2D(
    in_channels=320,
    out_channels=640,
    temb_channels=1280,
    num_layers=3,
    add_downsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

block_two.load_state_dict(block_two_sd_new)

print("DOWN BLOCK THREE")

block_three_sd_orig = model.down[2].state_dict()
block_three_sd_new = {}

for i in range(3):
    block_three_sd_new[f"resnets.{i}.norm1.weight"] = block_three_sd_orig.pop(f"{i}.gn_1.weight")
    block_three_sd_new[f"resnets.{i}.norm1.bias"] = block_three_sd_orig.pop(f"{i}.gn_1.bias")
    block_three_sd_new[f"resnets.{i}.conv1.weight"] = block_three_sd_orig.pop(f"{i}.f_1.weight")
    block_three_sd_new[f"resnets.{i}.conv1.bias"] = block_three_sd_orig.pop(f"{i}.f_1.bias")
    block_three_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_three_sd_orig.pop(f"{i}.f_t.weight")
    block_three_sd_new[f"resnets.{i}.time_emb_proj.bias"] = block_three_sd_orig.pop(f"{i}.f_t.bias")
    block_three_sd_new[f"resnets.{i}.norm2.weight"] = block_three_sd_orig.pop(f"{i}.gn_2.weight")
    block_three_sd_new[f"resnets.{i}.norm2.bias"] = block_three_sd_orig.pop(f"{i}.gn_2.bias")
    block_three_sd_new[f"resnets.{i}.conv2.weight"] = block_three_sd_orig.pop(f"{i}.f_2.weight")
    block_three_sd_new[f"resnets.{i}.conv2.bias"] = block_three_sd_orig.pop(f"{i}.f_2.bias")

    if i == 0:
        block_three_sd_new[f"resnets.{i}.conv_shortcut.weight"] = block_three_sd_orig.pop(f"{i}.f_s.weight")
        block_three_sd_new[f"resnets.{i}.conv_shortcut.bias"] = block_three_sd_orig.pop(f"{i}.f_s.bias")

block_three_sd_new["downsamplers.0.norm1.weight"] = block_three_sd_orig.pop("3.gn_1.weight")
block_three_sd_new["downsamplers.0.norm1.bias"] = block_three_sd_orig.pop("3.gn_1.bias")
block_three_sd_new["downsamplers.0.conv1.weight"] = block_three_sd_orig.pop("3.f_1.weight")
block_three_sd_new["downsamplers.0.conv1.bias"] = block_three_sd_orig.pop("3.f_1.bias")
block_three_sd_new["downsamplers.0.time_emb_proj.weight"] = block_three_sd_orig.pop("3.f_t.weight")
block_three_sd_new["downsamplers.0.time_emb_proj.bias"] = block_three_sd_orig.pop("3.f_t.bias")
block_three_sd_new["downsamplers.0.norm2.weight"] = block_three_sd_orig.pop("3.gn_2.weight")
block_three_sd_new["downsamplers.0.norm2.bias"] = block_three_sd_orig.pop("3.gn_2.bias")
block_three_sd_new["downsamplers.0.conv2.weight"] = block_three_sd_orig.pop("3.f_2.weight")
block_three_sd_new["downsamplers.0.conv2.bias"] = block_three_sd_orig.pop("3.f_2.bias")

assert len(block_three_sd_orig) == 0

block_three = ResnetDownsampleBlock2D(
    in_channels=640,
    out_channels=1024,
    temb_channels=1280,
    num_layers=3,
    add_downsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

block_three.load_state_dict(block_three_sd_new)

print("DOWN BLOCK FOUR")

block_four_sd_orig = model.down[3].state_dict()
block_four_sd_new = {}

for i in range(3):
    block_four_sd_new[f"resnets.{i}.norm1.weight"] = block_four_sd_orig.pop(f"{i}.gn_1.weight")
    block_four_sd_new[f"resnets.{i}.norm1.bias"] = block_four_sd_orig.pop(f"{i}.gn_1.bias")
    block_four_sd_new[f"resnets.{i}.conv1.weight"] = block_four_sd_orig.pop(f"{i}.f_1.weight")
    block_four_sd_new[f"resnets.{i}.conv1.bias"] = block_four_sd_orig.pop(f"{i}.f_1.bias")
    block_four_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_four_sd_orig.pop(f"{i}.f_t.weight")
    block_four_sd_new[f"resnets.{i}.time_emb_proj.bias"] = block_four_sd_orig.pop(f"{i}.f_t.bias")
    block_four_sd_new[f"resnets.{i}.norm2.weight"] = block_four_sd_orig.pop(f"{i}.gn_2.weight")
    block_four_sd_new[f"resnets.{i}.norm2.bias"] = block_four_sd_orig.pop(f"{i}.gn_2.bias")
    block_four_sd_new[f"resnets.{i}.conv2.weight"] = block_four_sd_orig.pop(f"{i}.f_2.weight")
    block_four_sd_new[f"resnets.{i}.conv2.bias"] = block_four_sd_orig.pop(f"{i}.f_2.bias")

assert len(block_four_sd_orig) == 0

block_four = ResnetDownsampleBlock2D(
    in_channels=1024,
    out_channels=1024,
    temb_channels=1280,
    num_layers=3,
    add_downsample=False,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

block_four.load_state_dict(block_four_sd_new)


print("MID BLOCK 1")

mid_block_one_sd_orig = model.mid.state_dict()
mid_block_one_sd_new = {}

for i in range(2):
    mid_block_one_sd_new[f"resnets.{i}.norm1.weight"] = mid_block_one_sd_orig.pop(f"{i}.gn_1.weight")
    mid_block_one_sd_new[f"resnets.{i}.norm1.bias"] = mid_block_one_sd_orig.pop(f"{i}.gn_1.bias")
    mid_block_one_sd_new[f"resnets.{i}.conv1.weight"] = mid_block_one_sd_orig.pop(f"{i}.f_1.weight")
    mid_block_one_sd_new[f"resnets.{i}.conv1.bias"] = mid_block_one_sd_orig.pop(f"{i}.f_1.bias")
    mid_block_one_sd_new[f"resnets.{i}.time_emb_proj.weight"] = mid_block_one_sd_orig.pop(f"{i}.f_t.weight")
    mid_block_one_sd_new[f"resnets.{i}.time_emb_proj.bias"] = mid_block_one_sd_orig.pop(f"{i}.f_t.bias")
    mid_block_one_sd_new[f"resnets.{i}.norm2.weight"] = mid_block_one_sd_orig.pop(f"{i}.gn_2.weight")
    mid_block_one_sd_new[f"resnets.{i}.norm2.bias"] = mid_block_one_sd_orig.pop(f"{i}.gn_2.bias")
    mid_block_one_sd_new[f"resnets.{i}.conv2.weight"] = mid_block_one_sd_orig.pop(f"{i}.f_2.weight")
    mid_block_one_sd_new[f"resnets.{i}.conv2.bias"] = mid_block_one_sd_orig.pop(f"{i}.f_2.bias")

assert len(mid_block_one_sd_orig) == 0

mid_block_one = UNetMidBlock2D(
    in_channels=1024,
    temb_channels=1280,
    num_layers=1,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
    add_attention=False,
)

mid_block_one.load_state_dict(mid_block_one_sd_new)

print("UP BLOCK ONE")

up_block_one_sd_orig = model.up[-1].state_dict()
up_block_one_sd_new = {}

for i in range(4):
    up_block_one_sd_new[f"resnets.{i}.norm1.weight"] = up_block_one_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_one_sd_new[f"resnets.{i}.norm1.bias"] = up_block_one_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_one_sd_new[f"resnets.{i}.conv1.weight"] = up_block_one_sd_orig.pop(f"{i}.f_1.weight")
    up_block_one_sd_new[f"resnets.{i}.conv1.bias"] = up_block_one_sd_orig.pop(f"{i}.f_1.bias")
    up_block_one_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_one_sd_orig.pop(f"{i}.f_t.weight")
    up_block_one_sd_new[f"resnets.{i}.time_emb_proj.bias"] = up_block_one_sd_orig.pop(f"{i}.f_t.bias")
    up_block_one_sd_new[f"resnets.{i}.norm2.weight"] = up_block_one_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_one_sd_new[f"resnets.{i}.norm2.bias"] = up_block_one_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_one_sd_new[f"resnets.{i}.conv2.weight"] = up_block_one_sd_orig.pop(f"{i}.f_2.weight")
    up_block_one_sd_new[f"resnets.{i}.conv2.bias"] = up_block_one_sd_orig.pop(f"{i}.f_2.bias")
    up_block_one_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_one_sd_orig.pop(f"{i}.f_s.weight")
    up_block_one_sd_new[f"resnets.{i}.conv_shortcut.bias"] = up_block_one_sd_orig.pop(f"{i}.f_s.bias")

up_block_one_sd_new["upsamplers.0.norm1.weight"] = up_block_one_sd_orig.pop("4.gn_1.weight")
up_block_one_sd_new["upsamplers.0.norm1.bias"] = up_block_one_sd_orig.pop("4.gn_1.bias")
up_block_one_sd_new["upsamplers.0.conv1.weight"] = up_block_one_sd_orig.pop("4.f_1.weight")
up_block_one_sd_new["upsamplers.0.conv1.bias"] = up_block_one_sd_orig.pop("4.f_1.bias")
up_block_one_sd_new["upsamplers.0.time_emb_proj.weight"] = up_block_one_sd_orig.pop("4.f_t.weight")
up_block_one_sd_new["upsamplers.0.time_emb_proj.bias"] = up_block_one_sd_orig.pop("4.f_t.bias")
up_block_one_sd_new["upsamplers.0.norm2.weight"] = up_block_one_sd_orig.pop("4.gn_2.weight")
up_block_one_sd_new["upsamplers.0.norm2.bias"] = up_block_one_sd_orig.pop("4.gn_2.bias")
up_block_one_sd_new["upsamplers.0.conv2.weight"] = up_block_one_sd_orig.pop("4.f_2.weight")
up_block_one_sd_new["upsamplers.0.conv2.bias"] = up_block_one_sd_orig.pop("4.f_2.bias")

assert len(up_block_one_sd_orig) == 0

up_block_one = ResnetUpsampleBlock2D(
    in_channels=1024,
    prev_output_channel=1024,
    out_channels=1024,
    temb_channels=1280,
    num_layers=4,
    add_upsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

up_block_one.load_state_dict(up_block_one_sd_new)

print("UP BLOCK TWO")

up_block_two_sd_orig = model.up[-2].state_dict()
up_block_two_sd_new = {}

for i in range(4):
    up_block_two_sd_new[f"resnets.{i}.norm1.weight"] = up_block_two_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_two_sd_new[f"resnets.{i}.norm1.bias"] = up_block_two_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_two_sd_new[f"resnets.{i}.conv1.weight"] = up_block_two_sd_orig.pop(f"{i}.f_1.weight")
    up_block_two_sd_new[f"resnets.{i}.conv1.bias"] = up_block_two_sd_orig.pop(f"{i}.f_1.bias")
    up_block_two_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_two_sd_orig.pop(f"{i}.f_t.weight")
    up_block_two_sd_new[f"resnets.{i}.time_emb_proj.bias"] = up_block_two_sd_orig.pop(f"{i}.f_t.bias")
    up_block_two_sd_new[f"resnets.{i}.norm2.weight"] = up_block_two_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_two_sd_new[f"resnets.{i}.norm2.bias"] = up_block_two_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_two_sd_new[f"resnets.{i}.conv2.weight"] = up_block_two_sd_orig.pop(f"{i}.f_2.weight")
    up_block_two_sd_new[f"resnets.{i}.conv2.bias"] = up_block_two_sd_orig.pop(f"{i}.f_2.bias")
    up_block_two_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_two_sd_orig.pop(f"{i}.f_s.weight")
    up_block_two_sd_new[f"resnets.{i}.conv_shortcut.bias"] = up_block_two_sd_orig.pop(f"{i}.f_s.bias")

up_block_two_sd_new["upsamplers.0.norm1.weight"] = up_block_two_sd_orig.pop("4.gn_1.weight")
up_block_two_sd_new["upsamplers.0.norm1.bias"] = up_block_two_sd_orig.pop("4.gn_1.bias")
up_block_two_sd_new["upsamplers.0.conv1.weight"] = up_block_two_sd_orig.pop("4.f_1.weight")
up_block_two_sd_new["upsamplers.0.conv1.bias"] = up_block_two_sd_orig.pop("4.f_1.bias")
up_block_two_sd_new["upsamplers.0.time_emb_proj.weight"] = up_block_two_sd_orig.pop("4.f_t.weight")
up_block_two_sd_new["upsamplers.0.time_emb_proj.bias"] = up_block_two_sd_orig.pop("4.f_t.bias")
up_block_two_sd_new["upsamplers.0.norm2.weight"] = up_block_two_sd_orig.pop("4.gn_2.weight")
up_block_two_sd_new["upsamplers.0.norm2.bias"] = up_block_two_sd_orig.pop("4.gn_2.bias")
up_block_two_sd_new["upsamplers.0.conv2.weight"] = up_block_two_sd_orig.pop("4.f_2.weight")
up_block_two_sd_new["upsamplers.0.conv2.bias"] = up_block_two_sd_orig.pop("4.f_2.bias")

assert len(up_block_two_sd_orig) == 0

up_block_two = ResnetUpsampleBlock2D(
    in_channels=640,
    prev_output_channel=1024,
    out_channels=1024,
    temb_channels=1280,
    num_layers=4,
    add_upsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

up_block_two.load_state_dict(up_block_two_sd_new)

print("UP BLOCK THREE")

up_block_three_sd_orig = model.up[-3].state_dict()
up_block_three_sd_new = {}

for i in range(4):
    up_block_three_sd_new[f"resnets.{i}.norm1.weight"] = up_block_three_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_three_sd_new[f"resnets.{i}.norm1.bias"] = up_block_three_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_three_sd_new[f"resnets.{i}.conv1.weight"] = up_block_three_sd_orig.pop(f"{i}.f_1.weight")
    up_block_three_sd_new[f"resnets.{i}.conv1.bias"] = up_block_three_sd_orig.pop(f"{i}.f_1.bias")
    up_block_three_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_three_sd_orig.pop(f"{i}.f_t.weight")
    up_block_three_sd_new[f"resnets.{i}.time_emb_proj.bias"] = up_block_three_sd_orig.pop(f"{i}.f_t.bias")
    up_block_three_sd_new[f"resnets.{i}.norm2.weight"] = up_block_three_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_three_sd_new[f"resnets.{i}.norm2.bias"] = up_block_three_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_three_sd_new[f"resnets.{i}.conv2.weight"] = up_block_three_sd_orig.pop(f"{i}.f_2.weight")
    up_block_three_sd_new[f"resnets.{i}.conv2.bias"] = up_block_three_sd_orig.pop(f"{i}.f_2.bias")
    up_block_three_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_three_sd_orig.pop(f"{i}.f_s.weight")
    up_block_three_sd_new[f"resnets.{i}.conv_shortcut.bias"] = up_block_three_sd_orig.pop(f"{i}.f_s.bias")

up_block_three_sd_new["upsamplers.0.norm1.weight"] = up_block_three_sd_orig.pop("4.gn_1.weight")
up_block_three_sd_new["upsamplers.0.norm1.bias"] = up_block_three_sd_orig.pop("4.gn_1.bias")
up_block_three_sd_new["upsamplers.0.conv1.weight"] = up_block_three_sd_orig.pop("4.f_1.weight")
up_block_three_sd_new["upsamplers.0.conv1.bias"] = up_block_three_sd_orig.pop("4.f_1.bias")
up_block_three_sd_new["upsamplers.0.time_emb_proj.weight"] = up_block_three_sd_orig.pop("4.f_t.weight")
up_block_three_sd_new["upsamplers.0.time_emb_proj.bias"] = up_block_three_sd_orig.pop("4.f_t.bias")
up_block_three_sd_new["upsamplers.0.norm2.weight"] = up_block_three_sd_orig.pop("4.gn_2.weight")
up_block_three_sd_new["upsamplers.0.norm2.bias"] = up_block_three_sd_orig.pop("4.gn_2.bias")
up_block_three_sd_new["upsamplers.0.conv2.weight"] = up_block_three_sd_orig.pop("4.f_2.weight")
up_block_three_sd_new["upsamplers.0.conv2.bias"] = up_block_three_sd_orig.pop("4.f_2.bias")

assert len(up_block_three_sd_orig) == 0

up_block_three = ResnetUpsampleBlock2D(
    in_channels=320,
    prev_output_channel=1024,
    out_channels=640,
    temb_channels=1280,
    num_layers=4,
    add_upsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

up_block_three.load_state_dict(up_block_three_sd_new)

print("UP BLOCK FOUR")

up_block_four_sd_orig = model.up[-4].state_dict()
up_block_four_sd_new = {}

for i in range(4):
    up_block_four_sd_new[f"resnets.{i}.norm1.weight"] = up_block_four_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_four_sd_new[f"resnets.{i}.norm1.bias"] = up_block_four_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_four_sd_new[f"resnets.{i}.conv1.weight"] = up_block_four_sd_orig.pop(f"{i}.f_1.weight")
    up_block_four_sd_new[f"resnets.{i}.conv1.bias"] = up_block_four_sd_orig.pop(f"{i}.f_1.bias")
    up_block_four_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_four_sd_orig.pop(f"{i}.f_t.weight")
    up_block_four_sd_new[f"resnets.{i}.time_emb_proj.bias"] = up_block_four_sd_orig.pop(f"{i}.f_t.bias")
    up_block_four_sd_new[f"resnets.{i}.norm2.weight"] = up_block_four_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_four_sd_new[f"resnets.{i}.norm2.bias"] = up_block_four_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_four_sd_new[f"resnets.{i}.conv2.weight"] = up_block_four_sd_orig.pop(f"{i}.f_2.weight")
    up_block_four_sd_new[f"resnets.{i}.conv2.bias"] = up_block_four_sd_orig.pop(f"{i}.f_2.bias")
    up_block_four_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_four_sd_orig.pop(f"{i}.f_s.weight")
    up_block_four_sd_new[f"resnets.{i}.conv_shortcut.bias"] = up_block_four_sd_orig.pop(f"{i}.f_s.bias")

assert len(up_block_four_sd_orig) == 0

up_block_four = ResnetUpsampleBlock2D(
    in_channels=320,
    prev_output_channel=640,
    out_channels=320,
    temb_channels=1280,
    num_layers=4,
    add_upsample=False,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
)

up_block_four.load_state_dict(up_block_four_sd_new)

print("initial projection (conv_in)")

conv_in_sd_orig = model.embed_image.state_dict()
conv_in_sd_new = {}

conv_in_sd_new["weight"] = conv_in_sd_orig.pop("f.weight")
conv_in_sd_new["bias"] = conv_in_sd_orig.pop("f.bias")

assert len(conv_in_sd_orig) == 0

block_out_channels = [320, 640, 1024, 1024]

in_channels = 7
conv_in_kernel = 3
conv_in_padding = (conv_in_kernel - 1) // 2
conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

conv_in.load_state_dict(conv_in_sd_new)

print("out projection (conv_out) (conv_norm_out)")
out_channels = 6
norm_num_groups = 32
norm_eps = 1e-5
act_fn = "silu"
conv_out_kernel = 3
conv_out_padding = (conv_out_kernel - 1) // 2
conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
# uses torch.functional in orig
# conv_act = get_activation(act_fn)
conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)

conv_norm_out.load_state_dict(model.output.gn.state_dict())
conv_out.load_state_dict(model.output.f.state_dict())

print("timestep projection (time_proj) (time_embedding)")

f1_sd = model.embed_time.f_1.state_dict()
f2_sd = model.embed_time.f_2.state_dict()

time_embedding_sd = {
    "linear_1.weight": f1_sd.pop("weight"),
    "linear_1.bias": f1_sd.pop("bias"),
    "linear_2.weight": f2_sd.pop("weight"),
    "linear_2.bias": f2_sd.pop("bias"),
}

assert len(f1_sd) == 0
assert len(f2_sd) == 0

time_embedding_type = "learned"
num_train_timesteps = 1024
time_embedding_dim = 1280

time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
timestep_input_dim = block_out_channels[0]

time_embedding = TimestepEmbedding(timestep_input_dim, time_embedding_dim)

time_proj.load_state_dict(model.embed_time.emb.state_dict())
time_embedding.load_state_dict(time_embedding_sd)

print("CONVERT")

time_embedding.to("cuda")
time_proj.to("cuda")
conv_in.to("cuda")

block_one.to("cuda")
block_two.to("cuda")
block_three.to("cuda")
block_four.to("cuda")

mid_block_one.to("cuda")

up_block_one.to("cuda")
up_block_two.to("cuda")
up_block_three.to("cuda")
up_block_four.to("cuda")

conv_norm_out.to("cuda")
conv_out.to("cuda")

model.time_proj = time_proj
model.time_embedding = time_embedding
model.embed_image = conv_in

model.down[0] = block_one
model.down[1] = block_two
model.down[2] = block_three
model.down[3] = block_four

model.mid = mid_block_one

model.up[-1] = up_block_one
model.up[-2] = up_block_two
model.up[-3] = up_block_three
model.up[-4] = up_block_four

model.output.gn = conv_norm_out
model.output.f = conv_out

model.converted = True

sample_consistency_new = decoder_consistency(latent, generator=torch.Generator("cpu").manual_seed(0))
save_image(sample_consistency_new, "con_new.png")

assert (sample_consistency_orig == sample_consistency_new).all()

print("making unet")

unet = UNet2DModel(
    in_channels=in_channels,
    out_channels=out_channels,
    down_block_types=(
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    up_block_types=(
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    block_out_channels=block_out_channels,
    layers_per_block=3,
    norm_num_groups=norm_num_groups,
    norm_eps=norm_eps,
    resnet_time_scale_shift="scale_shift",
    time_embedding_type="learned",
    num_train_timesteps=num_train_timesteps,
    add_attention=False,
)

unet_state_dict = {}


def add_state_dict(prefix, mod):
    for k, v in mod.state_dict().items():
        unet_state_dict[f"{prefix}.{k}"] = v


add_state_dict("conv_in", conv_in)
add_state_dict("time_proj", time_proj)
add_state_dict("time_embedding", time_embedding)
add_state_dict("down_blocks.0", block_one)
add_state_dict("down_blocks.1", block_two)
add_state_dict("down_blocks.2", block_three)
add_state_dict("down_blocks.3", block_four)
add_state_dict("mid_block", mid_block_one)
add_state_dict("up_blocks.0", up_block_one)
add_state_dict("up_blocks.1", up_block_two)
add_state_dict("up_blocks.2", up_block_three)
add_state_dict("up_blocks.3", up_block_four)
add_state_dict("conv_norm_out", conv_norm_out)
add_state_dict("conv_out", conv_out)

unet.load_state_dict(unet_state_dict)

print("running with diffusers unet")

unet.to("cuda")

decoder_consistency.ckpt = unet

sample_consistency_new_2 = decoder_consistency(latent, generator=torch.Generator("cpu").manual_seed(0))
save_image(sample_consistency_new_2, "con_new_2.png")

assert (sample_consistency_orig == sample_consistency_new_2).all()

print("running with diffusers model")

Encoder.old_constructor = Encoder.__init__


def new_constructor(self, **kwargs):
    self.old_constructor(**kwargs)
    self.constructor_arguments = kwargs


Encoder.__init__ = new_constructor


vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
consistency_vae = ConsistencyDecoderVAE(
    encoder_args=vae.encoder.constructor_arguments,
    decoder_args=unet.config,
    scaling_factor=vae.config.scaling_factor,
    block_out_channels=vae.config.block_out_channels,
    latent_channels=vae.config.latent_channels,
)
consistency_vae.encoder.load_state_dict(vae.encoder.state_dict())
consistency_vae.quant_conv.load_state_dict(vae.quant_conv.state_dict())
consistency_vae.decoder_unet.load_state_dict(unet.state_dict())

consistency_vae.to(dtype=torch.float16, device="cuda")

sample_consistency_new_3 = consistency_vae.decode(
    0.18215 * latent, generator=torch.Generator("cpu").manual_seed(0)
).sample

print("max difference")
print((sample_consistency_orig - sample_consistency_new_3).abs().max())
print("total difference")
print((sample_consistency_orig - sample_consistency_new_3).abs().sum())
# assert (sample_consistency_orig == sample_consistency_new_3).all()

print("running with diffusers pipeline")

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", vae=consistency_vae, torch_dtype=torch.float16
)
pipe.to("cuda")

pipe("horse", generator=torch.Generator("cpu").manual_seed(0)).images[0].save("horse.png")


if args.save_pretrained is not None:
    consistency_vae.save_pretrained(args.save_pretrained)
