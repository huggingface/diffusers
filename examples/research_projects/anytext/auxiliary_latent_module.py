# text -> glyph render -> glyph l_g -> glyph block ->
# +> fuse layer
# position l_p -> position block ->

import math
from typing import Optional

import cv2
import numpy as np
import torch
from einops import repeat
from PIL import ImageFont
from torch import nn

from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.controlnet.zero_module
def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
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


class AuxiliaryLatentModule(nn.Module):
    def __init__(self, dims=2, glyph_channels=1, position_channels=1, model_channels=320, **kwargs):
        super().__init__()
        self.font = ImageFont.truetype("/home/cosmos/Documents/gits/AnyText/font/Arial_Unicode.ttf", 60)
        self.use_fp16 = kwargs.get("use_fp16", False)
        self.device = kwargs.get("device", "cpu")
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.glyph_block = nn.Sequential(
            nn.Conv2d(glyph_channels, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        self.position_block = nn.Sequential(
            nn.Conv2d(position_channels, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.time_embed = self.time_embed.to(device="cuda", dtype=torch.float16)
        self.glyph_block = self.glyph_block.to(device="cuda", dtype=torch.float16)
        self.position_block = self.position_block.to(device="cuda", dtype=torch.float16)

        self.vae = kwargs.get("vae")
        self.vae.eval()

        self.fuse_block = zero_module(nn.Conv2d(256 + 64 + 4, model_channels, 3, padding=1))
        self.fuse_block = self.fuse_block.to(device="cuda", dtype=torch.float16)

    @torch.no_grad()
    def forward(
        self,
        context,
        text_info,
        mode,
        draw_pos,
        ori_image,
        num_images_per_prompt,
        np_hint,
        h=512,
        w=512,
    ):
        if mode == "generate":
            edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode == "edit":
            if draw_pos is None or ori_image is None:
                raise ValueError("Reference image and position image are needed for text editing!")
            if isinstance(ori_image, str):
                ori_image = cv2.imread(ori_image)[..., ::-1]
                if ori_image is None:
                    raise ValueError(f"Can't read ori_image image from {ori_image}!")
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
            else:
                if not isinstance(ori_image, np.ndarray):
                    raise ValueError(f"Unknown format of ori_image: {type(ori_image)}")
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = self.check_channels(edit_image)
            edit_image = self.resize_image(
                edit_image, max_length=768
            )  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img

        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().to(self.device)
        if self.use_fp16:
            masked_img = masked_img.half()
        masked_x = self.encode_first_stage(masked_img[None, ...]).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        text_info["masked_x"] = torch.cat([masked_x for _ in range(num_images_per_prompt)], dim=0)

        glyphs = torch.cat(text_info["glyphs"], dim=1).sum(dim=1, keepdim=True)
        positions = torch.cat(text_info["positions"], dim=1).sum(dim=1, keepdim=True)
        t_emb = self.timestep_embedding(torch.tensor([1000], device="cuda"), self.model_channels, repeat_only=False)
        if self.use_fp16:
            t_emb = t_emb.half()
        emb = self.time_embed(t_emb)
        print(glyphs.shape, emb.shape, positions.shape, context.shape)
        enc_glyph = self.glyph_block(glyphs.cuda(), emb, context)
        enc_pos = self.position_block(positions.cuda(), emb, context)
        guided_hint = self.fuse_block(torch.cat([enc_glyph, enc_pos, text_info["masked_x"].cuda()], dim=1))

        return guided_hint

    def timestep_embedding(self, timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
                device=timesteps.device
            )
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(timesteps, "b -> b d", d=dim)
        return embedding

    def encode_first_stage(self, masked_img):
        return retrieve_latents(self.vae.encode(masked_img)) * self.vae.config.scaling_factor

    def check_channels(self, image):
        channels = image.shape[2] if len(image.shape) == 3 else 1
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif channels > 3:
            image = image[:, :, :3]
        return image

    def resize_image(self, img, max_length=768):
        height, width = img.shape[:2]
        max_dimension = max(height, width)

        if max_dimension > max_length:
            scale_factor = max_length / max_dimension
            new_width = int(round(width * scale_factor))
            new_height = int(round(height * scale_factor))
            new_size = (new_width, new_height)
            img = cv2.resize(img, new_size)
        height, width = img.shape[:2]
        img = cv2.resize(img, (width - (width % 64), height - (height % 64)))
        return img

    def insert_spaces(self, string, nSpace):
        if nSpace == 0:
            return string
        new_string = ""
        for char in string:
            new_string += char + " " * nSpace
        return new_string[:-nSpace]

    def to(self, device):
        self.device = device
        self.glyph_block = self.glyph_block.to(device)
        self.position_block = self.position_block.to(device)
        self.vae = self.vae.to(device)
        self.fuse_block = self.fuse_block.to(device)
        return self
