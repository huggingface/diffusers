from typing import Optional

import cv2
import numpy as np
import torch
from PIL import ImageFont
from safetensors.torch import load_file
from torch import nn

from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
    def __init__(
        self,
        font_path,
        glyph_channels=1,
        position_channels=1,
        model_channels=320,
        vae=None,
        device="cpu",
        use_fp16=False,
    ):
        super().__init__()
        self.font = ImageFont.truetype(font_path, 60)
        self.use_fp16 = use_fp16
        self.device = device

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

        self.vae = vae.eval() if vae is not None else None

        self.fuse_block = nn.Conv2d(256 + 64 + 4, model_channels, 3, padding=1)

        self.glyph_block.load_state_dict(load_file("glyph_block.safetensors", device=str(self.device)))
        self.position_block.load_state_dict(load_file("position_block.safetensors", device=str(self.device)))
        self.fuse_block.load_state_dict(load_file("fuse_block.safetensors", device=str(self.device)))

        if use_fp16:
            self.glyph_block = self.glyph_block.to(dtype=torch.float16)
            self.position_block = self.position_block.to(dtype=torch.float16)
            self.fuse_block = self.fuse_block.to(dtype=torch.float16)

    @torch.no_grad()
    def forward(
        self,
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

        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().to(self.device)
        if self.use_fp16:
            masked_img = masked_img.half()
        masked_x = (retrieve_latents(self.vae.encode(masked_img[None, ...])) * self.vae.config.scaling_factor).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        text_info["masked_x"] = torch.cat([masked_x for _ in range(num_images_per_prompt)], dim=0)

        glyphs = torch.cat(text_info["glyphs"], dim=1).sum(dim=1, keepdim=True)
        positions = torch.cat(text_info["positions"], dim=1).sum(dim=1, keepdim=True)
        enc_glyph = self.glyph_block(glyphs)
        enc_pos = self.position_block(positions)
        guided_hint = self.fuse_block(torch.cat([enc_glyph, enc_pos, text_info["masked_x"]], dim=1))

        return guided_hint

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
        self.fuse_block = self.fuse_block.to(device)
        self.vae = self.vae.to(device)
        return self
