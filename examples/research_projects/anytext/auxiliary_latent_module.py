# text -> glyph render -> glyph l_g -> glyph block ->
# +> fuse layer
# position l_p -> position block ->

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


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
        self.glyph_block = nn.Sequential(
            conv_nd(dims, glyph_channels, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        self.position_block = nn.Sequential(
            conv_nd(dims, position_channels, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        self.vae = kwargs.get("vae")
        self.vae.eval()

        self.fuse_block = zero_module(conv_nd(dims, 256 + 64 + 4, model_channels, 3, padding=1))

    @torch.no_grad()
    def forward(
        self,
        emb,
        context,
        text_info,
    ):
        glyphs = torch.cat(text_info["glyphs"], dim=1).sum(dim=1, keepdim=True)
        positions = torch.cat(text_info["positions"], dim=1).sum(dim=1, keepdim=True)
        enc_glyph = self.glyph_block(glyphs, emb, context)
        enc_pos = self.position_block(positions, emb, context)
        guided_hint = self.fuse_block(torch.cat([enc_glyph, enc_pos, text_info["masked_x"]], dim=1))

        return guided_hint

    def encode_first_stage(self, masked_img):
        return retrieve_latents(self.vae.encode(masked_img)) * self.vae.scale_factor

    def arr2tensor(self, arr, bs):
        arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().cpu()
        if self.use_fp16:
            _arr = _arr.half()
        _arr = torch.stack([_arr for _ in range(bs)], dim=0)
        return _arr

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

    def draw_glyph2(self, font, text, polygon, vertAng=10, scale=1, width=512, height=512, add_space=True):
        enlarge_polygon = polygon * scale
        rect = cv2.minAreaRect(enlarge_polygon)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w, h = rect[1]
        angle = rect[2]
        if angle < -45:
            angle += 90
        angle = -angle
        if w < h:
            angle += 90

        vert = False
        if abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng:
            _w = max(box[:, 0]) - min(box[:, 0])
            _h = max(box[:, 1]) - min(box[:, 1])
            if _h >= _w:
                vert = True
                angle = 0

        img = np.zeros((height * scale, width * scale, 3), np.uint8)
        img = Image.fromarray(img)

        # infer font size
        image4ratio = Image.new("RGB", img.size, "white")
        draw = ImageDraw.Draw(image4ratio)
        _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
        text_w = min(w, h) * (_tw / _th)
        if text_w <= max(w, h):
            # add space
            if len(text) > 1 and not vert and add_space:
                for i in range(1, 100):
                    text_space = self.insert_spaces(text, i)
                    _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                    if min(w, h) * (_tw2 / _th2) > max(w, h):
                        break
                text = self.insert_spaces(text, i - 1)
            font_size = min(w, h) * 0.80
        else:
            shrink = 0.75 if vert else 0.85
            font_size = min(w, h) / (text_w / max(w, h)) * shrink
        new_font = font.font_variant(size=int(font_size))

        left, top, right, bottom = new_font.getbbox(text)
        text_width = right - left
        text_height = bottom - top

        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if not vert:
            draw.text(
                (rect[0][0] - text_width // 2, rect[0][1] - text_height // 2 - top),
                text,
                font=new_font,
                fill=(255, 255, 255, 255),
            )
        else:
            x_s = min(box[:, 0]) + _w // 2 - text_height // 2
            y_s = min(box[:, 1])
            for c in text:
                draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
                _, _t, _, _b = new_font.getbbox(c)
                y_s += _b

        rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

        x_offset = int((img.width - rotated_layer.width) / 2)
        y_offset = int((img.height - rotated_layer.height) / 2)
        img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
        img = np.expand_dims(np.array(img.convert("1")), axis=2).astype(np.float64)
        return img

    def to(self, device):
        self.device = device
        self.glyph_block = self.glyph_block.to(device)
        self.position_block = self.position_block.to(device)
        self.vae = self.vae.to(device)
        self.fuse_block = self.fuse_block.to(device)
        return self
