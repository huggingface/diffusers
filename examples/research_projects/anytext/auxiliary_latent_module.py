# text -> glyph render -> glyph l_g -> glyph block ->
# +> fuse layer
# position l_p -> position block ->

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from diffusers.models.autoencoders import AutoencoderKL
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
    def __init__(self, font_path, dims=2, glyph_channels=256, position_channels=64, model_channels=256, **kwargs):
        super().__init__()
        if font_path is None:
            raise ValueError("font_path must be provided!")
        self.font = ImageFont.truetype(font_path, 60)
        self.use_fp16 = kwargs.get("use_fp16", False)
        self.device = kwargs.get("device", "cpu")
        self.scale_factor = 0.18215
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

        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            variant="fp16" if self.use_fp16 else "fp32",
        )
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.fuse_block = zero_module(conv_nd(dims, 256 + 64 + 4, model_channels, 3, padding=1))

    @torch.no_grad()
    def forward(
        self,
        emb,
        context,
        mode,
        texts,
        prompt,
        draw_pos,
        ori_image,
        img_count,
        max_chars=77,
        revise_pos=False,
        sort_priority=False,
        h=512,
        w=512,
    ):
        if prompt is None and texts is None:
            raise ValueError("Prompt or texts must be provided!")
        n_lines = len(texts)
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
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if draw_pos is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            if draw_pos is None:
                raise ValueError(f"Can't read draw_pos image from {draw_pos}!")
            pos_imgs = 255 - draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            if not isinstance(draw_pos, np.ndarray):
                raise ValueError(f"Unknown format of draw_pos: {type(draw_pos)}")
        if mode == "edit":
            pos_imgs = cv2.resize(pos_imgs, (w, h))
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # separate pos_imgs
        pos_imgs = self.separate_pos_imgs(pos_imgs, sort_priority)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == " ":
                pass  # text-to-image without text
            else:
                raise ValueError(
                    f"Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!"
                )
        elif len(pos_imgs) > n_lines:
            str_warning = f"Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt."
            logger.warning(str_warning)
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img / 255.0]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
        # prepare info dict
        glyphs_list = []
        positions = []
        n_lines = [len(texts)] * img_count
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                logger.warning(str_warning)
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                glyphs = self.draw_glyph2(
                    self.font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False
                )
                gly_pos_img = cv2.drawContours(glyphs * 255, [poly_list[i] * gly_scale], 0, (255, 255, 255), 1)
                if revise_pos:
                    resize_gly = cv2.resize(glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0]))
                    new_pos = cv2.morphologyEx(
                        (resize_gly * 255).astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        kernel=np.ones((resize_gly.shape[0] // 10, resize_gly.shape[1] // 10), dtype=np.uint8),
                        iterations=1,
                    )
                    new_pos = new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    contours, _ = cv2.findContours(new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 1:
                        str_warning = f"Fail to revise position {i} to bounding rect, remain position unchanged..."
                        logger.warning(str_warning)
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        pre_pos[i] = cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.0
                        gly_pos_img = cv2.drawContours(glyphs * 255, [poly * gly_scale], 0, (255, 255, 255), 1)
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h * gly_scale, w * gly_scale, 1))
                gly_pos_imgs += [np.zeros((h * gly_scale, w * gly_scale, 1))]  # for show
            pos = pre_pos[i]
            glyphs_list += [self.arr2tensor(glyphs, img_count)]
            positions += [self.arr2tensor(pos, img_count)]

        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cpu()
        if self.use_fp16:
            masked_img = masked_img.half()
        masked_x = self.encode_first_stage(masked_img[None, ...]).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        masked_x = torch.cat([masked_x for _ in range(img_count)], dim=0)

        glyphs = torch.cat(glyphs_list, dim=1).sum(dim=1, keepdim=True)
        positions = torch.cat(positions, dim=1).sum(dim=1, keepdim=True)
        enc_glyph = self.glyph_block(glyphs, emb, context)
        enc_pos = self.position_block(positions, emb, context)
        guided_hint = self.fuse_block(torch.cat([enc_glyph, enc_pos, masked_x], dim=1))

        hint = self.arr2tensor(np_hint, img_count)

        return guided_hint, hint  # , gly_pos_imgs

    def encode_first_stage(self, masked_img):
        return retrieve_latents(self.vae.encode(masked_img)) * self.scale_factor

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
