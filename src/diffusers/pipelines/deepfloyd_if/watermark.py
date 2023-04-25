from typing import List

import PIL
import torch
from PIL import Image

from ...configuration_utils import ConfigMixin
from ...models.modeling_utils import ModelMixin
from ...utils import PIL_INTERPOLATION


class IFWatermarker(ModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()

        self.register_buffer("watermark_image", torch.zeros((62, 62, 4)))
        self.watermark_image_as_pil = None

    def apply_watermark(self, images: List[PIL.Image.Image], sample_size=None):
        # copied from https://github.com/deep-floyd/IF/blob/b77482e36ca2031cb94dbca1001fc1e6400bf4ab/deepfloyd_if/modules/base.py#L287

        h = images[0].height
        w = images[0].width

        sample_size = sample_size or h

        coef = min(h / sample_size, w / sample_size)
        img_h, img_w = (int(h / coef), int(w / coef)) if coef < 1 else (h, w)

        S1, S2 = 1024**2, img_w * img_h
        K = (S2 / S1) ** 0.5
        wm_size, wm_x, wm_y = int(K * 62), img_w - int(14 * K), img_h - int(14 * K)

        if self.watermark_image_as_pil is None:
            watermark_image = self.watermark_image.to(torch.uint8).cpu().numpy()
            watermark_image = Image.fromarray(watermark_image, mode="RGBA")
            self.watermark_image_as_pil = watermark_image

        wm_img = self.watermark_image_as_pil.resize(
            (wm_size, wm_size), PIL_INTERPOLATION["bicubic"], reducing_gap=None
        )

        for pil_img in images:
            pil_img.paste(wm_img, box=(wm_x - wm_size, wm_y - wm_size, wm_x, wm_y), mask=wm_img.split()[-1])

        return images
