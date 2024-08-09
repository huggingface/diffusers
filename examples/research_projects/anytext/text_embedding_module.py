import cv2
import numpy as np
import torch
from embedding_manager import EmbeddingManager
from frozen_clip_embedder_t3 import FrozenCLIPEmbedderT3
from PIL import Image, ImageDraw, ImageFont
from recognizer import TextRecognizer, create_predictor
from torch import nn

from diffusers.utils import (
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextEmbeddingModule(nn.Module):
    def __init__(self, font_path, use_fp16=False, device="cpu"):
        super().__init__()
        self.use_fp16 = use_fp16
        self.device = device
        # TODO: Learn if the recommended font file is free to use
        self.font = ImageFont.truetype(font_path, 60)
        self.frozen_CLIP_embedder_t3 = FrozenCLIPEmbedderT3(device=self.device, use_fp16=self.use_fp16)
        self.embedding_manager = EmbeddingManager(self.frozen_CLIP_embedder_t3, use_fp16=self.use_fp16)
        # for param in self.embedding_manager.embedding_parameters():
        #     param.requires_grad = True
        rec_model_dir = "OCR/ppv3_rec.pth"
        self.text_predictor = create_predictor(rec_model_dir, device=self.device, use_fp16=self.use_fp16).eval()
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        args = {}
        args["rec_image_shape"] = "3, 48, 320"
        args["rec_batch_num"] = 6
        args["rec_char_dict_path"] = "OCR/ppocr_keys_v1.txt"
        args["use_fp16"] = self.use_fp16
        self.embedding_manager.recog = TextRecognizer(args, self.text_predictor)

    @torch.no_grad()
    def forward(
        self,
        prompt,
        texts,
        negative_prompt,
        num_images_per_prompt,
        mode,
        draw_pos,
        sort_priority="↕",
        max_chars=77,
        revise_pos=False,
        h=512,
        w=512,
    ):
        if prompt is None and texts is None:
            raise ValueError("Prompt or texts must be provided!")
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
        n_lines = len(texts)
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
        text_info = {}
        text_info["glyphs"] = []
        text_info["gly_line"] = []
        text_info["positions"] = []
        text_info["n_lines"] = [len(texts)] * num_images_per_prompt
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                logger.warning(str_warning)
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = self.draw_glyph(self.font, text)
                glyphs = self.draw_glyph2(
                    self.font, text, poly_list[i], scale=gly_scale, width=w, height=h, add_space=False
                )
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
            else:
                glyphs = np.zeros((h * gly_scale, w * gly_scale, 1))
                gly_line = np.zeros((80, 512, 1))
            pos = pre_pos[i]
            text_info["glyphs"] += [self.arr2tensor(glyphs, num_images_per_prompt)]
            text_info["gly_line"] += [self.arr2tensor(gly_line, num_images_per_prompt)]
            text_info["positions"] += [self.arr2tensor(pos, num_images_per_prompt)]

        # hint = self.arr2tensor(np_hint, len(prompt))

        self.embedding_manager.encode_text(text_info)
        prompt_embeds = self.frozen_CLIP_embedder_t3.encode([prompt], embedding_manager=self.embedding_manager)

        self.embedding_manager.encode_text(text_info)
        negative_prompt_embeds = self.frozen_CLIP_embedder_t3.encode(
            [negative_prompt], embedding_manager=self.embedding_manager
        )

        return prompt_embeds, negative_prompt_embeds, text_info, np_hint

    def arr2tensor(self, arr, bs):
        arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().cpu()
        if self.use_fp16:
            _arr = _arr.half()
        _arr = torch.stack([_arr for _ in range(bs)], dim=0)
        return _arr

    def separate_pos_imgs(self, img, sort_priority, gap=102):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        components = []
        for label in range(1, num_labels):
            component = np.zeros_like(img)
            component[labels == label] = 255
            components.append((component, centroids[label]))
        if sort_priority == "↕":
            fir, sec = 1, 0  # top-down first
        elif sort_priority == "↔":
            fir, sec = 0, 1  # left-right first
        else:
            raise ValueError(f"Unknown sort_priority: {sort_priority}")
        components.sort(key=lambda c: (c[1][fir] // gap, c[1][sec] // gap))
        sorted_components = [c[0] for c in components]
        return sorted_components

    def find_polygon(self, image, min_rect=False):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
        if min_rect:
            # get minimum enclosing rectangle
            rect = cv2.minAreaRect(max_contour)
            poly = np.int0(cv2.boxPoints(rect))
        else:
            # get approximate polygon
            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            poly = cv2.approxPolyDP(max_contour, epsilon, True)
            n, _, xy = poly.shape
            poly = poly.reshape(n, xy)
        cv2.drawContours(image, [poly], -1, 255, -1)
        return poly, image

    def draw_glyph(self, font, text):
        g_size = 50
        W, H = (512, 80)
        new_font = font.font_variant(size=g_size)
        img = Image.new(mode="1", size=(W, H), color=0)
        draw = ImageDraw.Draw(img)
        left, top, right, bottom = new_font.getbbox(text)
        text_width = max(right - left, 5)
        text_height = max(bottom - top, 5)
        ratio = min(W * 0.9 / text_width, H * 0.9 / text_height)
        new_font = font.font_variant(size=int(g_size * ratio))

        text_width, text_height = new_font.getsize(text)
        offset_x, offset_y = new_font.getoffset(text)
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2 - offset_y // 2
        draw.text((x, y), text, font=new_font, fill="white")
        img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
        return img

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

    def insert_spaces(self, string, nSpace):
        if nSpace == 0:
            return string
        new_string = ""
        for char in string:
            new_string += char + " " * nSpace
        return new_string[:-nSpace]

    def to(self, device):
        self.device = device
        self.frozen_CLIP_embedder_t3.to(device)
        self.embedding_manager.to(device)
        self.text_predictor.to(device)
        return self
