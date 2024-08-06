# text -> glyph render -> glyph lines -> OCR -> linear ->
# +> Token Replacement -> FrozenCLIPEmbedderT3
# text -> tokenizer ->


from typing import Optional

import torch
from embedding_manager import EmbeddingManager
from frozen_clip_embedder_t3 import FrozenCLIPEmbedderT3
from PIL import ImageFont
from recognizer import TextRecognizer, create_predictor
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2

from diffusers.utils import (
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextEmbeddingModule(nn.Module):
    def __init__(self, use_fp16):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # TODO: Learn if the recommended font file is free to use
        self.font = ImageFont.truetype("/home/cosmos/Documents/gits/AnyText/font/Arial_Unicode.ttf", 60)
        self.frozen_CLIP_embedder_t3 = FrozenCLIPEmbedderT3(device=self.device)
        self.embedding_manager_config = {
            "valid": True,
            "emb_type": "ocr",
            "glyph_channels": 1,
            "position_channels": 1,
            "add_pos": False,
            "placeholder_string": "*",
        }
        self.embedding_manager = EmbeddingManager(self.frozen_CLIP_embedder_t3, **self.embedding_manager_config)
        # TODO: Understand the reason of param.requires_grad = True
        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True
        rec_model_dir = "/home/cosmos/Documents/gits/AnyText/ocr_weights/ppv3_rec.pth"
        self.text_predictor = create_predictor(rec_model_dir).eval()
        args = {}
        args["rec_image_shape"] = "3, 48, 320"
        args["rec_batch_num"] = 6
        args["rec_char_dict_path"] = "/home/cosmos/Documents/gits/AnyText/ocr_weights/ppocr_keys_v1.txt"
        args["use_fp16"] = use_fp16
        self.cn_recognizer = TextRecognizer(args, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.embedding_manager.recog = self.cn_recognizer

    @torch.no_grad()
    def forward(
        self,
        prompt,
        texts,
        negative_prompt,
        num_images_per_prompt,
        mode,
        draw_pos,
        ori_image,
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
            text_info["glyphs"] += [self.arr2tensor(glyphs, len(prompt))]
            text_info["gly_line"] += [self.arr2tensor(gly_line, len(prompt))]
            text_info["positions"] += [self.arr2tensor(pos, len(prompt))]
        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().to(self.device)
        if self.use_fp16:
            masked_img = masked_img.half()
        masked_x = self.encode_first_stage(masked_img[None, ...]).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        text_info["masked_x"] = torch.cat([masked_x for _ in range(len(prompt))], dim=0)
        # hint = self.arr2tensor(np_hint, len(prompt))

        self.embedding_manager.encode_text(text_info)
        prompt_embeds = self.frozen_CLIP_embedder_t3.encode([prompt], embedding_manager=self.embedding_manager)

        self.embedding_manager.encode_text(text_info)
        negative_prompt_embeds = self.frozen_CLIP_embedder_t3.encode(
            [negative_prompt], embedding_manager=self.embedding_manager
        )

        return prompt_embeds, negative_prompt_embeds
