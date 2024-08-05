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

from diffusers.utils import (
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TextEmbeddingModule(nn.Module):
    def __init__(self, use_fp16):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # TODO: Learn if the recommended font file is free to use
        self.font = ImageFont.truetype("/home/x/Documents/gits/AnyText/font/Arial_Unicode.ttf", 60)
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
        rec_model_dir = "/home/x/Documents/gits/AnyText/ocr_weights/ppv3_rec.pth"
        self.text_predictor = create_predictor(rec_model_dir).eval()
        args = {}
        args["rec_image_shape"] = "3, 48, 320"
        args["rec_batch_num"] = 6
        args["rec_char_dict_path"] = "/home/x/Documents/gits/AnyText/ocr_weights/ppocr_keys_v1.txt"
        args["use_fp16"] = use_fp16
        self.cn_recognizer = TextRecognizer(args, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.embedding_manager.recog = self.cn_recognizer

    @torch.no_grad()
    def forward(
        self,
        prompt,
        text_info,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        self.embedding_manager.encode_text(text_info)
        prompt_embeds = self.frozen_CLIP_embedder_t3.encode([prompt], embedding_manager=self.embedding_manager)

        self.embedding_manager.encode_text(text_info)
        negative_prompt_embeds = self.frozen_CLIP_embedder_t3.encode(
            [negative_prompt], embedding_manager=self.embedding_manager
        )

        return prompt_embeds, negative_prompt_embeds
