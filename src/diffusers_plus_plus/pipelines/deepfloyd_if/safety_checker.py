import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModelWithProjection, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


class IFSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModelWithProjection(config.vision_config)

        self.p_head = nn.Linear(config.vision_config.projection_dim, 1)
        self.w_head = nn.Linear(config.vision_config.projection_dim, 1)

    @torch.no_grad()
    def forward(self, clip_input, images, p_threshold=0.5, w_threshold=0.5):
        image_embeds = self.vision_model(clip_input)[0]

        nsfw_detected = self.p_head(image_embeds)
        nsfw_detected = nsfw_detected.flatten()
        nsfw_detected = nsfw_detected > p_threshold
        nsfw_detected = nsfw_detected.tolist()

        if any(nsfw_detected):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        for idx, nsfw_detected_ in enumerate(nsfw_detected):
            if nsfw_detected_:
                images[idx] = np.zeros(images[idx].shape)

        watermark_detected = self.w_head(image_embeds)
        watermark_detected = watermark_detected.flatten()
        watermark_detected = watermark_detected > w_threshold
        watermark_detected = watermark_detected.tolist()

        if any(watermark_detected):
            logger.warning(
                "Potential watermarked content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        for idx, watermark_detected_ in enumerate(watermark_detected):
            if watermark_detected_:
                images[idx] = np.zeros(images[idx].shape)

        return images, nsfw_detected, watermark_detected
