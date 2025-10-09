# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

        self.adjustment = 0.0

    def update_safety_checker_Level(self, Level):
        """
        Adjust the safety checker level.

        Parameters:
            Level (`int` or `float` or one of the following [`WEAK`], [`MEDIUM`], [`NOMAL`], [`STRONG`], [`MAX`]):
                The level of safety checker adjustment, either as an integer, a float, or one of the predefined levels.
                Negative values decrease the filtering strength, while positive values increase it.
        """
        Level_dict = {
            "WEAK": -0.0690,
            "MEDIUM": -0.0175,
            "NOMAL": 0.0,
            "STRONG": 0.0150,
            "MAX": 0.0740,
        }

        # If the provided Level is a predefined string, convert it to the corresponding float value
        if Level in Level_dict:
            Level = Level_dict[Level]

        # Check if the Level is a float or an integer
        if isinstance(Level, (float, int)):
            setattr(self, "adjustment", Level)  # Set the adjustment attribute to the Level value
        else:
            # Raise an error if Level is not a valid type or predefined string
            raise ValueError(
                "`int` or `float` or one of the following ['WEAK'], ['MEDIUM'], ['NOMAL'], ['STRONG'], ['MAX']"
            )

        # Log a warning if the adjustment level is weakened (negative value)
        if self.adjustment < 0:
            logger.warning(
                "You have weakened the filtering strength of safety checker. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " When reducing the filtering strength, take the same action as when disabling the safety checker."
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + self.adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    self.adjustment += 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + self.adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                else:
                    images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts

    @torch.no_grad()
    def forward_onnx(self, clip_input: torch.Tensor, images: torch.Tensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images

        special_scores = special_cos_dist - self.special_care_embeds_weights + self.adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        images[has_nsfw_concepts] = 0.0  # black image

        return images, has_nsfw_concepts
