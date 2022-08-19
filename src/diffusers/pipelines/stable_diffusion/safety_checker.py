import numpy as np
import torch
import torch.nn as nn

from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

from ...utils import logging


logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.T)


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.register_buffer("concept_embeds_weights", torch.ones(17))
        self.register_buffer("special_care_embeds_weights", torch.ones(3))

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
            adjustment = 0.05

            for concet_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concet_idx]
                concept_threshold = self.special_care_embeds_weights[concet_idx].item()
                result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concet_idx] > 0:
                    result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                    adjustment = 0.01

            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = self.concept_embeds_weights[concet_idx].item()
                result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concet_idx] > 0:
                    result_img["bad_concepts"].append(concet_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(result[i]["bad_concepts"]) > 0 or i in range(len(result))]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts
