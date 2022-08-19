#!/usr/bin/env python3
from cgitb import reset
from typing import OrderedDict
import torch, torch.nn as nn
import open_clip
import numpy as np
import yaml

from open_clip import create_model_and_transforms

model, _, preprocess = create_model_and_transforms("ViT-L-14", "openai")

def normalized(a, axis=-1, order=2):
    """Normalize the given array along the specified axis in order to"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def pw_cosine_distance(input_a, input_b):
    normalized_input_a = torch.nn.functional.normalize(input_a)
    normalized_input_b = torch.nn.functional.normalize(input_b)
    return torch.mm(normalized_input_a, normalized_input_b.T)

class SafetyChecker(nn.Module):
    def __init__(self, device = 'cuda') -> None:
        super().__init__()
        self.clip_model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        safety_settings = yaml.safe_load(open("/home/patrick/safety_settings.yml", "r"))
        self.concepts_dict = dict(safety_settings["nsfw"]["concepts"])
        self.special_care_dict = dict(safety_settings["special"]["concepts"])
        self.concept_embeds = self.get_text_embeds(
            list(self.concepts_dict.keys()))
        self.special_care_embeds = self.get_text_embeds(
            list(self.special_care_dict.keys()))

    def get_image_embeds(self, input):
        """Get embeddings for images or tensor"""
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Preprocess if input is a list of PIL images
                if isinstance(input, list):
                    l = []
                    for image in input:
                        l.append(self.preprocess(image))
                    img_tensor = torch.stack(l)
                # input is a tensor
                elif isinstance(input, torch.Tensor):
                    img_tensor = input
                return self.clip_model.encode_image(img_tensor.half().to(self.device))

    def get_text_embeds(self, input):
        """Get text embeddings for a list of text"""
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                input = open_clip.tokenize(input).to(self.device)
                return(self.clip_model.encode_text(input))

    def forward(self, images):
        """Get embeddings for images and output nsfw and concept scores"""
        image_embeds = self.get_image_embeds(images)
        concept_list = list(self.concepts_dict.keys())
        special_list = list(self.special_care_dict.keys())
        special_cos_dist = pw_cosine_distance(image_embeds,
                                              self.special_care_embeds).cpu().numpy()
        cos_dist = pw_cosine_distance(image_embeds,
                                      self.concept_embeds).cpu().numpy()
        result = []
        for i in range(image_embeds.shape[0]):
            result_img = {
                "special_scores":{},
                "special_care":[],
                "concept_scores":{},
                "bad_concepts":[]}
            adjustment = 0.05
            for j in range(len(special_cos_dist[0])):
                concept_name = special_list[j]
                concept_cos = special_cos_dist[i][j]
                concept_threshold = self.special_care_dict[concept_name]
                result_img["special_scores"][concept_name] = round(
                    concept_cos - concept_threshold + adjustment,3)
                if result_img["special_scores"][concept_name] > 0:
                    result_img["special_care"].append({concept_name,result_img["special_scores"][concept_name]})
                    adjustment = 0.01
            for j in range(len(cos_dist[0])):
                concept_name = concept_list[j]
                concept_cos = cos_dist[i][j]
                concept_threshold = self.concepts_dict[concept_name]
                result_img["concept_scores"][concept_name] = round(concept_cos - concept_threshold + adjustment,3)
                if result_img["concept_scores"][concept_name]> 0: 
                    result_img["bad_concepts"].append(concept_name)
            result.append(result_img)
        return result
