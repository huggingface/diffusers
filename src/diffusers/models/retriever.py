from transformers import CLIPVisionModel, CLIPFeatureExtractor
from datasets import load_dataset, Image, load_dataset_builder
import torch
from typing import Callable, List, Optional, Union
import numpy as np

def preprocess_images(images: List[Image.Image], feature_extractor: CLIPFeatureExtractor) -> torch.FloatTensor:
    """
    Preprocesses a list of images into a batch of tensors.

    Args:
        images (:obj:`List[Image.Image]`):
            A list of images to preprocess.

    Returns:
        :obj:`torch.FloatTensor`: A batch of tensors.
    """
    images = [np.array(image) for image in images]
    images = [(image + 1.) / 2. for image in images]
    images = feature_extractor(images, return_tensors="pt").pixel_values
    return images

def map_img_to_clip_feature(clip, feature_extractor, device, imgs):
    retrieved_images = preprocess_images(retrieved_images, feature_extractor).to(device)
    image_embeddings = clip.get_image_features(retrieved_images)
    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)
    image_embeddings = image_embeddings[None, ...]
    return image_embeddings

class Retriever:
    def __init__(self, clip_model, feature_extractor, device, dataset_path="dalle-mini/open-images", split="train"):
        dataset = load_dataset(dataset_path, split=split)
        self.clip_model = clip_model
        self.feature_extractor = feature_extractor
        self.dataset = dataset.map(lambda example: {'embeddings': map_img_to_clip_feature(clip_model, feature_extractor, device, example["Image"]).numpy()})
        self.dataset.add_faiss_index(column='embeddings')
    def get_knn(self, vec, k=10):
        return self.dataset.get_nearest_examples('embeddings', vec, k=k)