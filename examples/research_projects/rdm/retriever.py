import os
from typing import List

import faiss
import numpy as np
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPModel, PretrainedConfig

from diffusers import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def normalize_images(images: List[Image.Image]):
    images = [np.array(image) for image in images]
    images = [image / 127.5 - 1 for image in images]
    return images


def preprocess_images(images: List[np.array], feature_extractor: CLIPFeatureExtractor) -> torch.FloatTensor:
    """
    Preprocesses a list of images into a batch of tensors.

    Args:
        images (:obj:`List[Image.Image]`):
            A list of images to preprocess.

    Returns:
        :obj:`torch.FloatTensor`: A batch of tensors.
    """
    images = [np.array(image) for image in images]
    images = [(image + 1.0) / 2.0 for image in images]
    images = feature_extractor(images, return_tensors="pt").pixel_values
    return images


class IndexConfig(PretrainedConfig):
    def __init__(
        self,
        clip_name_or_path="openai/clip-vit-large-patch14",
        dataset_name="Isamu136/oxford_pets_with_l14_emb",
        image_column="image",
        index_name="embeddings",
        index_path=None,
        dataset_set="train",
        metric_type=faiss.METRIC_L2,
        faiss_device=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_name_or_path = clip_name_or_path
        self.dataset_name = dataset_name
        self.image_column = image_column
        self.index_name = index_name
        self.index_path = index_path
        self.dataset_set = dataset_set
        self.metric_type = metric_type
        self.faiss_device = faiss_device


class Index:
    """
    Each index for a retrieval model is specific to the clip model used and the dataset used.
    """

    def __init__(self, config: IndexConfig, dataset: Dataset):
        self.config = config
        self.dataset = dataset
        self.index_initialized = False
        self.index_name = config.index_name
        self.index_path = config.index_path
        self.init_index()

    def set_index_name(self, index_name: str):
        self.index_name = index_name

    def init_index(self):
        if not self.index_initialized:
            if self.index_path and self.index_name:
                try:
                    self.dataset.add_faiss_index(
                        column=self.index_name, metric_type=self.config.metric_type, device=self.config.faiss_device
                    )
                    self.index_initialized = True
                except Exception as e:
                    print(e)
                    logger.info("Index not initialized")
            if self.index_name in self.dataset.features:
                self.dataset.add_faiss_index(column=self.index_name)
                self.index_initialized = True

    def build_index(
        self,
        model=None,
        feature_extractor: CLIPFeatureExtractor = None,
        torch_dtype=torch.float32,
    ):
        if not self.index_initialized:
            model = model or CLIPModel.from_pretrained(self.config.clip_name_or_path).to(dtype=torch_dtype)
            feature_extractor = feature_extractor or CLIPFeatureExtractor.from_pretrained(
                self.config.clip_name_or_path
            )
            self.dataset = get_dataset_with_emb_from_clip_model(
                self.dataset,
                model,
                feature_extractor,
                image_column=self.config.image_column,
                index_name=self.config.index_name,
            )
            self.init_index()

    def retrieve_imgs(self, vec, k: int = 20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.get_nearest_examples(self.index_name, vec, k=k)

    def retrieve_imgs_batch(self, vec, k: int = 20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.get_nearest_examples_batch(self.index_name, vec, k=k)

    def retrieve_indices(self, vec, k: int = 20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.search(self.index_name, vec, k=k)

    def retrieve_indices_batch(self, vec, k: int = 20):
        vec = np.array(vec).astype(np.float32)
        return self.dataset.search_batch(self.index_name, vec, k=k)


class Retriever:
    def __init__(
        self,
        config: IndexConfig,
        index: Index = None,
        dataset: Dataset = None,
        model=None,
        feature_extractor: CLIPFeatureExtractor = None,
    ):
        self.config = config
        self.index = index or self._build_index(config, dataset, model=model, feature_extractor=feature_extractor)

    @classmethod
    def from_pretrained(
        cls,
        retriever_name_or_path: str,
        index: Index = None,
        dataset: Dataset = None,
        model=None,
        feature_extractor: CLIPFeatureExtractor = None,
        **kwargs,
    ):
        config = kwargs.pop("config", None) or IndexConfig.from_pretrained(retriever_name_or_path, **kwargs)
        return cls(config, index=index, dataset=dataset, model=model, feature_extractor=feature_extractor)

    @staticmethod
    def _build_index(
        config: IndexConfig, dataset: Dataset = None, model=None, feature_extractor: CLIPFeatureExtractor = None
    ):
        dataset = dataset or load_dataset(config.dataset_name)
        dataset = dataset[config.dataset_set]
        index = Index(config, dataset)
        index.build_index(model=model, feature_extractor=feature_extractor)
        return index

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        if self.config.index_path is None:
            index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
            self.index.dataset.get_index(self.config.index_name).save(index_path)
            self.config.index_path = index_path
        self.config.save_pretrained(save_directory)

    def init_retrieval(self):
        logger.info("initializing retrieval")
        self.index.init_index()

    def retrieve_imgs(self, embeddings: np.ndarray, k: int):
        return self.index.retrieve_imgs(embeddings, k)

    def retrieve_imgs_batch(self, embeddings: np.ndarray, k: int):
        return self.index.retrieve_imgs_batch(embeddings, k)

    def retrieve_indices(self, embeddings: np.ndarray, k: int):
        return self.index.retrieve_indices(embeddings, k)

    def retrieve_indices_batch(self, embeddings: np.ndarray, k: int):
        return self.index.retrieve_indices_batch(embeddings, k)

    def __call__(
        self,
        embeddings,
        k: int = 20,
    ):
        return self.index.retrieve_imgs(embeddings, k)


def map_txt_to_clip_feature(clip_model, tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if text_input_ids.shape[-1] > tokenizer.model_max_length:
        removed_text = tokenizer.batch_decode(text_input_ids[:, tokenizer.model_max_length :])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )
        text_input_ids = text_input_ids[:, : tokenizer.model_max_length]
    text_embeddings = clip_model.get_text_features(text_input_ids.to(clip_model.device))
    text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings[:, None, :]
    return text_embeddings[0][0].cpu().detach().numpy()


def map_img_to_model_feature(model, feature_extractor, imgs, device):
    for i, image in enumerate(imgs):
        if not image.mode == "RGB":
            imgs[i] = image.convert("RGB")
    imgs = normalize_images(imgs)
    retrieved_images = preprocess_images(imgs, feature_extractor).to(device)
    image_embeddings = model(retrieved_images)
    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings, dim=-1, keepdim=True)
    image_embeddings = image_embeddings[None, ...]
    return image_embeddings.cpu().detach().numpy()[0][0]


def get_dataset_with_emb_from_model(dataset, model, feature_extractor, image_column="image", index_name="embeddings"):
    return dataset.map(
        lambda example: {
            index_name: map_img_to_model_feature(model, feature_extractor, [example[image_column]], model.device)
        }
    )


def get_dataset_with_emb_from_clip_model(
    dataset, clip_model, feature_extractor, image_column="image", index_name="embeddings"
):
    return dataset.map(
        lambda example: {
            index_name: map_img_to_model_feature(
                clip_model.get_image_features, feature_extractor, [example[image_column]], clip_model.device
            )
        }
    )
