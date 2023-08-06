import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPFeatureExtractor, CLIPModel, CLIPTokenizer

from diffusers.utils import torch_device
from diffusers.utils.testing_utils import is_faiss_available, require_faiss


torch.backends.cuda.matmul.allow_tf32 = False
if is_faiss_available():
    from diffusers import (
        IndexConfig,
        Retriever,
    )
    from diffusers.pipelines.rdm.retriever import (
        map_img_to_model_feature,
        map_txt_to_clip_feature,
    )


@require_faiss
class RetrieverFastTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        clip_config = CLIPConfig.from_pretrained("hf-internal-testing/tiny-random-clip")
        clip_config.text_config.vocab_size = 49408

        clip = CLIPModel.from_pretrained(
            "hf-internal-testing/tiny-random-clip", config=clip_config, ignore_mismatched_sizes=True
        )
        clip = clip.to(torch_device)
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "hf-internal-testing/tiny-random-clip", size={"shortest_edge": 30}, crop_size={"height": 30, "width": 30}
        )

        config = IndexConfig(dataset_name="hf-internal-testing/dummy_image_text_data")
        retriever = Retriever(config, model=clip, feature_extractor=feature_extractor)
        return {"retriever": retriever, "clip": clip, "feature_extractor": feature_extractor}

    def test_retrieving_images_given_prompt(self):
        components = self.get_dummy_components()
        retriever, clip = components["retriever"], components["clip"]
        prompt = "a pizza"
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        clip_feature = map_txt_to_clip_feature(clip, tokenizer, prompt)
        retrieved_examples = retriever.retrieve_imgs(clip_feature, 5)
        assert retrieved_examples.examples["text"] == [
            "a red and white ball with an angry look on its face",
            "a cartoon picture of a green vegetable with eyes",
            "a green and yellow toy with a red nose",
            "a picture of a pink and yellow pokemon figure",
            "a blue and black object with two eyes",
        ]
    def test_batch_retrieving_images_given_prompt(self):
        components = self.get_dummy_components()
        retriever, clip = components["retriever"], components["clip"]
        prompt = "a pizza"
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        clip_feature = map_txt_to_clip_feature(clip, tokenizer, prompt)
        retrieved_examples = retriever.retrieve_imgs_batch(clip_feature[None], 5)
        assert retrieved_examples.total_examples[0]["text"] == [
            "a red and white ball with an angry look on its face",
            "a cartoon picture of a green vegetable with eyes",
            "a green and yellow toy with a red nose",
            "a picture of a pink and yellow pokemon figure",
            "a blue and black object with two eyes",
        ]

    def test_retrieving_images_given_image(self):
        components = self.get_dummy_components()
        retriever, clip, feature_extractor = (
            components["retriever"],
            components["clip"],
            components["feature_extractor"],
        )
        image = Image.fromarray(np.zeros((30, 30, 3)).astype(np.uint8))
        clip_feature = map_img_to_model_feature(
            clip.get_image_features, feature_extractor, [image], device=torch_device
        )
        retrieved_examples = retriever.retrieve_imgs(clip_feature, 5)
        assert retrieved_examples.examples["text"] == [
            "a blue and black object with two eyes",
            "a drawing of a gray and black dragon",
            "a drawing of a green pokemon with red eyes",
            "a cartoon picture of a green vegetable with eyes",
            "a green and yellow toy with a red nose",
        ]

    def test_retrieve_indices(self):
        components = self.get_dummy_components()
        retriever, clip = components["retriever"], components["clip"]
        prompt = "a pizza"
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        clip_feature = map_txt_to_clip_feature(clip, tokenizer, prompt)
        retrieved_examples = retriever.retrieve_indices(clip_feature, 5)
        indices = retrieved_examples.indices
        assert (indices - np.array([2, 12, 1, 10, 17])).max() < 1e-8
    def test_batch_retrieve_indices(self):
        components = self.get_dummy_components()
        retriever, clip = components["retriever"], components["clip"]
        prompt = "a pizza"
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        clip_feature = map_txt_to_clip_feature(clip, tokenizer, prompt)
        retrieved_examples = retriever.retrieve_indices_batch(clip_feature[None], 5)
        indices = retrieved_examples.total_indices[0]
        assert (indices - np.array([2, 12, 1, 10, 17])).max() < 1e-8
    def test_indexing(self):
        components = self.get_dummy_components()
        retriever = components["retriever"]
        assert retriever.config.index_name in retriever.index.dataset[0]
        assert np.array(retriever.index.dataset[0][retriever.config.index_name]).shape[0] == 64
