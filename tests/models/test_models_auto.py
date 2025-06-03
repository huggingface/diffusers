import unittest
from unittest.mock import patch

from transformers import CLIPTextModel, LongformerModel

from diffusers.models import AutoModel, UNet2DConditionModel


class TestAutoModel(unittest.TestCase):
    @patch(
        "diffusers.models.AutoModel.load_config",
        side_effect=[EnvironmentError("File not found"), {"_class_name": "UNet2DConditionModel"}],
    )
    def test_load_from_config_diffusers_with_subfolder(self, mock_load_config):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet")
        assert isinstance(model, UNet2DConditionModel)

    @patch(
        "diffusers.models.AutoModel.load_config",
        side_effect=[EnvironmentError("File not found"), {"model_type": "clip_text_model"}],
    )
    def test_load_from_config_transformers_with_subfolder(self, mock_load_config):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder")
        assert isinstance(model, CLIPTextModel)

    def test_load_from_config_without_subfolder(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-longformer")
        assert isinstance(model, LongformerModel)

    def test_load_from_model_index(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder")
        assert isinstance(model, CLIPTextModel)
