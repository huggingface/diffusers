import unittest
from unittest.mock import patch

from transformers import AlbertForMaskedLM, CLIPTextModel

from diffusers.models import AutoModel, UNet2DConditionModel


class TestAutoModel(unittest.TestCase):
    @patch("diffusers.models.AutoModel.load_config", side_effect=[EnvironmentError("File not found"), {"_class_name": "UNet2DConditionModel"}])
    def test_load_from_config_diffusers_with_subfolder(self, mock_load_config):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet")
        assert isinstance(model, UNet2DConditionModel)

    @patch("diffusers.models.AutoModel.load_config", side_effect=[EnvironmentError("File not found"), {"architectures": [ "CLIPTextModel"]}])
    def test_load_from_config_transformers_with_subfolder(self, mock_load_config):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder")
        assert isinstance(model, CLIPTextModel)

    def test_load_from_config_without_subfolder(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-albert")
        assert isinstance(model, AlbertForMaskedLM)

    def test_load_from_model_index(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder")
        assert isinstance(model, CLIPTextModel)
