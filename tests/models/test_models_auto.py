import unittest
from unittest.mock import patch

from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPTextModel

from diffusers.models import AutoModel, UNet2DConditionModel


class TestAutoModel(unittest.TestCase):
    @patch("diffusers.models.auto_model.hf_hub_download", side_effect=EntryNotFoundError("File not found"))
    def test_from_pretrained_falls_back_on_entry_error(self, mock_hf_hub_download):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet")
        assert isinstance(model, UNet2DConditionModel)

    def test_from_pretrained_loads_successfully(
        self
    ):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder")
        assert isinstance(model, CLIPTextModel)
