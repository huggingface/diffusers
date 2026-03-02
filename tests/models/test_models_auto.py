import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
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
        model = AutoModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder", use_safetensors=False
        )
        assert isinstance(model, CLIPTextModel)

    def test_load_from_config_without_subfolder(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-longformer")
        assert isinstance(model, LongformerModel)

    def test_load_from_model_index(self):
        model = AutoModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="text_encoder", use_safetensors=False
        )
        assert isinstance(model, CLIPTextModel)

    def test_load_dynamic_module_from_local_path_with_subfolder(self):
        CUSTOM_MODEL_CODE = (
            "import torch\n"
            "from diffusers import ModelMixin, ConfigMixin\n"
            "from diffusers.configuration_utils import register_to_config\n"
            "\n"
            "class CustomModel(ModelMixin, ConfigMixin):\n"
            "    @register_to_config\n"
            "    def __init__(self, hidden_size=8):\n"
            "        super().__init__()\n"
            "        self.linear = torch.nn.Linear(hidden_size, hidden_size)\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.linear(x)\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            subfolder = "custom_model"
            model_dir = os.path.join(tmpdir, subfolder)
            os.makedirs(model_dir)

            with open(os.path.join(model_dir, "modeling.py"), "w") as f:
                f.write(CUSTOM_MODEL_CODE)

            config = {
                "_class_name": "CustomModel",
                "_diffusers_version": "0.0.0",
                "auto_map": {"AutoModel": "modeling.CustomModel"},
                "hidden_size": 8,
            }
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(config, f)

            torch.save({}, os.path.join(model_dir, "diffusion_pytorch_model.bin"))

            model = AutoModel.from_pretrained(tmpdir, subfolder=subfolder, trust_remote_code=True)
            assert model.__class__.__name__ == "CustomModel"
            assert model.config["hidden_size"] == 8


class TestAutoModelFromConfig(unittest.TestCase):
    @patch(
        "diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates",
        return_value=(MagicMock(), None),
    )
    def test_from_config_with_dict_diffusers_class(self, mock_get_class):
        config = {"_class_name": "UNet2DConditionModel", "sample_size": 64}
        mock_model = MagicMock()
        mock_get_class.return_value[0].from_config.return_value = mock_model

        result = AutoModel.from_config(config)

        mock_get_class.assert_called_once_with(
            library_name="diffusers",
            class_name="UNet2DConditionModel",
            importable_classes=unittest.mock.ANY,
            pipelines=None,
            is_pipeline_module=False,
        )
        mock_get_class.return_value[0].from_config.assert_called_once_with(config)
        assert result is mock_model

    @patch(
        "diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates",
        return_value=(MagicMock(), None),
    )
    @patch("diffusers.models.AutoModel.load_config", return_value={"_class_name": "UNet2DConditionModel"})
    def test_from_config_with_string_path(self, mock_load_config, mock_get_class):
        mock_model = MagicMock()
        mock_get_class.return_value[0].from_config.return_value = mock_model

        result = AutoModel.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet")

        mock_load_config.assert_called_once()
        assert result is mock_model

    def test_from_config_raises_on_missing_class_info(self):
        config = {"some_key": "some_value"}
        with self.assertRaises(ValueError, msg="Couldn't find a model class"):
            AutoModel.from_config(config)

    @patch(
        "diffusers.pipelines.pipeline_loading_utils.get_class_obj_and_candidates",
        return_value=(MagicMock(), None),
    )
    def test_from_config_with_model_type_routes_to_transformers(self, mock_get_class):
        config = {"model_type": "clip_text_model"}
        mock_model = MagicMock()
        mock_get_class.return_value[0].from_config.return_value = mock_model

        result = AutoModel.from_config(config)

        mock_get_class.assert_called_once_with(
            library_name="transformers",
            class_name="AutoModel",
            importable_classes=unittest.mock.ANY,
            pipelines=None,
            is_pipeline_module=False,
        )
        assert result is mock_model

    def test_from_config_raises_on_none(self):
        with self.assertRaises(ValueError, msg="Please provide a `pretrained_model_name_or_path_or_dict`"):
            AutoModel.from_config(None)
