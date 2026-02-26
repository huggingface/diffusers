import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from transformers import CLIPTextModel, LongformerModel

from diffusers import ConfigMixin
from diffusers.models import AutoModel, UNet2DConditionModel
from diffusers.models.modeling_utils import ModelMixin


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


class TestRegisterForAutoClass(unittest.TestCase):
    def test_register_for_auto_class_sets_attribute(self):
        class DummyModel(ModelMixin, ConfigMixin):
            config_name = "config.json"

        DummyModel.register_for_auto_class("AutoModel")
        self.assertEqual(DummyModel._auto_class, "AutoModel")

    def test_register_for_auto_class_rejects_unsupported(self):
        class DummyModel(ModelMixin, ConfigMixin):
            config_name = "config.json"

        with self.assertRaises(ValueError, msg="Only 'AutoModel' is supported"):
            DummyModel.register_for_auto_class("AutoPipeline")

    def test_auto_map_in_saved_config(self):
        class DummyModel(ModelMixin, ConfigMixin):
            config_name = "config.json"

        DummyModel.register_for_auto_class("AutoModel")
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_config(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

        self.assertIn("auto_map", config)
        self.assertIn("AutoModel", config["auto_map"])
        module_name = DummyModel.__module__.split(".")[-1]
        self.assertEqual(config["auto_map"]["AutoModel"], f"{module_name}.DummyModel")

    def test_no_auto_map_without_register(self):
        class DummyModel(ModelMixin, ConfigMixin):
            config_name = "config.json"

        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_config(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

        self.assertNotIn("auto_map", config)
