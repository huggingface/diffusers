from unittest.mock import patch

from diffusers import DiffusionPipeline, JoyAIImagePipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.joyai_image import pipeline_joyai_image


class _DummyModule:
    pass


def test_joyai_pipeline_uses_base_from_pretrained():
    assert JoyAIImagePipeline.from_pretrained.__func__ is DiffusionPipeline.from_pretrained.__func__


def test_joyai_pipeline_does_not_expose_source_loader_api():
    assert not hasattr(JoyAIImagePipeline, "from_joyai_sources")


def test_joyai_pipeline_module_does_not_expose_raw_source_helpers():
    assert not hasattr(pipeline_joyai_image, "load_joyai_components")


def test_joyai_pipeline_keeps_passed_processor_without_reloading():
    pipe = object.__new__(JoyAIImagePipeline)
    pipe._internal_dict = FrozenDict({})
    pipe.args = type("Args", (), {"text_encoder_arch_config": {"params": {"text_encoder_ckpt": "/tmp/raw"}}})()
    pipe.vae = type("VAE", (), {"ffactor_spatial": 8, "ffactor_temporal": 4})()

    registered = {}

    def fake_register_modules(**kwargs):
        registered.update(kwargs)
        for key, value in kwargs.items():
            setattr(pipe, key, value)

    pipe.register_modules = fake_register_modules

    processor = _DummyModule()
    with patch(
        "diffusers.pipelines.joyai_image.pipeline_joyai_image.AutoProcessor.from_pretrained"
    ) as mock_from_pretrained:
        JoyAIImagePipeline.__init__(
            pipe,
            vae=pipe.vae,
            text_encoder=_DummyModule(),
            tokenizer=_DummyModule(),
            transformer=_DummyModule(),
            scheduler=_DummyModule(),
            processor=processor,
            args=pipe.args,
        )

    assert pipe.qwen_processor is processor
    mock_from_pretrained.assert_not_called()
    assert registered["processor"] is processor
