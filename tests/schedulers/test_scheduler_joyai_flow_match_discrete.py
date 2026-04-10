import tempfile

import torch

from diffusers import JoyAIFlowMatchDiscreteScheduler
from diffusers.utils import logging

from .test_schedulers import CaptureLogger


def test_joyai_scheduler_roundtrip_config_has_no_unexpected_warning():
    scheduler = JoyAIFlowMatchDiscreteScheduler(num_train_timesteps=1000, shift=4.0, reverse=True)
    logger = logging.get_logger("diffusers.configuration_utils")

    with tempfile.TemporaryDirectory() as tmpdirname:
        scheduler.save_config(tmpdirname)
        with CaptureLogger(logger) as cap_logger:
            config = JoyAIFlowMatchDiscreteScheduler.load_config(tmpdirname)
            reloaded = JoyAIFlowMatchDiscreteScheduler.from_config(config)

    assert isinstance(reloaded, JoyAIFlowMatchDiscreteScheduler)
    assert cap_logger.out == ""


def test_joyai_scheduler_reloaded_instance_supports_step():
    scheduler = JoyAIFlowMatchDiscreteScheduler(num_train_timesteps=1000, shift=4.0, reverse=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        scheduler.save_config(tmpdirname)
        reloaded = JoyAIFlowMatchDiscreteScheduler.from_pretrained(tmpdirname)

    reloaded.set_timesteps(2)
    sample = torch.zeros(1, 2, 2)
    model_output = torch.zeros_like(sample)
    prev_sample = reloaded.step(model_output, reloaded.timesteps[0], sample, return_dict=False)[0]

    assert prev_sample.shape == sample.shape
