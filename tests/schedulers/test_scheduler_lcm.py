import torch

from diffusers import LCMScheduler

from .test_schedulers import SchedulerCommonTest


class LCMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LCMScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "beta_start": 0.00085,
            "beta_end": 0.0120,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        }

        config.update(**kwargs)
        return config
