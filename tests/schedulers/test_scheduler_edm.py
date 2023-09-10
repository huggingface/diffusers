import torch

from diffusers import KarrasEDMScheduler
from diffusers.utils import torch_device

from .test_schedulers import SchedulerCommonTest


class KarrasEDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KarrasEDMScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 40,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_clip_sample(self):
        for clip_sample_range in [1.0, 2.0, 3.0]:
            self.check_over_configs(clip_sample_range=clip_sample_range, clip_sample=True)

    def test_prediction_type(self):
        for prediction_type in ["edm", "epsilon", "v_prediction", "sample"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [39, 30, 12, 15, 0]

        with self.assertRaises(ValueError, msg="`timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [39, 30, 12, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `timesteps`."):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    def test_custom_timesteps_too_large(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [scheduler.config.num_train_timesteps]

        with self.assertRaises(
            ValueError,
            msg="`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}",
        ):
            scheduler.set_timesteps(timesteps=timesteps)
    
    def test_full_loop_no_noise(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 0.00053180) < 1e-8
        assert abs(result_mean.item() - 6.92451861e-07) < 1e-8

    def test_full_loop_no_noise_cm(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler_config["precondition_type"] = "cm_edm"
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 608.5814) < 1e-2
        assert abs(result_mean.item() - 0.7924) < 1e-3

    def test_full_loop_device(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for t in scheduler.timesteps:
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 0.00053180) < 1e-8
            assert abs(result_mean.item() - 6.92451861e-07) < 1e-8
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 6.92451861e-07) < 1e-8
        else:
            # CUDA
            assert abs(result_sum.item() - 0.00053180) < 1e-8
            assert abs(result_mean.item() - 6.92451861e-07) < 1e-8
    
    def test_full_loop_device_cm(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler_config["precondition_type"] = "cm_edm"
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for t in scheduler.timesteps:
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 608.5814) < 1e-2
            assert abs(result_mean.item() - 0.7924) < 1e-3
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 0.7924) < 1e-2
        else:
            # CUDA
            assert abs(result_sum.item() - 608.5814) < 1e-2
            assert abs(result_mean.item() - 0.7924) < 1e-3
