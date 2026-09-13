import torch

from diffusers import VDMScheduler

from .test_schedulers import SchedulerCommonTest


class VDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (VDMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [None, 1, 5, 100, 1000]:
            self.check_over_configs(time_step=0.0 if timesteps is None else 0, num_train_timesteps=timesteps)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2", "sigmoid"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "sample"]:
                self.check_over_configs(
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)  # Discrete
            self.check_over_forward(time_step=t / 1000)  # Continuous

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        print(sample.abs().sum())
        generator = torch.manual_seed(0)

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)
            if t == len(scheduler) - 1:
                print(residual.abs().sum())

            # 2. predict previous mean of sample x_t-1
            sample = scheduler.step(residual, t, sample, generator=generator).prev_sample
            if t == len(scheduler) - 1:
                print(sample.abs().sum())

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 256.4699) < 1e-2, f" expected result sum 256.4699, but get {result_sum}"
        assert abs(result_mean.item() - 0.3339) < 1e-3, f" expected result mean 0.3339, but get {result_mean}"

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)
        t_start = num_trained_timesteps - 2

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = torch.manual_seed(0)

        # add noise
        noise = self.dummy_noise_deter
        timesteps = scheduler.timesteps[t_start:]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for t in timesteps:
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 387.6469) < 1e-2, f" expected result sum 387.6469, but get {result_sum}"
        assert abs(result_mean.item() - 0.5051) < 1e-3, f" expected result mean 0.5051, but get {result_mean}"
