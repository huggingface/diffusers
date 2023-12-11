import torch

from diffusers import UFOGenScheduler

from .test_schedulers import SchedulerCommonTest


class UFOGenSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (UFOGenScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for num_train_timesteps, timestep in zip([1, 5, 100, 1000], [0, 0, 0, 0]):
            self.check_over_configs(num_train_timesteps=num_train_timesteps, time_step=timestep)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "sample", "v_prediction"]:
                self.check_over_configs(
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)
    
    def full_loop(self, num_inference_steps=10, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = torch.manual_seed(0)

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator=generator).prev_sample
        
        return sample

    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 201.3429) < 1e-2
        assert abs(result_mean.item() - 0.2622) < 1e-3
    
    def test_full_loop_no_noise_onestep(self):
        sample = self.full_loop(num_inference_steps=1)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 61.5819) < 1e-2
        assert abs(result_mean.item() - 0.0802) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 141.1963) < 1e-2
        assert abs(result_mean.item() - 0.1838) < 1e-3

    def test_full_loop_with_noise(self, num_inference_steps=10):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = torch.manual_seed(0)

        t_start = num_inference_steps - 2
        scheduler.set_timesteps(num_inference_steps)

        # add noise
        noise = self.dummy_noise_deter
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for t in timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 350.1980) < 1e-2, f" expected result sum 387.9466, but get {result_sum}"
        assert abs(result_mean.item() - 0.4560) < 1e-3, f" expected result mean 0.5051, but get {result_mean}"

    def test_custom_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]

        scheduler.set_timesteps(timesteps=timesteps)

        scheduler_timesteps = scheduler.timesteps

        for i, timestep in enumerate(scheduler_timesteps):
            if i == len(timesteps) - 1:
                expected_prev_t = -1
            else:
                expected_prev_t = timesteps[i + 1]

            prev_t = scheduler.previous_timestep(timestep)
            prev_t = prev_t.item()

            self.assertEqual(prev_t, expected_prev_t)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 51, 0]

        with self.assertRaises(ValueError, msg="`custom_timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `custom_timesteps`."):
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
