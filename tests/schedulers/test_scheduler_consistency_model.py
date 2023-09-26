import torch

from diffusers import CMStochasticIterativeScheduler

from .test_schedulers import SchedulerCommonTest


class CMStochasticIterativeSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (CMStochasticIterativeScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 201,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
        }

        config.update(**kwargs)
        return config

    # Override test_step_shape to add CMStochasticIterativeScheduler-specific logic regarding timesteps
    # Problem is that we don't know two timesteps that will always be in the timestep schedule from only the scheduler
    # config; scaled sigma_max is always in the timestep schedule, but sigma_min is in the sigma schedule while scaled
    # sigma_min is not in the timestep schedule
    def test_step_shape(self):
        num_inference_steps = 10

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        timestep_0 = scheduler.timesteps[0]
        timestep_1 = scheduler.timesteps[1]

        sample = self.dummy_sample
        residual = 0.1 * sample

        output_0 = scheduler.step(residual, timestep_0, sample).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample).prev_sample

        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_clip_denoised(self):
        for clip_denoised in [True, False]:
            self.check_over_configs(clip_denoised=clip_denoised)

    def test_full_loop_no_noise_onestep(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 1
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(timesteps):
            # 1. scale model input
            scaled_sample = scheduler.scale_model_input(sample, t)

            # 2. predict noise residual
            residual = model(scaled_sample, t)

            # 3. predict previous sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            sample = pred_prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 192.7614) < 1e-2
        assert abs(result_mean.item() - 0.2510) < 1e-3

    def test_full_loop_no_noise_multistep(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [106, 0]
        scheduler.set_timesteps(timesteps=timesteps)
        timesteps = scheduler.timesteps

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for t in timesteps:
            # 1. scale model input
            scaled_sample = scheduler.scale_model_input(sample, t)

            # 2. predict noise residual
            residual = model(scaled_sample, t)

            # 3. predict previous sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            sample = pred_prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 347.6357) < 1e-2
        assert abs(result_mean.item() - 0.4527) < 1e-3

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        t_start = 8

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        noise = self.dummy_noise_deter
        timesteps = scheduler.timesteps[t_start * scheduler.order :]

        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for t in timesteps:
            # 1. scale model input
            scaled_sample = scheduler.scale_model_input(sample, t)

            # 2. predict noise residual
            residual = model(scaled_sample, t)

            # 3. predict previous sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            sample = pred_prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 763.9186) < 1e-2, f" expected result sum 763.9186, but get {result_sum}"
        assert abs(result_mean.item() - 0.9947) < 1e-3, f" expected result mean 0.9947, but get {result_mean}"

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
