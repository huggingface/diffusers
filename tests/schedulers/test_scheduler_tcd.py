import torch

from diffusers import TCDScheduler

from .test_schedulers import SchedulerCommonTest


class TCDSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (TCDScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    def test_truncated_custom_schedule_terminal_step_preserves_generator(self):
        scheduler = TCDScheduler()
        scheduler.set_timesteps(timesteps=[999, 499, 19], strength=0.5, device="cpu")
        self.assertEqual(scheduler.timesteps.tolist(), [19])
        generator = torch.Generator(device="cpu").manual_seed(0)
        state = generator.get_state().clone()
        sample = torch.ones(1, 1, device="cpu")
        scheduler.step(torch.zeros_like(sample), scheduler.timesteps[0], sample, eta=0.3, generator=generator)
        self.assertTrue(torch.equal(generator.get_state(), state))

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.0120,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        }

        config.update(**kwargs)
        return config

    @property
    def default_num_inference_steps(self):
        return 10

    @property
    def default_valid_timestep(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)
        timestep = scheduler.timesteps[-1]
        return timestep

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            # 0 is not guaranteed to be in the timestep schedule, but timesteps - 1 is
            self.check_over_configs(time_step=timesteps - 1, num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(time_step=self.default_valid_timestep, beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear", "squaredcos_cap_v2"]:
            self.check_over_configs(time_step=self.default_valid_timestep, beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(time_step=self.default_valid_timestep, prediction_type=prediction_type)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(time_step=self.default_valid_timestep, clip_sample=clip_sample)

    def test_thresholding(self):
        self.check_over_configs(time_step=self.default_valid_timestep, thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "v_prediction"]:
                self.check_over_configs(
                    time_step=self.default_valid_timestep,
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_time_indices(self):
        # Get default timestep schedule.
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        for t in timesteps:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        # Hardcoded for now
        for t, num_inference_steps in zip([99, 39, 39, 19], [10, 25, 26, 50]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    def full_loop(self, num_inference_steps=10, seed=0, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        eta = 0.0  # refer to gamma in the paper

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = torch.manual_seed(seed)
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, eta, generator).prev_sample

        return sample

    def test_full_loop_onestep_deter(self):
        sample = self.full_loop(num_inference_steps=1)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 29.8715) < 1e-3  # 0.0778918
        assert abs(result_mean.item() - 0.0389) < 1e-3

    def test_full_loop_multistep_deter(self):
        sample = self.full_loop(num_inference_steps=10)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 181.2040) < 1e-3
        assert abs(result_mean.item() - 0.2359) < 1e-3

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

    def test_rng_only_advances_for_effective_stochastic_steps(self):
        # Issue #15288: truncation may leave the actual last step before
        # num_inference_steps - 1. Also test zero noise on non-terminal steps.
        schedules = [
            ([999, 499, 19], 0.5, [19]),
            ([999, 799, 599, 399, 199], 0.6, [599, 399, 199]),
            ([999, 799, 599, 399, 199], 1.0, [999, 799, 599, 399, 199]),
            ([999, 0], 1.0, [999, 0]),
            ([999, 1, 0], 1.0, [999, 1, 0]),
        ]
        for requested, strength, expected in schedules:
            for prediction_type in ["epsilon", "sample", "v_prediction"]:
                for set_alpha_to_one in [True, False]:
                    for eta in [0.0, 1e-12, 0.3, 1.0]:
                        for return_dict in [False, True]:
                            with self.subTest(
                                requested=requested,
                                strength=strength,
                                prediction_type=prediction_type,
                                set_alpha_to_one=set_alpha_to_one,
                                eta=eta,
                                return_dict=return_dict,
                            ):
                                scheduler = TCDScheduler(
                                    prediction_type=prediction_type,
                                    set_alpha_to_one=set_alpha_to_one,
                                    original_inference_steps=1000,
                                )
                                scheduler.set_timesteps(timesteps=requested, strength=strength, device="cpu")
                                self.assertEqual(scheduler.timesteps.tolist(), expected)
                                generator = torch.Generator(device="cpu").manual_seed(0)
                                control = torch.Generator(device="cpu").manual_seed(0)
                                sample = torch.ones((1, 1), device="cpu")

                                for i, timestep in enumerate(scheduler.timesteps):
                                    before = generator.get_state().clone()
                                    result = scheduler.step(
                                        torch.zeros_like(sample),
                                        timestep,
                                        sample,
                                        eta=eta,
                                        generator=generator,
                                        return_dict=return_dict,
                                    )
                                    prev_sample, pred_noised = (
                                        (result.prev_sample, result.pred_noised_sample) if return_dict else result
                                    )
                                    self.assertTrue(torch.isfinite(prev_sample).all())
                                    has_next = i + 1 < len(expected)
                                    next_t = expected[i + 1] if has_next else 0
                                    timestep_s = int(torch.floor((1 - eta) * torch.tensor(next_t)).item())
                                    should_draw = eta > 0 and has_next and timestep_s < next_t
                                    moved = not torch.equal(before, generator.get_state())
                                    self.assertEqual(moved, should_draw)
                                    if should_draw:
                                        torch.randn(sample.shape, generator=control, device="cpu")
                                    else:
                                        torch.testing.assert_close(prev_sample, pred_noised, rtol=0, atol=0)
                                    sample = prev_sample

                                torch.testing.assert_close(
                                    torch.randn(4, generator=generator),
                                    torch.randn(4, generator=control),
                                    rtol=0,
                                    atol=0,
                                )

    def test_rng_standard_schedule_unchanged(self):
        for eta in [0.0, 1e-12, 0.3, 1.0]:
            scheduler = TCDScheduler(original_inference_steps=1000)
            scheduler.set_timesteps(num_inference_steps=4, device="cpu")
            generator = torch.Generator(device="cpu").manual_seed(0)
            control = torch.Generator(device="cpu").manual_seed(0)
            sample = torch.ones((1, 1), device="cpu")
            for i, timestep in enumerate(scheduler.timesteps):
                scheduler.step(torch.zeros_like(sample), timestep, sample, eta=eta, generator=generator)
                if i + 1 < len(scheduler.timesteps):
                    next_t = int(scheduler.timesteps[i + 1].item())
                    timestep_s = int(torch.floor((1 - eta) * torch.tensor(next_t)).item())
                    if eta > 0 and timestep_s < next_t:
                        torch.randn(sample.shape, generator=control)
            torch.testing.assert_close(
                torch.randn(4, generator=generator),
                torch.randn(4, generator=control),
                rtol=0,
                atol=0,
            )
