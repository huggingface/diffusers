import tempfile

import torch

from diffusers import PNDMScheduler

from .test_schedulers import SchedulerCommonTest


class PNDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (PNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_prk(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step_plms(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.prk_timesteps):
            residual = model(sample, t)
            sample = scheduler.step_prk(residual, t, sample).prev_sample

        for i, t in enumerate(scheduler.plms_timesteps):
            residual = model(sample, t)
            sample = scheduler.step_plms(residual, t, sample).prev_sample

        return sample

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]
            scheduler.ets = dummy_past_residuals[:]

            output_0 = scheduler.step_prk(residual, 0, sample, **kwargs).prev_sample
            output_1 = scheduler.step_prk(residual, 1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

            output_0 = scheduler.step_plms(residual, 0, sample, **kwargs).prev_sample
            output_1 = scheduler.step_plms(residual, 1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(10)
        assert torch.equal(
            scheduler.timesteps,
            torch.LongTensor(
                [901, 851, 851, 801, 801, 751, 751, 701, 701, 651, 651, 601, 601, 501, 401, 301, 201, 101, 1]
            ),
        )

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001], [0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for t in [1, 5, 10]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
            self.check_over_forward(num_inference_steps=num_inference_steps)

    def test_pow_of_3_inference_steps(self):
        # earlier version of set_timesteps() caused an error indexing alpha's with inference steps as power of 3
        num_inference_steps = 27

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.set_timesteps(num_inference_steps)

            # before power of 3 fix, would error on first step, so we only need to do two
            for i, t in enumerate(scheduler.prk_timesteps[:2]):
                sample = scheduler.step_prk(residual, t, sample).prev_sample

    def test_inference_plms_no_past_residuals(self):
        with self.assertRaises(ValueError):
            scheduler_class = self.scheduler_classes[0]
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.step_plms(self.dummy_sample, 1, self.dummy_sample).prev_sample

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 198.1318) < 1e-2
        assert abs(result_mean.item() - 0.2580) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 67.3986) < 1e-2
        assert abs(result_mean.item() - 0.0878) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 230.0399) < 1e-2
        assert abs(result_mean.item() - 0.2995) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 186.9482) < 1e-2
        assert abs(result_mean.item() - 0.2434) < 1e-3
