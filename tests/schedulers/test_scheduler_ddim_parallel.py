# Copyright 2025 ParaDiGMS authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from diffusers import DDIMParallelScheduler

from .test_schedulers import SchedulerCommonTest


class DDIMParallelSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDIMParallelScheduler,)
    forward_default_kwargs = (("eta", 0.0), ("num_inference_steps", 50))

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

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps, eta = 10, 0.0

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, eta).prev_sample

        return sample

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(5)
        assert torch.equal(scheduler.timesteps, torch.LongTensor([801, 601, 401, 201, 1]))

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_timestep_spacing(self):
        for timestep_spacing in ["trailing", "leading"]:
            self.check_over_configs(timestep_spacing=timestep_spacing)

    def test_rescale_betas_zero_snr(self):
        for rescale_betas_zero_snr in [True, False]:
            self.check_over_configs(rescale_betas_zero_snr=rescale_betas_zero_snr)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "v_prediction"]:
                self.check_over_configs(
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_time_indices(self):
        for t in [1, 10, 49]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    def test_eta(self):
        for t, eta in zip([1, 10, 49], [0.0, 0.5, 1.0]):
            self.check_over_forward(time_step=t, eta=eta)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert torch.sum(torch.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-5
        assert torch.sum(torch.abs(scheduler._get_variance(420, 400) - 0.14771)) < 1e-5
        assert torch.sum(torch.abs(scheduler._get_variance(980, 960) - 0.32460)) < 1e-5
        assert torch.sum(torch.abs(scheduler._get_variance(0, 0) - 0.0)) < 1e-5
        assert torch.sum(torch.abs(scheduler._get_variance(487, 486) - 0.00979)) < 1e-5
        assert torch.sum(torch.abs(scheduler._get_variance(999, 998) - 0.02)) < 1e-5

    def test_batch_step_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps, eta = 10, 0.0
        scheduler.set_timesteps(num_inference_steps)

        model = self.dummy_model()
        sample1 = self.dummy_sample_deter
        sample2 = self.dummy_sample_deter + 0.1
        sample3 = self.dummy_sample_deter - 0.1

        per_sample_batch = sample1.shape[0]
        samples = torch.stack([sample1, sample2, sample3], dim=0)
        timesteps = torch.arange(num_inference_steps)[0:3, None].repeat(1, per_sample_batch)

        residual = model(samples.flatten(0, 1), timesteps.flatten(0, 1))
        pred_prev_sample = scheduler.batch_step_no_noise(residual, timesteps.flatten(0, 1), samples.flatten(0, 1), eta)

        result_sum = torch.sum(torch.abs(pred_prev_sample))
        result_mean = torch.mean(torch.abs(pred_prev_sample))

        assert abs(result_sum.item() - 1147.7904) < 1e-2
        assert abs(result_mean.item() - 0.4982) < 1e-3

    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 172.0067) < 1e-2
        assert abs(result_mean.item() - 0.223967) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 52.5302) < 1e-2
        assert abs(result_mean.item() - 0.0684) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 149.8295) < 1e-2
        assert abs(result_mean.item() - 0.1951) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 149.0784) < 1e-2
        assert abs(result_mean.item() - 0.1941) < 1e-3

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps, eta = 10, 0.0
        t_start = 8

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_timesteps(num_inference_steps)

        # add noise
        noise = self.dummy_noise_deter
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for t in timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, eta).prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 354.5418) < 1e-2, f" expected result sum 354.5418, but get {result_sum}"
        assert abs(result_mean.item() - 0.4616) < 1e-3, f" expected result mean 0.4616, but get {result_mean}"
