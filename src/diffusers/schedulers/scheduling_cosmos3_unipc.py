# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch

from .scheduling_unipc_multistep import UniPCMultistepScheduler


class Cosmos3EdgeUniPCMultistepScheduler(UniPCMultistepScheduler):
    """UniPC scheduler with the flow schedule and arithmetic used by Cosmos3 Edge."""

    def set_cosmos3_edge_native_flow_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        shift: float | None = None,
    ) -> None:
        if shift is None:
            shift = float(self.config.flow_shift)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")

        sigma_max = torch.tensor(1 - 1 / self.config.num_train_timesteps, dtype=torch.float32).item()
        sigmas = np.linspace(sigma_max, 0, num_inference_steps + 1).copy()[:-1]
        self.register_to_config(use_karras_sigmas=False, use_flow_sigmas=True, flow_shift=shift)
        super().set_timesteps(num_inference_steps, device=device, sigmas=sigmas)

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if sample is None:
            sample = args[1]
        if order is None:
            order = args[2]

        model_output_list = self.model_outputs
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            return self.solver_p.step(model_output, s0, x).prev_sample

        device = sample.device
        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        r = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            r.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        r = torch.stack(r)
        b = torch.tensor(b, device=device)
        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(r[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            d1s = None

        if self.predict_x0:
            x_t = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t - alpha_t * b_h * pred_res
        else:
            x_t = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
            x_t = x_t - sigma_t * b_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor | None = None,
        this_sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if last_sample is None:
            last_sample = args[1]
        if this_sample is None:
            this_sample = args[2]
        if order is None:
            order = args[3]

        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output
        device = this_sample.device
        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        r = []
        b = []
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        if self.config.solver_type == "bh1":
            b_h = hh
        elif self.config.solver_type == "bh2":
            b_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            r.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        r = torch.stack(r)
        b = torch.tensor(b, device=device)
        if len(d1s) > 0:
            d1s = torch.stack(d1s, dim=1)
        else:
            d1s = None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(r, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * (model_t - m0))
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
            x_t = x_t_ - sigma_t * b_h * (corr_res + rhos_c[-1] * (model_t - m0))
        return x_t.to(x.dtype)
