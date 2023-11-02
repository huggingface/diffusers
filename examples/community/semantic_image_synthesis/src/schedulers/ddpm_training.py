from typing import Union

import numpy as np
import torch

from diffusers.schedulers import DDPMScheduler


def pick_tensor(values: Union[torch.Tensor, np.ndarray, list], t: Union[torch.Tensor, np.ndarray, int]):
    """Returns a specific index t from a passed list of values.

    Ot will reshape the output regarding the batch size...

    Args:
        values (_type_): _description_
        t (_type_): _description_

    Returns :
        values at indices t resized as x_shape

    """
    assert isinstance(values, (list, torch.Tensor, np.ndarray)), "values should be a list, a tensor or an array"
    assert isinstance(t, (int, torch.Tensor, np.ndarray)), "t should be an int, a tensor or an array"
    values = torch.tensor(values) if not isinstance(values, torch.Tensor) else values
    t = torch.tensor(t) if isinstance(t, np.ndarray) else t
    # We remove items where t = -1
    t = t.clamp(min=0)
    if isinstance(t, torch.Tensor):
        if len(t.shape) == 0:
            indices = torch.tensor([t], dtype=torch.long, device=values.device)
        else:
            indices = t.long().to(values.device)
    else:
        # In this case, t is like a scalar.
        indices = torch.tensor([t], dtype=torch.long, device=values.device)
    out = values.gather(-1, indices)
    return out


def adapt_tensor(vector: torch.Tensor, x: torch.Tensor):
    b = x.shape[0]
    assert len(vector) == b, "vector should be the same lenght as batch"
    return vector.reshape(b, *((1,) * (len(x.shape) - 1))).type_as(x).to(x.device)


class DDPMTrainingScheduler(DDPMScheduler):
    """Override DDPM Scheduler in order to handle training methods...

    Args:
        DDPMScheduler (_type_): _description_
    """

    @property
    def snr(self):
        """We compute the snr based on :
        https://arxiv.org/abs/2303.09556

        Returns:
            snr matrix
        """
        if not hasattr(self, "_snr"):
            signal_squared = self.alphas_cumprod
            noise_squared = 1.0 - self.alphas_cumprod
            self._snr = signal_squared / noise_squared
        return self._snr

    def get_minsnr_k_weight(self, timesteps: torch.Tensor, k: float = 5.0):
        """We compute the min-snr_k like in the paper :
        https://arxiv.org/abs/2303.09556

        Args:
            k (float): _description_
            timesteps (torch.Tensor): _description_

        """
        snr = pick_tensor(self.snr, timesteps).to(timesteps.device)
        min_snr_k = torch.stack([snr, float(k) * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        return min_snr_k

    def previous_timesteps(self, timesteps: torch.Tensor):
        """Get the previous timestep, returns 0 for timestep 0.

        Args:
            timesteps (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        return torch.tensor([self.previous_timestep(t) for t in timesteps]).long()

    def _get_q_mean_variance(self, timesteps: torch.Tensor, sample_0: torch.Tensor, sample_t: torch.Tensor):
        """Returns the posterior mean variance
        like there :
        https://arxiv.org/pdf/2006.11239.pdf (7)

        q(x_{t-1} | x_t,x_0)

        Args:
            timesteps (torch.Tensor|list): current timestep matrix
            sample_0 (torch.Tensor): sample at step 0
            sample_t (torch.Tensor): sample at step t
        """
        assert isinstance(timesteps, (torch.Tensor, list, np.ndarray)), "timesteps should be a list..."
        prev_ts = self.previous_timesteps(timesteps)
        # To compute β_pos : https://arxiv.org/pdf/2006.11239.pdf (7)
        alpha_prod_ts = pick_tensor(self.alphas_cumprod, timesteps)
        alpha_prod_ts_prev = torch.where(
            prev_ts > 0, pick_tensor(self.alphas_cumprod, prev_ts), torch.ones_like(prev_ts)
        )
        current_alpha_ts = alpha_prod_ts / alpha_prod_ts_prev
        current_beta_ts = 1 - current_alpha_ts
        posterior_var = (1 - alpha_prod_ts_prev) / (1 - alpha_prod_ts) * current_beta_ts
        # In order to have the same shape than every other
        posterior_var = adapt_tensor(posterior_var, sample_0) * torch.ones_like(sample_0)
        # To compute µ_pos : https://arxiv.org/pdf/2006.11239.pdf (7)
        coef_x0 = torch.sqrt(alpha_prod_ts_prev) * current_beta_ts / (1 - alpha_prod_ts)
        coef_xt = torch.sqrt(current_alpha_ts) * (1 - alpha_prod_ts_prev) / (1 - alpha_prod_ts)
        coef_x0 = adapt_tensor(coef_x0, sample_0)
        coef_xt = adapt_tensor(coef_xt, sample_t)

        posterior_mean = coef_x0 * sample_0 + coef_xt * sample_t
        # We clamp posterior var for stability...
        posterior_var = posterior_var.clamp(1e-5)
        # We return
        return posterior_mean, posterior_var, torch.log(posterior_var)

    def _get_variances(self, timesteps: torch.Tensor, predicted_variances: torch.Tensor = None):
        """Get multiple variances at a time.

        Args:
            timesteps (torch.Tensor): matrix containing timesteps.
            predicted_variances (torch.Tensor): variances predicted by the model if necessary
        Returns:
            _type_: _description_
        """
        assert isinstance(timesteps, (torch.Tensor, list, np.ndarray)), "timesteps should be a list..."
        prev_ts = self.previous_timesteps(timesteps)

        alpha_prod_ts = pick_tensor(self.alphas_cumprod, timesteps)
        alpha_prod_ts_prev = torch.where(
            prev_ts > 0, pick_tensor(self.alphas_cumprod, prev_ts), torch.ones_like(prev_ts)
        )
        current_beta_ts = 1 - alpha_prod_ts / alpha_prod_ts_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variances = (1 - alpha_prod_ts_prev) / (1 - alpha_prod_ts) * current_beta_ts

        # we always take the log of variance, so clamp it to ensure it's not 0
        variances = torch.clamp(variances, min=1e-5)  # Important for mixed precision not to have a too low value.
        current_beta_ts = torch.clamp(variances, min=1e-5)

        variance_type = self.config.variance_type
        if variance_type == "fixed_small":
            variances = variances
            log_variances = torch.log(variances)
        elif variance_type == "fixed_large":
            log_variances = torch.log(current_beta_ts)
            variances = current_beta_ts
        elif variance_type == "learned":
            log_variances = predicted_variances
            variances = torch.exp(log_variances)
        elif variance_type == "learned_range":
            min_log = torch.log(variances)
            max_log = torch.log(current_beta_ts)
            frac = (predicted_variances + 1) / 2
            log_variances = frac * adapt_tensor(max_log, frac) + (1 - frac) * adapt_tensor(min_log, frac)
            variances = torch.exp(log_variances)

        return variances, log_variances

    def _get_p_mean_variance(self, model_output: torch.Tensor, timesteps: torch.Tensor, scale: float = 1.0):
        """Compute the predicted mean variance
        p(x_{t-1} | x_t)

        Args:
            model_output (torch.Tensor):
                output of the model is :
                    [(Optional 2)*B,(Optional 2)*C,H,W].
                conditional and unconditional prediction on dim 0
                mean and variance on dim 1

            timesteps (torch.Tensor): timesteps corresponding to the output

        Returns:
            model_mean
            model_var
            model_log_var
        """
        # We separate mean and var if necessary...
        model_var = None
        if "learned" in self.variance_type:
            model_mean, model_var = model_output.chunk(2, dim=1)
        else:
            model_mean = model_output

        # We compute scale if necessary...
        if scale > 1.0:
            model_mean_cond, model_mean_ucond = model_mean.chunk(2, dim=0)
            model_mean = model_mean_ucond + scale * (model_mean_cond - model_mean_ucond)
            # We only use the first part to get the variance
            model_var, model_log_var = self._get_variances(timesteps, model_var.chunk(2, dim=0)[0])
            return model_mean, model_var, model_log_var
        else:
            model_var, model_log_var = self._get_variances(timesteps, model_var)
            return model_mean, model_var, model_log_var


if __name__ == "__main__":
    noise_scheduler = DDPMTrainingScheduler(variance_type="learned_range")
    x0 = torch.ones((4, 3, 128, 128))
    xt = torch.ones((4, 3, 128, 128))
    m_pred = torch.randn((4, 6, 128, 128))
    t_tensor = torch.tensor([0, 13, 18, 32])

    mean, var, logvar = noise_scheduler._get_q_mean_variance(t_tensor, x0, xt)
    mean_p, var_p, logvar = noise_scheduler._get_p_mean_variance(m_pred, t_tensor)
    print("terminé")
