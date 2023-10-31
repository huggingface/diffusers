import numpy as np
import torch
import torch.nn as nn


"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class VLBLoss(nn.Module):
    def __init__(self, discrete_L0: bool = True):
        """Compute the VLB Loss

        Args:
            discrete_L0 (bool, optional): Bool. Defaults to True.
            Compute L0 as a discretized log likelihood or as a MSE with p_mean.
            In case of LDM, we use MSE
            In case of DDM, we use NLL
        """
        super().__init__()
        self.discrete_L0 = discrete_L0

    def forward(
        self,
        p_mean: torch.Tensor,
        p_log_var: torch.Tensor,
        q_mean: torch.Tensor,
        q_log_var: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ):
        """KL Divergence from log_var
        We take the equation here (based on sigma = sqrt(log))

        log(x^1/2) = 1/2 . log(x)

        Args:
            p_mean (torch.Tensor): _description_
            p_log_var (torch.Tensor): _description_
            q_mean (torch.Tensor): _description_
            q_log_var (torch.Tensor): _description_
            x_0 (torch.Tensor): starting image
            t (torch.Tensor): timesteps

        Returns:
            _type_: _description_
        """
        # We detach p_pean like in : https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf
        # In order the VLB to contribute only to train the variance term.
        # We compute L_{1} + ... + L_{t-1}
        kld = (
            -0.5
            + 0.5 * (q_log_var - p_log_var)
            + 0.5
            * (torch.exp(p_log_var) / torch.exp(q_log_var) + (p_mean.detach() - q_mean) ** 2 / torch.exp(q_log_var))
        )
        kl_div_mean = kld.reshape(x_0.shape[0], -1).mean(dim=1)
        # We compute L_{0}
        if self.discrete_L0:
            l_0 = -discretized_gaussian_log_likelihood(x_0, p_mean, 0.5 * p_log_var)
        else:
            l_0 = nn.MSELoss(reduction="none")(x_0, p_mean)
        l_0_mean = l_0.reshape(x_0.shape[0], -1).mean(dim=1)
        vlb_loss_batch = torch.where((t > 0), kl_div_mean, l_0_mean).mean()
        return vlb_loss_batch


if __name__ == "__main__":
    p_mean = torch.randn((2, 3, 122, 122))
    p_var = torch.randn((2, 3, 122, 122))
    q_mean = torch.randn((2, 3, 122, 122))
    q_var = torch.randn((2, 3, 122, 122))
    x_0 = torch.randn((2, 3, 122, 122))
    t = torch.tensor([0, 2])
    loss = VLBLoss(False).forward(p_mean, p_var, q_mean, q_var, x_0, t)
    loss = VLBLoss(True).forward(p_mean, p_var, q_mean, q_var, x_0, t)
    print("terminé")
