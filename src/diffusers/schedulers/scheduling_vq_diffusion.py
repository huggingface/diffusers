# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


@dataclass
class VQDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
            Computed sample x_{t-1} of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.LongTensor


def index_to_log_onehot(x: torch.LongTensor, num_classes: int) -> torch.Tensor:
    """
    Convert batch of vector of class indices into batch of log onehot vectors

    Args:
        x (`torch.LongTensor` of shape `(batch size, vector length)`):
            Batch of class indices

        num_classes (`int`):
            number of classes to be used for the onehot vectors

    Returns:
        `torch.Tensor` of shape `(batch size, num classes, vector length)`:
            Log onehot vectors
    """
    x_onehot = F.one_hot(x, num_classes)
    x_onehot = x_onehot.permute(0, 2, 1)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def gumbel_noised(logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """
    Apply gumbel noise to `logits`
    """
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    noised = gumbel_noise + logits
    return noised


def alpha_schedules(num_diffusion_timesteps: int, alpha_cum_start=0.99999, alpha_cum_end=0.000009):
    """
    Cumulative and non-cumulative alpha schedules.

    See section 4.1.
    """
    att = (
        np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1) * (alpha_cum_end - alpha_cum_start)
        + alpha_cum_start
    )
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    att = np.concatenate((att[1:], [1]))
    return at, att


def gamma_schedules(num_diffusion_timesteps: int, gamma_cum_start=0.000009, gamma_cum_end=0.99999):
    """
    Cumulative and non-cumulative gamma schedules.

    See section 4.1.
    """
    ctt = (
        np.arange(0, num_diffusion_timesteps) / (num_diffusion_timesteps - 1) * (gamma_cum_end - gamma_cum_start)
        + gamma_cum_start
    )
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    ctt = np.concatenate((ctt[1:], [0]))
    return ct, ctt


class VQDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    A scheduler for vector quantized diffusion.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_vec_classes (`int`):
            The number of classes of the vector embeddings of the latent pixels. Includes the class for the masked
            latent pixel.
        num_train_timesteps (`int`, defaults to 100):
            The number of diffusion steps to train the model.
        alpha_cum_start (`float`, defaults to 0.99999):
            The starting cumulative alpha value.
        alpha_cum_end (`float`, defaults to 0.00009):
            The ending cumulative alpha value.
        gamma_cum_start (`float`, defaults to 0.00009):
            The starting cumulative gamma value.
        gamma_cum_end (`float`, defaults to 0.99999):
            The ending cumulative gamma value.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_vec_classes: int,
        num_train_timesteps: int = 100,
        alpha_cum_start: float = 0.99999,
        alpha_cum_end: float = 0.000009,
        gamma_cum_start: float = 0.000009,
        gamma_cum_end: float = 0.99999,
    ):
        self.num_embed = num_vec_classes

        # By convention, the index for the mask class is the last class index
        self.mask_class = self.num_embed - 1

        at, att = alpha_schedules(num_train_timesteps, alpha_cum_start=alpha_cum_start, alpha_cum_end=alpha_cum_end)
        ct, ctt = gamma_schedules(num_train_timesteps, gamma_cum_start=gamma_cum_start, gamma_cum_end=gamma_cum_end)

        num_non_mask_classes = self.num_embed - 1
        bt = (1 - at - ct) / num_non_mask_classes
        btt = (1 - att - ctt) / num_non_mask_classes

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)

        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        self.log_at = log_at.float()
        self.log_bt = log_bt.float()
        self.log_ct = log_ct.float()
        self.log_cumprod_at = log_cumprod_at.float()
        self.log_cumprod_bt = log_cumprod_bt.float()
        self.log_cumprod_ct = log_cumprod_ct.float()

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps and diffusion process parameters (alpha, beta, gamma) should be moved
                to.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.log_at = self.log_at.to(device)
        self.log_bt = self.log_bt.to(device)
        self.log_ct = self.log_ct.to(device)
        self.log_cumprod_at = self.log_cumprod_at.to(device)
        self.log_cumprod_bt = self.log_cumprod_bt.to(device)
        self.log_cumprod_ct = self.log_cumprod_ct.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[VQDiffusionSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by the reverse transition distribution. See
        [`~VQDiffusionScheduler.q_posterior`] for more details about how the distribution is computer.

        Args:
            log_p_x_0: (`torch.Tensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.
            t (`torch.long`):
                The timestep that determines which transition matrices are used.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            generator (`torch.Generator`, or `None`):
                A random number generator for the noise applied to `p(x_{t-1} | x_t)` before it is sampled from.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] or
                `tuple`.

        Returns:
            [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_vq_diffusion.VQDiffusionSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        if timestep == 0:
            log_p_x_t_min_1 = model_output
        else:
            log_p_x_t_min_1 = self.q_posterior(model_output, sample, timestep)

        log_p_x_t_min_1 = gumbel_noised(log_p_x_t_min_1, generator)

        x_t_min_1 = log_p_x_t_min_1.argmax(dim=1)

        if not return_dict:
            return (x_t_min_1,)

        return VQDiffusionSchedulerOutput(prev_sample=x_t_min_1)

    def q_posterior(self, log_p_x_0, x_t, t):
        """
        Calculates the log probabilities for the predicted classes of the image at timestep `t-1`:

        ```
        p(x_{t-1} | x_t) = sum( q(x_t | x_{t-1}) * q(x_{t-1} | x_0) * p(x_0) / q(x_t | x_0) )
        ```

        Args:
            log_p_x_0 (`torch.Tensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            t (`torch.Long`):
                The timestep that determines which transition matrix is used.

        Returns:
            `torch.Tensor` of shape `(batch size, num classes, num latent pixels)`:
                The log probabilities for the predicted classes of the image at timestep `t-1`.
        """
        log_onehot_x_t = index_to_log_onehot(x_t, self.num_embed)

        log_q_x_t_given_x_0 = self.log_Q_t_transitioning_to_known_class(
            t=t, x_t=x_t, log_onehot_x_t=log_onehot_x_t, cumulative=True
        )

        log_q_t_given_x_t_min_1 = self.log_Q_t_transitioning_to_known_class(
            t=t, x_t=x_t, log_onehot_x_t=log_onehot_x_t, cumulative=False
        )

        # p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0)          ...      p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0)
        #               .                    .                                   .
        #               .                            .                           .
        #               .                                      .                 .
        # p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})  ...      p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})
        q = log_p_x_0 - log_q_x_t_given_x_0

        # sum_0 = p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) + ... + p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}), ... ,
        # sum_n = p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) + ... + p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1})
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)

        # p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_0          ...      p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_n
        #                        .                             .                                   .
        #                        .                                     .                           .
        #                        .                                               .                 .
        # p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_0  ...      p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_n
        q = q - q_log_sum_exp

        # (p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}          ...      (p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}
        #                                         .                                                .                                              .
        #                                         .                                                        .                                      .
        #                                         .                                                                  .                            .
        # (p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}  ...      (p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}
        # c_cumulative_{t-1}                                                                                 ...      c_cumulative_{t-1}
        q = self.apply_cumulative_transitions(q, t - 1)

        # ((p_0(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_0) * sum_0              ...      ((p_n(x_0=C_0 | x_t) / q(x_t | x_0=C_0) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_0) * sum_n
        #                                                            .                                                                 .                                              .
        #                                                            .                                                                         .                                      .
        #                                                            .                                                                                   .                            .
        # ((p_0(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_0) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_{k-1}) * sum_0  ...      ((p_n(x_0=C_{k-1} | x_t) / q(x_t | x_0=C_{k-1}) / sum_n) * a_cumulative_{t-1} + b_cumulative_{t-1}) * q(x_t | x_{t-1}=C_{k-1}) * sum_n
        # c_cumulative_{t-1} * q(x_t | x_{t-1}=C_k) * sum_0                                                                                       ...      c_cumulative_{t-1} * q(x_t | x_{t-1}=C_k) * sum_0
        log_p_x_t_min_1 = q + log_q_t_given_x_t_min_1 + q_log_sum_exp

        # For each column, there are two possible cases.
        #
        # Where:
        # - sum(p_n(x_0))) is summing over all classes for x_0
        # - C_i is the class transitioning from (not to be confused with c_t and c_cumulative_t being used for gamma's)
        # - C_j is the class transitioning to
        #
        # 1. x_t is masked i.e. x_t = c_k
        #
        # Simplifying the expression, the column vector is:
        #                                                      .
        #                                                      .
        #                                                      .
        # (c_t / c_cumulative_t) * (a_cumulative_{t-1} * p_n(x_0 = C_i | x_t) + b_cumulative_{t-1} * sum(p_n(x_0)))
        #                                                      .
        #                                                      .
        #                                                      .
        # (c_cumulative_{t-1} / c_cumulative_t) * sum(p_n(x_0))
        #
        # From equation (11) stated in terms of forward probabilities, the last row is trivially verified.
        #
        # For the other rows, we can state the equation as ...
        #
        # (c_t / c_cumulative_t) * [b_cumulative_{t-1} * p(x_0=c_0) + ... + (a_cumulative_{t-1} + b_cumulative_{t-1}) * p(x_0=C_i) + ... + b_cumulative_{k-1} * p(x_0=c_{k-1})]
        #
        # This verifies the other rows.
        #
        # 2. x_t is not masked
        #
        # Simplifying the expression, there are two cases for the rows of the column vector, where C_j = C_i and where C_j != C_i:
        #                                                      .
        #                                                      .
        #                                                      .
        # C_j != C_i:        b_t * ((b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_0) + ... + ((a_cumulative_{t-1} + b_cumulative_{t-1}) / b_cumulative_t) * p_n(x_0 = C_i) + ... + (b_cumulative_{t-1} / (a_cumulative_t + b_cumulative_t)) * p_n(c_0=C_j) + ... + (b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_{k-1}))
        #                                                      .
        #                                                      .
        #                                                      .
        # C_j = C_i: (a_t + b_t) * ((b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_0) + ... + ((a_cumulative_{t-1} + b_cumulative_{t-1}) / (a_cumulative_t + b_cumulative_t)) * p_n(x_0 = C_i = C_j) + ... + (b_cumulative_{t-1} / b_cumulative_t) * p_n(x_0 = c_{k-1}))
        #                                                      .
        #                                                      .
        #                                                      .
        # 0
        #
        # The last row is trivially verified. The other rows can be verified by directly expanding equation (11) stated in terms of forward probabilities.
        return log_p_x_t_min_1

    def log_Q_t_transitioning_to_known_class(
        self, *, t: torch.int, x_t: torch.LongTensor, log_onehot_x_t: torch.Tensor, cumulative: bool
    ):
        """
        Calculates the log probabilities of the rows from the (cumulative or non-cumulative) transition matrix for each
        latent pixel in `x_t`.

        Args:
            t (`torch.Long`):
                The timestep that determines which transition matrix is used.
            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.
            log_onehot_x_t (`torch.Tensor` of shape `(batch size, num classes, num latent pixels)`):
                The log one-hot vectors of `x_t`.
            cumulative (`bool`):
                If cumulative is `False`, the single step transition matrix `t-1`->`t` is used. If cumulative is
                `True`, the cumulative transition matrix `0`->`t` is used.

        Returns:
            `torch.Tensor` of shape `(batch size, num classes - 1, num latent pixels)`:
                Each _column_ of the returned matrix is a _row_ of log probabilities of the complete probability
                transition matrix.

                When non cumulative, returns `self.num_classes - 1` rows because the initial latent pixel cannot be
                masked.

                Where:
                - `q_n` is the probability distribution for the forward process of the `n`th latent pixel.
                - C_0 is a class of a latent pixel embedding
                - C_k is the class of the masked latent pixel

                non-cumulative result (omitting logarithms):
                ```
                q_0(x_t | x_{t-1} = C_0) ... q_n(x_t | x_{t-1} = C_0)
                          .      .                     .
                          .               .            .
                          .                      .     .
                q_0(x_t | x_{t-1} = C_k) ... q_n(x_t | x_{t-1} = C_k)
                ```

                cumulative result (omitting logarithms):
                ```
                q_0_cumulative(x_t | x_0 = C_0)    ...  q_n_cumulative(x_t | x_0 = C_0)
                          .               .                          .
                          .                        .                 .
                          .                               .          .
                q_0_cumulative(x_t | x_0 = C_{k-1}) ... q_n_cumulative(x_t | x_0 = C_{k-1})
                ```
        """
        if cumulative:
            a = self.log_cumprod_at[t]
            b = self.log_cumprod_bt[t]
            c = self.log_cumprod_ct[t]
        else:
            a = self.log_at[t]
            b = self.log_bt[t]
            c = self.log_ct[t]

        if not cumulative:
            # The values in the onehot vector can also be used as the logprobs for transitioning
            # from masked latent pixels. If we are not calculating the cumulative transitions,
            # we need to save these vectors to be re-appended to the final matrix so the values
            # aren't overwritten.
            #
            # `P(x_t!=mask|x_{t-1=mask}) = 0` and 0 will be the value of the last row of the onehot vector
            # if x_t is not masked
            #
            # `P(x_t=mask|x_{t-1=mask}) = 1` and 1 will be the value of the last row of the onehot vector
            # if x_t is masked
            log_onehot_x_t_transitioning_from_masked = log_onehot_x_t[:, -1, :].unsqueeze(1)

        # `index_to_log_onehot` will add onehot vectors for masked pixels,
        # so the default one hot matrix has one too many rows. See the doc string
        # for an explanation of the dimensionality of the returned matrix.
        log_onehot_x_t = log_onehot_x_t[:, :-1, :]

        # this is a cheeky trick to produce the transition probabilities using log one-hot vectors.
        #
        # Don't worry about what values this sets in the columns that mark transitions
        # to masked latent pixels. They are overwrote later with the `mask_class_mask`.
        #
        # Looking at the below logspace formula in non-logspace, each value will evaluate to either
        # `1 * a + b = a + b` where `log_Q_t` has the one hot value in the column
        # or
        # `0 * a + b = b` where `log_Q_t` has the 0 values in the column.
        #
        # See equation 7 for more details.
        log_Q_t = (log_onehot_x_t + a).logaddexp(b)

        # The whole column of each masked pixel is `c`
        mask_class_mask = x_t == self.mask_class
        mask_class_mask = mask_class_mask.unsqueeze(1).expand(-1, self.num_embed - 1, -1)
        log_Q_t[mask_class_mask] = c

        if not cumulative:
            log_Q_t = torch.cat((log_Q_t, log_onehot_x_t_transitioning_from_masked), dim=1)

        return log_Q_t

    def apply_cumulative_transitions(self, q, t):
        bsz = q.shape[0]
        a = self.log_cumprod_at[t]
        b = self.log_cumprod_bt[t]
        c = self.log_cumprod_ct[t]

        num_latent_pixels = q.shape[2]
        c = c.expand(bsz, 1, num_latent_pixels)

        q = (q + a).logaddexp(b)
        q = torch.cat((q, c), dim=1)

        return q
