import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import SchedulerMixin


def gumbel_noise(t: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Generate Gumbel noise for sampling.

    Args:
        t (`torch.Tensor`):
            Input tensor to match the shape and dtype of the output noise.
        generator (`torch.Generator`, *optional*):
            A random number generator for reproducible sampling.

    Returns:
        `torch.Tensor`:
            Gumbel-distributed noise with the same shape, dtype, and device as the input tensor.
    """
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(
    mask_len: torch.Tensor,
    probs: torch.Tensor,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Mask tokens by selecting the top-k lowest confidence scores with temperature-based randomness.

    Args:
        mask_len (`torch.Tensor`):
            Number of tokens to mask per sample in the batch.
        probs (`torch.Tensor`):
            Probability scores for each token.
        temperature (`float`, *optional*, defaults to 1.0):
            Temperature parameter for controlling randomness in the masking process.
        generator (`torch.Generator`, *optional*):
            A random number generator for reproducible sampling.

    Returns:
        `torch.Tensor`:
            Boolean mask indicating which tokens should be masked.
    """
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


@dataclass
class AmusedSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.LongTensor` of shape `(batch_size, height, width)` or `(batch_size, sequence_length)`):
            Computed sample `(x_{t-1})` of previous timestep with token IDs. `prev_sample` should be used as next model
            input in the denoising loop.
        pred_original_sample (`torch.LongTensor` of shape `(batch_size, height, width)` or `(batch_size, sequence_length)`, *optional*):
            The predicted fully denoised sample `(x_{0})` with token IDs based on the model output from the current
            timestep. `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class AmusedScheduler(SchedulerMixin, ConfigMixin):
    """
    A scheduler for masked token generation as used in [`AmusedPipeline`].

    This scheduler iteratively unmasks tokens based on their confidence scores, following either a cosine or linear
    schedule. Unlike traditional diffusion schedulers that work with continuous pixel values, this scheduler operates
    on discrete token IDs, making it suitable for autoregressive and non-autoregressive masked token generation models.

    This scheduler inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the
    generic methods the library implements for all schedulers such as loading and saving.

    Args:
        mask_token_id (`int`):
            The token ID used to represent masked tokens in the sequence.
        masking_schedule (`Literal["cosine", "linear"]`, *optional*, defaults to `"cosine"`):
            The schedule type for determining the mask ratio at each timestep. Can be either `"cosine"` or `"linear"`.
    """

    order = 1

    temperatures: Optional[torch.Tensor]
    timesteps: Optional[torch.Tensor]

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: Literal["cosine", "linear"] = "cosine",
    ):
        self.temperatures = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[float, Tuple[float, float], List[float]] = (2, 0),
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Set the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            temperature (`Union[float, Tuple[float, float], List[float]]`, *optional*, defaults to `(2, 0)`):
                Temperature parameter(s) for controlling the randomness of sampling. If a tuple or list is provided,
                temperatures will be linearly interpolated between the first and second values across all timesteps. If
                a single value is provided, temperatures will be linearly interpolated from that value to 0.01.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps and temperatures should be moved to. If `None`, the timesteps are not
                moved.
        """
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
        else:
            self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.LongTensor,
        starting_mask_ratio: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[AmusedSchedulerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the sample at the previous timestep by masking tokens based on confidence scores.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model. Typically of shape `(batch_size, num_tokens,
                codebook_size)` or `(batch_size, codebook_size, height, width)` for 2D inputs.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.LongTensor`):
                A current instance of a sample created by the diffusion process. Contains token IDs, with masked
                positions indicated by `mask_token_id`.
            starting_mask_ratio (`float`, *optional*, defaults to 1.0):
                A multiplier applied to the mask ratio schedule. Values less than 1.0 will result in fewer tokens being
                masked at each step.
            generator (`torch.Generator`, *optional*):
                A random number generator for reproducible sampling.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an [`~schedulers.scheduling_amused.AmusedSchedulerOutput`] or a plain tuple.

        Returns:
            [`~schedulers.scheduling_amused.AmusedSchedulerOutput`] or `tuple`:
                If `return_dict` is `True`, [`~schedulers.scheduling_amused.AmusedSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor (`prev_sample`) and the
                second element is the predicted original sample tensor (`pred_original_sample`).
        """
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4

        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)

        unknown_map = sample == self.config.mask_token_id

        probs = model_output.softmax(dim=-1)

        device = probs.device
        probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
        probs_ = probs_.reshape(-1, probs.size(-1))
        pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)
        pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
        pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()
            # do not mask more than amount previously masked
            mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            # mask at least one
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)

            selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

            masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)

            # Masks tokens with lower confidence.
            prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return AmusedSchedulerOutput(prev_sample, pred_original_sample)

    def add_noise(
        self,
        sample: torch.LongTensor,
        timesteps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """
        Add noise to a sample by randomly masking tokens according to the masking schedule.

        Args:
            sample (`torch.LongTensor`):
                The input sample containing token IDs to be partially masked.
            timesteps (`int`):
                The timestep that determines how much masking to apply. Higher timesteps result in more masking.
            generator (`torch.Generator`, *optional*):
                A random number generator for reproducible masking.

        Returns:
            `torch.LongTensor`:
                The sample with some tokens replaced by `mask_token_id` according to the masking schedule.
        """
        step_idx = (self.timesteps == timesteps).nonzero()
        ratio = (step_idx + 1) / len(self.timesteps)

        if self.config.masking_schedule == "cosine":
            mask_ratio = torch.cos(ratio * math.pi / 2)
        elif self.config.masking_schedule == "linear":
            mask_ratio = 1 - ratio
        else:
            raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

        mask_indices = (
            torch.rand(
                sample.shape, device=generator.device if generator is not None else sample.device, generator=generator
            ).to(sample.device)
            < mask_ratio
        )

        masked_sample = sample.clone()

        masked_sample[mask_indices] = self.config.mask_token_id

        return masked_sample
