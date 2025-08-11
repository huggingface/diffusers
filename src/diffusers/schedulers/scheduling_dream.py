from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def create_schedule(
    schedule_params: Optional[Union[float, Tuple[float, float], List[float], torch.Tensor]],
    num_inference_steps: int,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if schedule_params is None:
        schedule = None
    elif isinstance(schedule_params, float):
        # Interpret as a constant schedule for all timesteps
        schedule = torch.full(num_inference_steps, schedule_params)
    elif isinstance(schedule_params, (tuple, list)):
        # Interpret first and second elems as start and end points of a linear schedule
        schedule = torch.linspace(schedule_params[0], schedule_params[1], num_inference_steps)
    elif isinstance(schedule_params, torch.Tensor):
        # Interpret this as the fully specified schedule
        if schedule_params.ndim != 1:
            raise ValueError(f"Expected torch tensor schedule to have 1 dim but has {schedule_params.ndim} dims")
        if schedule_params.shape[0] != num_inference_steps:
            raise ValueError(
                f"Receive torch tensor schedule but length ({schedule_params}) does not match num_inference_steps "
                f"({num_inference_steps})"
            )
        schedule = schedule_params
    else:
        raise ValueError(
            f"`schedule_params` is of unrecognized type {type(schedule_params)}; should be either a float, tuple, "
            f"list, or `torch.Tensor`."
        )

    if schedule is not None:
        schedule = schedule.to(device=device)
    return schedule


def top_p_logits(logits: torch.Tensor, top_p: Optional[float] = None) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    generator: Optional[torch.Generator] = None,
 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples from a sequence of logits of shape [..., vocab_size] and returns both the sampled sequence (as the second
    return elem) and the model probabilities for the chosen tokens (as the first return elem).
    """
    # logits shape: [B, L, V]
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)
    device = probs.device
    probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
    if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
        probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
    if probs.ndim > 2:
        probs_ = probs_.reshape(-1, probs.size(-1))  # [B, L, V] --> [B * L, V]

    if temperature > 0:
        try:
            # Sample x0 ~ Cat(probs)
            x0 = torch.multinomial(probs_, 1, generator=generator).to(device=device)
            if probs.ndim > 2:
                x0 = x0[:, 0].view(*probs.shape[:-1])  # [B * L, 1] --> [B, L]
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamMaskedDiffusionSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, seq_len)` for text):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, seg_len)` for text):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None



class DreamMaskedDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for the Dream 7B masked diffusion model.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        masking_schedule (`str`, defaults to `"linear"`):
            The noise schedule for discrete diffusion, often represented as alpha_t. This determines the probability
            that tokens are masked in the forward process. Available choices are `"linear"`, `"cosine"`, and
            `"polynomial"`.
        timestep_discretization (`str`, defaults to `"linear"`):
            The function which specifies how we discretize (continuous) time [0, 1]. Available strategies are
            `"linear"` (evenly spaced timesteps) and `"cosine"`.
        logit_sampling_alg (`str`, defaults to `"entropy"`):
            The algorithm used to sample from the predicted logits. This incorporates sampling techniques such as
            temperature, top-p, top-k, etc. Available algorithms are `"origin"`, `"maskgit_plus"`, `"topk_margin"`,
            and `"entropy"` (names match those of original code).
        shift (`bool`, defaults to `True`):
            Whether to shift the logits before sampling. Dream models shift the logits such that the (n - 1)-th token
            predicts the n-th token, mirroring the behavior of AR models.
        polynomial_exp (`int`, defaults to `1`):
            When `masking_schedule` is set to `"polynomial"`, this specifies the exponent of the polynomial. The
            default value of `1` is equivalent to a `"linear"` masking schedule.
        final_timestep (`float`, defaults to `1e-3`):
            The value of the final timestep in the schedule, which should be a small positive number close to 0 for
            numerical stability reasons.
        temperature: (`float` or `tuple` or `list`, defaults to `0.2`):
            The temperature used when taking the softmax of the predicted logits. If this is a float, we will use that
            value at each timestep; if a tuple or list, we will interpret the first and second elements as the start
            and end points of a linear schedule. If `None`, a temperature of 1.0 will be used.
        top_p: (`float` or `tuple` or `list`, *optional*, defaults to `0.96`):
            The probability for top-p sampling. If this is a float, we will use that value at each timestep; if a
            tuple or list, we will interpret the first and second elements as the start and end points of a linear
            schedule. If `None`, top-p sampling will not be performed.
        top_k: (`int`, *optional*, defaults to `None`):
            The k value for top-p sampling. If not set, top-k sampling will not be performed.
        alg_temperature: (`float`, *optional*, defaults to `0.0`):
            Used for certain logit sampling strategies, such as `"maskgit_plus"`, `"topk_margin"`, and `"entropy"`. If
            > 0, we will use this as a temperature when taking the softmax over the model confidences to decide which
            tokens to unmask. Otherwise, we will deterministically select the tokens with the highest confidences.
        mask_token_id (`int`, defaults to `151666`):
            The token id of the mask token in the tokenizer. The default value corresponds to the mask token id in the
            official Dream 7B tokenizer.
        start_token_id (`int`, defaults to `151643`):
            The token id of the start/BOS token in the tokenizer. The default value corresponds to the BOS token id
            in the official Dream 7B tokenizer.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        masking_schedule: str = "linear",
        timestep_discretization: str = "linear",
        logit_sampling_alg: str = "entropy",
        shift: bool = True,
        polynomial_exp: int = 1,
        final_timestep: float = 1e-3,  # small positive value for final timestep (eps in original code)
        temperature: Optional[Union[float, Tuple[float], List[float]]] = 0.2,
        top_p: Optional[Union[float, Tuple[float], List[float]]] = 0.95,
        top_k: Optional[int] = None,
        alg_temperature: Optional[float] = 0.0,
        mask_token_id: int = 151666,
        start_token_id: int = 151643,
    ):
        # Setable values
        self.num_inference_steps = None
        self.temperatures = None
        self.top_p_schedule = None
        self.top_k_schedule = None

        self.timesteps = None
        self.alphas = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Optional[Union[float, Tuple[float, float], List[int], torch.Tensor]] = None,
        top_p: Optional[Union[float, Tuple[float, float], List[int], torch.Tensor]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            temperature (`float` or `tuple` or `list` or `torch.Tensor`, *optional*, defaults to `None`):
                A custom temperature schedule to override the configured temperature schedule. If this is a float, we
                will use that float at each timestep; if a tuple or list, we will interpret the first and second
                elements as the start and end points of a linear schedule; if a `torch.Tensor`, we will interpret this
                as the full temperature schedule (must have length `num_inference_steps`).
            top-p (`float` or `tuple` or `list` or `torch.Tensor`, *optional*, defaults to `None`):
                A custom top-p schedule to override the configured top-p schedule. If this is a float, we will use
                that value at each timestep; if a tuple or list, we will interpret the first and second elements as
                the start and end points of a linear schedule; if a `torch.Tensor`, we will interpret this as the full
                top-p schedule (must have length `num_inference_steps`).
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    
        if self.config.timestep_discretization == "linear":
            timesteps = torch.linspace(1.0, self.config.final_timestep, num_inference_steps + 1, device=device)
        elif self.config.timestep_discretization == "cosine":
            timesteps = torch.linspace(self.config.final_timestep, 1.0, num_inference_steps + 1)
            timesteps = torch.cos((torch.pi / 2) * (1.0 - timesteps)).to(device)
        else:
            raise ValueError(
                f"{self.config.timestep_discretization} is not a supported timestep discretization strategy. Current "
                f"supported strategies are `linear` and `cosine`."
            )
        self.timesteps = timesteps

        # Now calculate the masking or noise schedule (alpha) values at the chosen timestep discretization
        if self.config.masking_schedule == "linear":
            alphas = 1.0 - self.timesteps
        elif self.config.masking_schedule == "cosine":
            alphas = 1.0 - torch.cos((torch.pi / 2) * (1.0 - self.timesteps))
        elif self.config.masking_schedule == "polynomial":
            alphas = 1.0 - torch.pow(self.timesteps, self.config.polynomial_exp)
        else:
            raise ValueError(
                f"{self.config.masking_schedule} is not a supported masking schedule. Currently supported schedules "
                f"are `linear`, `cosine`, and `polynomial`."
            )
        self.alphas = alphas.to(device=device)

        # Allow overriding of specific sampling parameters (temperature, top_p, etc.)
        if temperature is None:
            temperature = self.config.temperature
        self.temperatures = create_schedule(temperature, num_inference_steps)

        if top_p is None:
            top_p = self.config.top_p
        self.top_p_schedule = create_schedule(top_p, num_inference_steps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DreamMaskedDiffusionSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reverse process. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_dream.DreamMaskedDiffusionSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_dream.DreamMaskedDiffusionSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        # model_output shape: [B, L, V]
        # sample shape: [B, L] (sequence of discrete tokens)
        step_idx = self.index_for_timestep(timestep)
        t = self.timesteps[step_idx]  # Current timestep
        s = self.timesteps[step_idx + 1]  # Previous timestep (next-largest timestep not yet processed)
        temperature = self.temperatures[step_idx] if self.temperatures is not None else 1.0
        top_p = self.top_p_schedule[step_idx] if self.top_p_schedule is not None else None
        top_k = self.top_k_schedule[step_idx] if self.top_k_schedule is not None else None

        mask_map = sample == self.config.mask_token_id

        if self.config.shift:
            # Right shift the logits from the model
            model_output = torch.cat(model_output[:, :1], model_output[:, :-1], dim=1)

        # Probability of unmasking each token at time t
        unmask_prob = (self.alphas[step_idx + 1] - self.alphas[step_idx]) / (1 - self.alphas[step_idx])
        # Unmask all remaining masked tokens at last inference step
        unmask_prob = unmask_prob if step_idx < self.num_inference_steps - 1 else 1.0

        # TODO: mask logits (model_output) beforehand? might make it more efficient?
        if self.config.logit_sampling_alg == "origin":
            to_unmask_mask = torch.rand(*sample.shape, generator=generator, device=sample.device) < unmask_prob
            confidence, pred_original_sample = sample_tokens(
                model_output, temperature=temperature, top_p=top_p, top_k=top_k, generator=generator
            )
            prev_sample = torch.where(to_unmask_mask, pred_original_sample, sample)
        else:
            if self.config.logit_sampling_alg == "maskgit_plus":
                confidence, pred_original_sample = sample_tokens(
                    model_output, temperature=temperature, top_p=top_p, top_k=top_k, generator=generator
                )
            elif self.config.logit_sampling_alg == "topk_margin":
                confidence, pred_original_sample = sample_tokens(
                    model_output,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                    generator=generator,
                )
            elif self.config.logit_sampling_alg == "entropy":
                confidence, pred_original_sample = sample_tokens(
                    model_output,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                    generator=generator,
                )
            
            # Unmask a fixed number of tokens at each timestep depending on unmask_prob
            num_masked_tokens = mask_map.sum() / mask_map.shape[0]
            num_tokens_to_unmask = int(num_masked_tokens * unmask_prob)
            full_confidence = torch.full_like(sample, -torch.inf, dtype=model_output.dtype, device=sample.device)
            full_confidence = torch.where(mask_map, confidence, full_confidence)

            if num_tokens_to_unmask > 0:
                if self.config.alg_temperature is None or self.config.alg_temperature == 0:
                    _, unmask_index = torch.topk(full_confidence, num_tokens_to_unmask)
                else:
                    full_confidence = full_confidence / self.config.alg_temperature
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    unmask_index = torch.multinomial(full_confidence, num_samples=num_tokens_to_unmask)
                
                prev_sample = torch.zeros_like(sample, device=sample.device)
                prev_sample = torch.where(unmask_index, pred_original_sample, sample)

        # TODO: do we need to shift the tokens again at the end???
        if not return_dict:
            return (prev_sample, pred_original_sample)

        return DreamMaskedDiffusionSchedulerOutput(prev_sample, pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        timesteps: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # For each batch instance i with timestep t_i, mask each position independently with prob 1 - alphas[t_i]
        # original_samples shape: [B, L]
        # Make sure alphas and timesteps have the same device and dtype as original_samples
        alphas = self.alphas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)
    
        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

        mask_probs = 1.0 - alphas[step_indices].flatten()
        while len(mask_probs).shape < len(original_samples.shape):
            mask_probs.unsqueeze(-1)

        mask_indices = (
            torch.rand(
                original_samples.shape,
                device=generator.device if generator is not None else original_samples.device,
                generator=generator,
            ).to(original_samples.device)
            < mask_probs
        )

        masked_samples = original_samples.clone()
        masked_samples[mask_indices] = self.config.mask_token_id

        return masked_samples
