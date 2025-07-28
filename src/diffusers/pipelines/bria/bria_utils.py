import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.optimization import get_scheduler
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_text(caption):
    existing_text_list = set()

    if caption[0] == '"' and caption[-1] == '"':
        caption = caption[1:-2]

    if caption[0] == "'" and caption[-1] == "'":
        caption = caption[1:-2]

    text_list = []
    current_text = ""
    text_present = False
    for c in caption:
        if c == '"' and not text_present:
            text_present = True
            continue

        if c == '"' and text_present:
            if current_text not in existing_text_list:
                text_list += [current_text]
                existing_text_list.add(current_text)

            text_present = False
            current_text = ""
            continue

        if text_present:
            current_text += c

    return text_list


def get_by_t5_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    max_sequence_length: int = 128,
    device: Optional[torch.device] = None,
):
    device = device or text_encoder.device

    if isinstance(prompt, list):
        assert len(prompt) == 1
        prompt = prompt[0]

    assert type(prompt) == str

    captions_list = get_text(prompt)
    embeddings_list = []
    for inner_prompt in captions_list:
        text_inputs = tokenizer(
            [inner_prompt],
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        embeddings_list += [prompt_embeds[0]]

    # No Text Found
    if len(embeddings_list) == 0:
        return None

    prompt_embeds = torch.concatenate(embeddings_list, axis=0)

    # Concat zeros to max_sequence
    seq_len, dim = prompt_embeds.shape
    if seq_len < max_sequence_length:
        padding = torch.zeros(
            (max_sequence_length - seq_len, dim), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )
        prompt_embeds = torch.concat([prompt_embeds, padding], dim=0)

    prompt_embeds = prompt_embeds.to(device=device)
    return prompt_embeds


def get_t5_prompt_embeds(
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 128,
    device: Optional[torch.device] = None,
):
    device = device or text_encoder.device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        # padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    # Concat zeros to max_sequence
    b, seq_len, dim = prompt_embeds.shape
    if seq_len < max_sequence_length:
        padding = torch.zeros(
            (b, max_sequence_length - seq_len, dim), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )
        prompt_embeds = torch.concat([prompt_embeds, padding], dim=1)

    prompt_embeds = prompt_embeds.to(device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# in order the get the same sigmas as in training and sample from them
def get_original_sigmas(num_train_timesteps=1000, num_inference_steps=1000):
    timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    sigmas = timesteps / num_train_timesteps

    inds = [int(ind) for ind in np.linspace(0, num_train_timesteps - 1, num_inference_steps)]
    new_sigmas = sigmas[inds]
    return new_sigmas


def is_ng_none(negative_prompt):
    return (
        negative_prompt is None
        or negative_prompt == ""
        or (isinstance(negative_prompt, list) and negative_prompt[0] is None)
        or (type(negative_prompt) == list and negative_prompt[0] == "")
    )


class CudaTimerContext:
    def __init__(self, times_arr):
        self.times_arr = times_arr

    def __enter__(self):
        self.before_event = torch.cuda.Event(enable_timing=True)
        self.after_event = torch.cuda.Event(enable_timing=True)
        self.before_event.record()

    def __exit__(self, type, value, traceback):
        self.after_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.before_event.elapsed_time(self.after_event) / 1000
        self.times_arr.append(elapsed_time)


def get_env_prefix():
    env = os.environ.get("CLOUD_PROVIDER", "AWS").upper()
    if env == "AWS":
        return "SM_CHANNEL"
    elif env == "AZURE":
        return "AZUREML_DATAREFERENCE"

    raise Exception(f"Env {env} not supported")


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def initialize_distributed():
    # Initialize the process group for distributed training
    dist.init_process_group("nccl")

    # Get the current process's rank (ID) and the total number of processes (world size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Initialized distributed training: Rank {rank}/{world_size}")


def get_clip_prompt_embeds(
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 77,
    device: Optional[torch.device] = None,
):
    device = device or text_encoder.device
    assert max_sequence_length == tokenizer.model_max_length
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2]

    # textual inversion: process multi-vector tokens if necessary
    prompt_embeds_list = []
    prompts = [prompt, prompt]
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]

        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return prompt_embeds, pooled_prompt_embeds


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


class FluxPosEmbed(torch.nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        freqs_dtype = torch.float32 if is_mps else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


# Not really cosine but with decay
def get_cosine_schedule_with_warmup_and_decay(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    constant_steps=-1,
    eps=1e-5,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        constant_steps (`int`):
            The total number of constant lr steps following a warmup

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if constant_steps <= 0:
        constant_steps = num_training_steps - num_warmup_steps

    def lr_lambda(current_step):
        # Accelerate sends current_step*num_processes
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + constant_steps:
            return 1

        # print(f'Inside LR: num_training_steps:{num_training_steps}, current_step:{current_step}, num_warmup_steps: {num_warmup_steps}, constant_steps: {constant_steps}')
        return max(
            eps,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps - constant_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(name, optimizer, num_warmup_steps, num_training_steps, constant_steps):
    if name != "constant_with_warmup_cosine_decay":
        return get_scheduler(
            name=name, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    # Usign custom warmup+cnstant+decay scheduler
    return get_cosine_schedule_with_warmup_and_decay(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        constant_steps=constant_steps,
    )
