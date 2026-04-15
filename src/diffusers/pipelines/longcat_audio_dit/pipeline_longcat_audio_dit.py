# Copyright 2026 MeiTuan LongCat-AudioDiT Team and The HuggingFace Team. All rights reserved.
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

# Adapted from the LongCat-AudioDiT reference implementation:
# https://github.com/meituan-longcat/LongCat-AudioDiT

import re
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, UMT5EncoderModel

from ...models import LongCatAudioDiTTransformer, LongCatAudioDiTVae
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline


logger = logging.get_logger(__name__)


def _lens_to_mask(lengths: torch.Tensor, length: int | None = None) -> torch.BoolTensor:
    if length is None:
        length = int(lengths.amax().item())
    seq = torch.arange(length, device=lengths.device)
    return seq[None, :] < lengths[:, None]


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["“”‘’]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _approx_duration_from_text(text: str | list[str], max_duration: float = 30.0) -> float:
    if not text:
        return 0.0
    if isinstance(text, str):
        text = [text]

    en_dur_per_char = 0.082
    zh_dur_per_char = 0.21
    durations = []
    for prompt in text:
        prompt = re.sub(r"\s+", "", prompt)
        num_zh = num_en = num_other = 0
        for char in prompt:
            if "一" <= char <= "鿿":
                num_zh += 1
            elif char.isalpha():
                num_en += 1
            else:
                num_other += 1
        if num_zh > num_en:
            num_zh += num_other
        else:
            num_en += num_other
        durations.append(num_zh * zh_dur_per_char + num_en * en_dur_per_char)
    return min(max_duration, max(durations)) if durations else 0.0


class LongCatAudioDiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae: LongCatAudioDiTVae,
        text_encoder: UMT5EncoderModel,
        tokenizer: PreTrainedTokenizerBase,
        transformer: LongCatAudioDiTTransformer,
        scheduler: FlowMatchEulerDiscreteScheduler | None = None,
    ):
        super().__init__()
        if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, invert_sigmas=True)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.sample_rate = getattr(vae.config, "sample_rate", 24000)
        self.vae_scale_factor = getattr(vae.config, "downsampling_ratio", 2048)
        self.latent_dim = getattr(transformer.config, "latent_dim", 64)
        self.max_wav_duration = 30.0
        self.text_norm_feat = True
        self.text_add_embed = True

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def encode_prompt(self, prompt: str | list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        model_max_length = getattr(self.tokenizer, "model_max_length", 512)
        if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 32768:
            model_max_length = 512
        text_inputs = self.tokenizer(
            prompt,
            padding="longest",
            truncation=True,
            max_length=model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        with torch.no_grad():
            output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        prompt_embeds = output.last_hidden_state
        if self.text_norm_feat:
            prompt_embeds = F.layer_norm(prompt_embeds, (prompt_embeds.shape[-1],), eps=1e-6)
        if self.text_add_embed and getattr(output, "hidden_states", None):
            first_hidden = output.hidden_states[0]
            if self.text_norm_feat:
                first_hidden = F.layer_norm(first_hidden, (first_hidden.shape[-1],), eps=1e-6)
            prompt_embeds = prompt_embeds + first_hidden
        lengths = attention_mask.sum(dim=1).to(device)
        return prompt_embeds, lengths

    def prepare_latents(
        self,
        batch_size: int,
        duration: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim != 3:
                raise ValueError(
                    f"`latents` must have shape (batch_size, duration, latent_dim), but got {tuple(latents.shape)}."
                )
            if latents.shape[0] != batch_size:
                raise ValueError(f"`latents` must have batch size {batch_size}, but got {latents.shape[0]}.")
            if latents.shape[2] != self.latent_dim:
                raise ValueError(f"`latents` must have latent_dim {self.latent_dim}, but got {latents.shape[2]}.")
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Expected {batch_size} generators for batch size {batch_size}, but got {len(generator)}."
            )

        return randn_tensor((batch_size, duration, self.latent_dim), generator=generator, device=device, dtype=dtype)

    def check_inputs(
        self,
        prompt: list[str],
        negative_prompt: str | list[str] | None,
        output_type: str,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> None:
        if len(prompt) == 0:
            raise ValueError("`prompt` must contain at least one prompt.")

        if output_type not in {"np", "pt", "latent"}:
            raise ValueError(f"Unsupported output_type: {output_type}")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if negative_prompt is not None and not isinstance(negative_prompt, str):
            negative_prompt = list(negative_prompt)
            if len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt` must have batch size {len(prompt)}, but got {len(negative_prompt)} prompts."
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        audio_duration_s: float | None = None,
        latents: torch.Tensor | None = None,
        num_inference_steps: int = 16,
        guidance_scale: float = 4.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`): Prompt or prompts that guide audio generation.
            negative_prompt (`str` or `list[str]`, *optional*): Negative prompt(s) for classifier-free guidance.
            audio_duration_s (`float`, *optional*):
                Target audio duration in seconds. Ignored when `latents` is provided.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents of shape `(batch_size, duration, latent_dim)`.
            num_inference_steps (`int`, defaults to 16): Number of denoising steps.
            guidance_scale (`float`, defaults to 4.0): Guidance scale for classifier-free guidance.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*): Random generator(s).
            output_type (`str`, defaults to `"np"`): Output format: `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`): Whether to return `AudioPipelineOutput`.
            callback_on_step_end (`Callable`, *optional*):
                A function called at the end of each denoising step with the pipeline, step index, timestep, and tensor
                inputs specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, defaults to `["latents"]`):
                Tensor inputs passed to `callback_on_step_end`.
        """
        if prompt is None:
            prompt = []
        elif isinstance(prompt, str):
            prompt = [prompt]
        else:
            prompt = list(prompt)
        self.check_inputs(prompt, negative_prompt, output_type, callback_on_step_end_tensor_inputs)
        batch_size = len(prompt)
        self._guidance_scale = guidance_scale

        device = self._execution_device
        normalized_prompts = [_normalize_text(text) for text in prompt]
        if latents is not None:
            duration = latents.shape[1]
        elif audio_duration_s is not None:
            duration = int(audio_duration_s * self.sample_rate // self.vae_scale_factor)
        else:
            duration = int(_approx_duration_from_text(normalized_prompts) * self.sample_rate // self.vae_scale_factor)
        max_duration = int(self.max_wav_duration * self.sample_rate // self.vae_scale_factor)
        if latents is None:
            duration = max(1, min(duration, max_duration))

        prompt_embeds, prompt_embeds_len = self.encode_prompt(normalized_prompts, device)
        duration_tensor = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        mask = _lens_to_mask(duration_tensor)
        text_mask = _lens_to_mask(prompt_embeds_len, length=prompt_embeds.shape[1])

        if negative_prompt is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_prompt_embeds_len = prompt_embeds_len
            negative_prompt_embeds_mask = text_mask
        else:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            else:
                negative_prompt = list(negative_prompt)
            negative_prompt_embeds, negative_prompt_embeds_len = self.encode_prompt(negative_prompt, device)
            negative_prompt_embeds_mask = _lens_to_mask(
                negative_prompt_embeds_len, length=negative_prompt_embeds.shape[1]
            )

        latent_cond = torch.zeros(batch_size, duration, self.latent_dim, device=device, dtype=prompt_embeds.dtype)
        latents = self.prepare_latents(
            batch_size, duration, device, prompt_embeds.dtype, generator=generator, latents=latents
        )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer.")

        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                curr_t = (
                    (t / self.scheduler.config.num_train_timesteps).expand(batch_size).to(dtype=prompt_embeds.dtype)
                )
                pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=text_mask,
                    timestep=curr_t,
                    attention_mask=mask,
                    latent_cond=latent_cond,
                ).sample
                if self.guidance_scale > 1.0:
                    null_pred = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_embeds_mask,
                        timestep=curr_t,
                        attention_mask=mask,
                        latent_cond=latent_cond,
                    ).sample
                    pred = null_pred + (pred - null_pred) * self.guidance_scale
                latents = self.scheduler.step(pred, t, latents, return_dict=False)[0]
                progress_bar.update()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        if output_type == "latent":
            waveform = latents
        else:
            waveform = self.vae.decode(latents.permute(0, 2, 1)).sample
            if output_type == "np":
                waveform = waveform.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (waveform,)
        return AudioPipelineOutput(audios=waveform)
