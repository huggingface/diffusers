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

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import validate_hf_hub_args
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizerBase, UMT5Config, UMT5EncoderModel

from ...models import LongCatAudioDiTTransformer, LongCatAudioDiTVae
from ...utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT, logging
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


def _approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    en_dur_per_char = 0.082
    zh_dur_per_char = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for char in text:
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
    return min(max_duration, num_zh * zh_dur_per_char + num_en * en_dur_per_char)


def _approx_batch_duration_from_prompts(prompts: list[str]) -> float:
    if not prompts:
        return 0.0
    return max(_approx_duration_from_text(prompt) for prompt in prompts)


def _extract_prefixed_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    prefix = f"{prefix}."
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _load_longcat_tokenizer(
    pretrained_model_name_or_path: str | Path,
    text_encoder_model: str | None,
    tokenizer: PreTrainedTokenizerBase | str | Path | None,
    local_files_only: bool | None,
    subfolder: str | None = None,
) -> PreTrainedTokenizerBase:
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        return tokenizer

    tokenizer_source: str | Path | None = tokenizer
    if tokenizer_source is None:
        pretrained_path = Path(pretrained_model_name_or_path)
        local_tokenizer_dir = pretrained_path / (subfolder or "") / "tokenizer"
        if pretrained_path.exists() and local_tokenizer_dir.is_dir():
            tokenizer_source = local_tokenizer_dir
        else:
            tokenizer_source = text_encoder_model or pretrained_model_name_or_path

    if tokenizer_source is None:
        raise ValueError("Could not determine tokenizer source for LongCatAudioDiT.")

    tokenizer_kwargs = {"local_files_only": local_files_only}
    if not isinstance(tokenizer_source, Path) and tokenizer_source == pretrained_model_name_or_path and subfolder:
        tokenizer_kwargs["subfolder"] = subfolder
    return AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)


def _resolve_longcat_file(
    pretrained_model_name_or_path: str | Path,
    filename: str,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    local_files_only: bool | None = None,
    token: str | bool | None = None,
    revision: str | None = None,
    subfolder: str | None = None,
    local_dir: str | Path | None = None,
    local_dir_use_symlinks: str | bool = "auto",
    user_agent: dict[str, str] | None = None,
) -> str:
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if Path(pretrained_model_name_or_path).is_dir():
        base = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            base = base / subfolder
        file_path = base / filename
        if not file_path.is_file():
            raise EnvironmentError(f"Error no file named {filename} found in directory {base}.")
        return str(file_path)

    try:
        return hf_hub_download(
            pretrained_model_name_or_path,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            subfolder=subfolder,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            user_agent=user_agent,
        )
    except Exception as err:
        raise EnvironmentError(
            f"Can't load {filename} for '{pretrained_model_name_or_path}'. If you were trying to load it from "
            f"'{HUGGINGFACE_CO_RESOLVE_ENDPOINT}', make sure the repo exists or that your local path is correct."
        ) from err


class LongCatAudioDiTPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        vae: LongCatAudioDiTVae,
        text_encoder: UMT5EncoderModel,
        tokenizer: PreTrainedTokenizerBase,
        transformer: LongCatAudioDiTTransformer,
    ):
        super().__init__()
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer)
        self.sample_rate = getattr(vae.config, "sample_rate", 24000)
        self.latent_hop = getattr(vae.config, "downsampling_ratio", 2048)
        self.latent_dim = getattr(transformer.config, "latent_dim", 64)
        self.max_wav_duration = 30.0
        self.text_norm_feat = True
        self.text_add_embed = True

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        tokenizer: PreTrainedTokenizerBase | str | Path | None = None,
        torch_dtype: torch.dtype | None = None,
        local_files_only: bool | None = None,
        **kwargs: Any,
    ) -> "LongCatAudioDiTPipeline":
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        try:
            cls.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
            )
        except (EnvironmentError, OSError, ValueError):
            pass
        else:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                tokenizer=tokenizer,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                force_download=force_download,
                proxies=proxies,
                token=token,
                revision=revision,
                subfolder=subfolder,
                **kwargs,
            )

        if kwargs:
            logger.warning("Ignoring unsupported LongCatAudioDiTPipeline.from_pretrained kwargs: %s", sorted(kwargs))

        config_path = _resolve_longcat_file(
            pretrained_model_name_or_path,
            "config.json",
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            user_agent={"file_type": "config"},
        )
        weights_path = _resolve_longcat_file(
            pretrained_model_name_or_path,
            "model.safetensors",
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            user_agent={"file_type": "weights"},
        )

        with open(config_path) as handle:
            config = json.load(handle)

        text_encoder_config = UMT5Config.from_dict(config["text_encoder_config"])
        text_encoder = UMT5EncoderModel(text_encoder_config)
        transformer = LongCatAudioDiTTransformer(
            dit_dim=config["dit_dim"],
            dit_depth=config["dit_depth"],
            dit_heads=config["dit_heads"],
            dit_text_dim=config["dit_text_dim"],
            latent_dim=config["latent_dim"],
            dropout=config.get("dit_dropout", 0.0),
            bias=config.get("dit_bias", True),
            cross_attn=config.get("dit_cross_attn", True),
            adaln_type=config.get("dit_adaln_type", "global"),
            adaln_use_text_cond=config.get("dit_adaln_use_text_cond", True),
            long_skip=config.get("dit_long_skip", True),
            text_conv=config.get("dit_text_conv", True),
            qk_norm=config.get("dit_qk_norm", True),
            cross_attn_norm=config.get("dit_cross_attn_norm", False),
            eps=config.get("dit_eps", 1e-6),
            use_latent_condition=config.get("dit_use_latent_condition", True),
        )
        vae_config = dict(config["vae_config"])
        vae_config.pop("model_type", None)
        vae = LongCatAudioDiTVae(**vae_config)

        state_dict = load_file(weights_path)
        transformer.load_state_dict(_extract_prefixed_state_dict(state_dict, "transformer"), strict=True)
        vae.load_state_dict(_extract_prefixed_state_dict(state_dict, "vae"), strict=True)
        text_missing, text_unexpected = text_encoder.load_state_dict(
            _extract_prefixed_state_dict(state_dict, "text_encoder"), strict=False
        )
        allowed_missing = {"shared.weight"}
        unexpected_missing = set(text_missing) - allowed_missing
        if unexpected_missing:
            raise RuntimeError(f"Unexpected missing LongCatAudioDiT text encoder weights: {sorted(unexpected_missing)}")
        if text_unexpected:
            raise RuntimeError(f"Unexpected LongCatAudioDiT text encoder weights: {sorted(text_unexpected)}")
        if "shared.weight" in text_missing:
            text_encoder.shared.weight.data.copy_(text_encoder.encoder.embed_tokens.weight.data)

        tokenizer = _load_longcat_tokenizer(
            pretrained_model_name_or_path,
            config.get("text_encoder_model"),
            tokenizer,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )

        if torch_dtype is not None:
            text_encoder = text_encoder.to(dtype=torch_dtype)
            transformer = transformer.to(dtype=torch_dtype)
            vae = vae.to(dtype=torch_dtype)

        text_encoder.eval()
        transformer.eval()
        vae.eval()

        pipe = cls(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer)
        pipe.sample_rate = config.get("sampling_rate", pipe.sample_rate)
        pipe.latent_hop = config.get("latent_hop", pipe.latent_hop)
        pipe.max_wav_duration = config.get("max_wav_duration", pipe.max_wav_duration)
        pipe.text_norm_feat = config.get("text_norm_feat", pipe.text_norm_feat)
        pipe.text_add_embed = config.get("text_add_embed", pipe.text_add_embed)
        return pipe

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
        return prompt_embeds.float(), lengths

    def prepare_latents(
        self,
        batch_size: int,
        duration: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} generators for batch size {batch_size}, but got {len(generator)}."
                )
            generators = generator
        else:
            generators = [generator] * batch_size

        latents = [
            torch.randn(
                duration,
                self.latent_dim,
                device=device,
                dtype=dtype,
                generator=generators[idx],
            )
            for idx in range(batch_size)
        ]
        return pad_sequence(latents, padding_value=0.0, batch_first=True)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        audio_end_in_s: float | None = None,
        duration: int | None = None,
        num_inference_steps: int = 16,
        guidance_scale: float = 4.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
    ):
        if prompt is None:
            prompt = []
        elif isinstance(prompt, str):
            prompt = [prompt]
        else:
            prompt = list(prompt)
        batch_size = len(prompt)
        if batch_size == 0:
            raise ValueError("`prompt` must contain at least one prompt.")

        device = self._execution_device
        normalized_prompts = [_normalize_text(text) for text in prompt]
        if duration is None:
            if audio_end_in_s is not None:
                duration = int(audio_end_in_s * self.sample_rate // self.latent_hop)
            else:
                duration = int(
                    _approx_batch_duration_from_prompts(normalized_prompts) * self.sample_rate // self.latent_hop
                )
        max_duration = int(self.max_wav_duration * self.sample_rate // self.latent_hop)
        duration = max(1, min(duration, max_duration))

        text_condition, text_condition_len = self.encode_prompt(normalized_prompts, device)
        duration_tensor = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        mask = _lens_to_mask(duration_tensor)
        text_mask = _lens_to_mask(text_condition_len, length=text_condition.shape[1])

        if negative_prompt is None:
            neg_text = torch.zeros_like(text_condition)
            neg_text_len = text_condition_len
            neg_text_mask = text_mask
        else:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            else:
                negative_prompt = list(negative_prompt)
                if len(negative_prompt) != batch_size:
                    raise ValueError(
                        f"`negative_prompt` must have batch size {batch_size}, but got {len(negative_prompt)} prompts."
                    )
            neg_text, neg_text_len = self.encode_prompt(negative_prompt, device)
            neg_text_mask = _lens_to_mask(neg_text_len, length=neg_text.shape[1])

        latent_cond = torch.zeros(batch_size, duration, self.latent_dim, device=device, dtype=text_condition.dtype)
        latents = self.prepare_latents(batch_size, duration, device, text_condition.dtype, generator=generator)
        num_inference_steps = max(2, num_inference_steps)
        timesteps = torch.linspace(0, 1, num_inference_steps, device=device, dtype=text_condition.dtype)
        sample = latents

        def model_step(curr_t: torch.Tensor, current_sample: torch.Tensor) -> torch.Tensor:
            pred = self.transformer(
                hidden_states=current_sample,
                encoder_hidden_states=text_condition,
                encoder_attention_mask=text_mask,
                timestep=curr_t.expand(batch_size),
                attention_mask=mask,
                latent_cond=latent_cond,
            ).sample
            if guidance_scale <= 1.0:
                return pred
            null_pred = self.transformer(
                hidden_states=current_sample,
                encoder_hidden_states=neg_text,
                encoder_attention_mask=neg_text_mask,
                timestep=curr_t.expand(batch_size),
                attention_mask=mask,
                latent_cond=latent_cond,
            ).sample
            return null_pred + (pred - null_pred) * guidance_scale

        for idx in range(len(timesteps) - 1):
            curr_t = timesteps[idx]
            dt = timesteps[idx + 1] - timesteps[idx]
            sample = sample + model_step(curr_t, sample) * dt

        if output_type == "latent":
            if not return_dict:
                return (sample,)
            return AudioPipelineOutput(audios=sample)

        waveform = self.vae.decode(sample.permute(0, 2, 1)).sample
        if output_type == "np":
            waveform = waveform.cpu().float().numpy()
        elif output_type != "pt":
            raise ValueError(f"Unsupported output_type: {output_type}")

        if not return_dict:
            return (waveform,)
        return AudioPipelineOutput(audios=waveform)
