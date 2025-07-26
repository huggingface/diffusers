"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

import torchaudio
from librosa.filters import mel as librosa_mel_fn

import os
import random
from collections import defaultdict
from importlib.resources import files


import torch
from torch.nn.utils.rnn import pad_sequence
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from vocos import Vocos

# helpers







class F5FlowPipeline(DiffusionPipeline):
    def __init__(
        self,
        model: CFM,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        mel_spec_type="vocos",
        mel_spec_kwargs: dict = dict(),
        device=None,
    ):
        super().__init__()
        self.model = model
        self.vocoder = vocoder
        self.mel_spec_type = mel_spec_type
        self.mel_spec_kwargs = mel_spec_kwargs
        self.device = device or torch.device("cpu")

    def __call__(self, *args, **kwargs):
        return infer_process(*args, **kwargs, model_obj=self.model, vocoder=self.vocoder, device=self.device)


    def infer_batch_process(
        ref_audio,
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type="vocos",
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1,
        speed=1,
        fix_duration=None,
        device=None,
        streaming=False,
        chunk_size=2048,
    ):
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(device)

        generated_waves = []
        spectrograms = []

        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "

        def process_batch(gen_text):
            local_speed = speed
            if len(gen_text.encode("utf-8")) < 10:
                local_speed = 0.3

            # Prepare the text
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            ref_audio_len = audio.shape[-1] // hop_length
            if fix_duration is not None:
                duration = int(fix_duration * target_sample_rate / hop_length)
            else:
                # Calculate duration
                ref_text_len = len(ref_text.encode("utf-8"))
                gen_text_len = len(gen_text.encode("utf-8"))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

            # inference
            with torch.inference_mode():
                generated, _ = model_obj.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                del _

                generated = generated.to(torch.float32)  # generated mel spectrogram
                generated = generated[:, ref_audio_len:, :]
                generated = generated.permute(0, 2, 1)
                if mel_spec_type == "vocos":
                    generated_wave = vocoder.decode(generated)
                elif mel_spec_type == "bigvgan":
                    generated_wave = vocoder(generated)
                if rms < target_rms:
                    generated_wave = generated_wave * rms / target_rms

                # wav -> numpy
                generated_wave = generated_wave.squeeze().cpu().numpy()

                if streaming:
                    for j in range(0, len(generated_wave), chunk_size):
                        yield generated_wave[j : j + chunk_size], target_sample_rate
                else:
                    generated_cpu = generated[0].cpu().numpy()
                    del generated
                    yield generated_wave, generated_cpu

        if streaming:
            for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
                for chunk in process_batch(gen_text):
                    yield chunk
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
                for future in progress.tqdm(futures) if progress is not None else futures:
                    result = future.result()
                    if result:
                        generated_wave, generated_mel_spec = next(result)
                        generated_waves.append(generated_wave)
                        spectrograms.append(generated_mel_spec)

            if generated_waves:
                if cross_fade_duration <= 0:
                    # Simply concatenate
                    final_wave = np.concatenate(generated_waves)
                else:
                    # Combine all generated waves with cross-fading
                    final_wave = generated_waves[0]
                    for i in range(1, len(generated_waves)):
                        prev_wave = final_wave
                        next_wave = generated_waves[i]

                        # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                        cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                        cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                        if cross_fade_samples <= 0:
                            # No overlap possible, concatenate
                            final_wave = np.concatenate([prev_wave, next_wave])
                            continue

                        # Overlapping parts
                        prev_overlap = prev_wave[-cross_fade_samples:]
                        next_overlap = next_wave[:cross_fade_samples]

                        # Fade out and fade in
                        fade_out = np.linspace(1, 0, cross_fade_samples)
                        fade_in = np.linspace(0, 1, cross_fade_samples)

                        # Cross-faded overlap
                        cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                        # Combine
                        new_wave = np.concatenate(
                            [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                        )

                        final_wave = new_wave

                # Create a combined spectrogram
                combined_spectrogram = np.concatenate(spectrograms, axis=1)

                yield final_wave, target_sample_rate, combined_spectrogram

            else:
                yield None, target_sample_rate, None




