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

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from vocos import Vocos
from diffusers.models.transformers.f5tts_transformer import   DiT, MelSpec, ConditioningEncoder
# helpers
import jieba
from pypinyin import lazy_pinyin, Style
from typing import Optional, Union, List






class F5FlowPipeline(DiffusionPipeline):
    def __init__(
        self,
        transformer: DiT,
        conditioning_encoder: ConditioningEncoder,
        odeint_kwargs: dict = dict(
            method="euler" 
        ),
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.conditioning_encoder = conditioning_encoder
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = self.mel_spec.n_mel_channels
        self.num_channels = num_channels
        # sampling related
        self.odeint_kwargs = odeint_kwargs
        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map



    # char tokenizer, based on custom dataset's extracted .txt file
    def list_str_to_idx(
        self,
        text: list[str] | list[list[str]],
        vocab_char_map: dict[str, int],  # {char: idx}
        padding_value=-1,
    ) -> int["b nt"]:  # noqa: F722
        list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
        text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
        return text
    



    def lens_to_mask(self, t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
        if length is None:
            length = t.amax()

        seq = torch.arange(length, device=t.device)
        return seq[None, :] < t[:, None]
    
    
    def check_inputs(self, ref_audio: torch.Tensor | None, ref_text: Union[str, List[str]], gen_text: Union[str, List[str]], duration: Optional[torch.Tensor] = None): 
        if ref_audio is None:
            raise ValueError("`ref_audio` must be provided.")
        if not isinstance(ref_text, (str, list)):
            raise ValueError("`ref_text` must be a string or a list of strings.")
        if not isinstance(gen_text, (str, list)):
            raise ValueError("`gen_text` must be a string or a list of strings.")

        if len(ref_text) != len(gen_text):
            raise ValueError("`ref_text` and `gen_text` must have the same length.")

        # check if duration is non negative
        if duration is not None:
            if not isinstance(duration, torch.Tensor):
                raise ValueError("`duration` must be a torch.Tensor.")
            if (duration < 0).any():
                raise ValueError("`duration` must be non-negative.")
            if duration.ndim != 1:
                raise ValueError("`duration` must be a 1D tensor.")
            if duration.shape[0] != len(ref_text):
                raise ValueError("`duration` must have the same length as `ref_text` and `gen_text`.")
            
    def prepare_latents(self, ref_audio: torch.Tensor, ref_text: Union[str, List[str]], gen_text: Union[str, List[str]], duration: Optional[torch.Tensor] = None, guidance_scale=2.0, generator: Optional[torch.Generator] = None):
        # each text in text_list is a combination of ref_text and gen_text
        if isinstance(ref_text, str):
            ref_text = [ref_text]
        if isinstance(gen_text, str):
            gen_text = [gen_text]
        text_list = [f"{r} {g}" for r, g in zip(ref_text, gen_text)]
        ref_audio_len = ref_audio.shape[-1] // self.mel_spec.hop_length

        if duration is None:
            # Calculate duration

            
            duration_list = []
            
            for i in range(len(ref_text)):
                ref_text_len = len(ref_text[i].encode("utf-8"))
                gen_text_len = len(gen_text[i].encode("utf-8"))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
                duration_list.append(duration)
            duration = torch.tensor(duration_list, dtype=torch.long, device=ref_audio.device)

        cond = ref_audio

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text = self.list_str_to_idx(text_list, self.vocab_char_map).to(device)
        assert text.shape[0] == batch

        # duration
        cond_mask = self.lens_to_mask(lens)
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        max_duration = duration.amax()
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond_input = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in
        if batch > 1:
            mask = self.lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None
            
        if guidance_scale >= 1e-5 and mask is not None:
            mask = torch.cat((mask, mask), dim=0)  # for classifier-free guidance, we need to double the batch size
            
        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            y0.append(torch.randn(dur, self.num_channels, device=device, dtype=step_cond_input.dtype, generator=generator))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        return y0, step_cond_input, text, cond, cond_mask, mask



    def __call__(
        self,
        ref_audio: torch.Tensor | None = None,
        ref_text: Union[str, List[str]] = None,
        gen_text: Union[str, List[str]] = None,
        num_inference_steps=32,
        guidance_scale=2.0,
        sway_sampling_coef=-1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        duration: Optional[torch.Tensor] = None,
        seed=None,
        device="cuda"
    ):


        # Check inputs
        self.check_inputs(ref_audio, ref_text, gen_text, duration)

        y0, step_cond_input, text, cond, cond_mask, mask = self.prepare_latents(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            duration=duration,
            guidance_scale=guidance_scale
        )
            
            
        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            step_cond = self.conditioning_encoder(x, step_cond_input, text, drop_audio_cond=False, drop_text=False)
            # predict flow (cond)
            if guidance_scale < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    time=t,
                    mask=mask,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            step_uncond = self.conditioning_encoder(x, step_cond_input, text, drop_audio_cond=True, drop_text=True)
            step_cond = torch.cat((step_cond, step_uncond), dim=0)            
            pred_cfg = self.transformer(
                x=step_cond,
                time=t,
                mask=mask,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * guidance_scale
        
        
        t_start = 0
        # TODO Add Empirically Pruned Step Sampling for low NFE
        t = torch.linspace(t_start, 1, num_inference_steps + 1, device=device, dtype=step_cond_input.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)


        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        out = out.to(torch.float32)  # generated mel spectrogram
        out = out.permute(0, 2, 1)
        generated_cpu = out[0]


        # This need to be in HF Output format
        return generated_cpu



if __name__ == "__main__":
    print('entering main funcitn')
    
    dit_config = {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_num_embeds": 256,
        "text_mask_padding": True,
        "qk_norm": None,  # null | rms_norm
        "conv_layers": 4,
        "pe_attn_head": None,
        "attn_backend": "torch",  # torch | flash_attn
        "attn_mask_enabled": False,
        "checkpoint_activations": False,  # recompute activations and save memory for extra compute
    }
    
  
    mel_spec_config = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
    }


    with open('vocab.txt', "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)


    dit = DiT(**dit_config)
    print("DiT model initialized with config:", dit_config)
    
    conditioning_encoder_config = {
        'dim': 1024,
        'text_num_embeds': vocab_size,
        'text_dim': 512,
        'text_mask_padding': True,
        'conv_layers': 4,
        'mel_dim': mel_spec_config['n_mel_channels'],
    }
    conditioning_encoder = ConditioningEncoder(**conditioning_encoder_config)
    print("Conditioning Encoder initialized with config:", conditioning_encoder_config)
    
    f5_pipeline = F5FlowPipeline(
        transformer=dit,
        conditioning_encoder=conditioning_encoder,
        odeint_kwargs={"method": "euler"},
        mel_spec_kwargs=mel_spec_config,
        vocab_char_map=vocab_char_map,
    )
    print("F5FlowPipeline initialized with DiT and Conditioning Encoder.")

    import torch 
    ref_audio = torch.randn(2, 16000)  # Dummy reference audio
    duration = 250

    ref_text = "This is a test sentence."  # Dummy reference text
    gen_text = "This is a generated sentence."  # Dummy generated text

    ref_text = [ref_text]*2
    gen_text = [gen_text]*2

    x = f5_pipeline(ref_audio=ref_audio,
                    ref_text=ref_text,
                     gen_text=gen_text,
                     device='cpu', num_inference_steps=2)
    print("Generated output shape:", x.shape)
                            