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
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = self.mel_spec.n_mel_channels
        self.num_channels = num_channels
        # sampling related
        self.odeint_kwargs = odeint_kwargs
        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map


    # simple utf-8 tokenizer, since paper went character based
    def list_str_to_tensor(self, text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
        list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
        text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
        return text


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
        if not exists(length):
            length = t.amax()

        seq = torch.arange(length, device=t.device)
        return seq[None, :] < t[:, None]
    

    def get_epss_timesteps(self, n, device, dtype):
        dt = 1 / 32
        predefined_timesteps = {
            5: [0, 2, 4, 8, 16, 32],
            6: [0, 2, 4, 6, 8, 16, 32],
            7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
            12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        }
        t = predefined_timesteps.get(n, [])
        if not t:
            return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
        return dt * torch.tensor(t, device=device, dtype=dtype)


    def convert_char_to_pinyin(self, text_list, polyphone=True):
        if jieba.dt.initialized is False:
            jieba.default_logger.setLevel(50)  # CRITICAL
            jieba.initialize()

        final_text_list = []
        custom_trans = str.maketrans(
            {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        )  # add custom trans here, to address oov

        def is_chinese(c):
            return (
                "\u3100" <= c <= "\u9fff"  # common chinese characters
            )

        for text in text_list:
            char_list = []
            text = text.translate(custom_trans)
            for seg in jieba.cut(text):
                seg_byte_len = len(bytes(seg, "UTF-8"))
                if seg_byte_len == len(seg):  # if pure alphabets and symbols
                    if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                        char_list.append(" ")
                    char_list.extend(seg)
                elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                    seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                    for i, c in enumerate(seg):
                        if is_chinese(c):
                            char_list.append(" ")
                        char_list.append(seg_[i])
                else:  # if mixed characters, alphabets and symbols
                    for c in seg:
                        if ord(c) < 256:
                            char_list.extend(c)
                        elif is_chinese(c):
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
            final_text_list.append(char_list)

        return final_text_list


    def __call__(
        self,
        ref_audio,
        ref_text,
        gen_text,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1,
        speed=1,
        fix_duration=None,
    ):


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


        cond = ref_audio

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(final_text_list, list):
            if exists(self.vocab_char_map):
                text = self.list_str_to_idx(final_text_list, self.vocab_char_map).to(device)
            else:
                text = self.list_str_to_tensor(final_text_list).to(device)
            assert text.shape[0] == batch

        # duration
        cond_mask = lens_to_mask(lens)
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in
        
        
        step_cond = self.conditioning_encoder(x, step_cond, text)

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode
        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            step_cond = self.conditioning_encoder(x, step_cond, text, drop_audio_cond=False, drop_text=False)
            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    time=t,
                    mask=mask,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance

            step_uncond = self.conditioning_encoder(x, step_uncond, text, drop_audio_cond=False, drop_text=False)
            step_cond = torch.cat((step_cond, step_uncond), dim=0)
            x = torch.cat((x, x), dim=0)
            t = torch.cat((t, t), dim=0)
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                time=t,
                mask=mask,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.transformer.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)
        t_start = 0

        # TODO Add Empirically Pruned Step Sampling for low NFE
        t = torch.linspace(t_start, 1, steps + 1, device=self.transformer.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        out = out.to(torch.float32)  # generated mel spectrogram
        out = out[:, ref_audio_len:, :]
        out = out.permute(0, 2, 1)
        generated_cpu = out[0].cpu().numpy()


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


    dit = DiT(**dit_config)
    print("DiT model initialized with config:", dit_config)
    
    conditioning_encoder_config = {
        'dim': 1024,
        'text_num_embeds': 256,
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
    )
    print("F5FlowPipeline initialized with DiT and Conditioning Encoder.")

