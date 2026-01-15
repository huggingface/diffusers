"""
ein notation: b - batch n - sequence nt - text sequence nw - raw wave length d - dimension
"""

import os

# helpers
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence

from diffusers.models.transformers.f5tts_transformer import F5ConditioningEncoder, F5DiTModel, MelSpec
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


class F5FlowPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "conditioning_encoder->transformer"

    def __init__(
        self,
        transformer: F5DiTModel,
        conditioning_encoder: F5ConditioningEncoder,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vocab_char_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.conditioning_encoder = conditioning_encoder
        self.mel_spec = MelSpec()
        num_channels = self.mel_spec.n_mel_channels
        self.num_channels = num_channels
        # sampling related
        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map
        self.scheduler = scheduler

        self.register_modules(
            transformer=transformer,
            conditioning_encoder=conditioning_encoder,
            scheduler=scheduler,
        )
        # self.register_to_config(
        #     vocab_char_map=vocab_char_map,
        # )

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        # Save vocab_char_map as JSON
        if self.vocab_char_map is not None:
            import json

            vocab_path = os.path.join(save_directory, "vocab_char_map.json")
            with open(vocab_path, "w") as f:
                json.dump(self.vocab_char_map, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load vocab_char_map if it exists
        import json

        vocab_char_map = None
        vocab_file = "vocab_char_map.json"

        # 1. Attempt to load from local directory
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_path = os.path.join(pretrained_model_name_or_path, vocab_file)
            if os.path.exists(vocab_path):
                with open(vocab_path, "r") as f:
                    vocab_char_map = json.load(f)
        else:
            # 2. Attempt to download from HF Hub
            vocab_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=vocab_file,
                subfolder=kwargs.get("subfolder"),
                revision=kwargs.get("revision"),
                cache_dir=kwargs.get("cache_dir"),
                token=kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            with open(vocab_path, "r") as f:
                vocab_char_map = json.load(f)

        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if vocab_char_map is not None:
            pipe.vocab_char_map = vocab_char_map
        return pipe

    # char tokenizer, based on custom dataset's extracted .txt file
    def list_str_to_idx(
        self,
        text: list[str] | list[list[str]],
        vocab_char_map: dict[str, int],  # {char: idx}
        padding_value=-1,
    ) -> "int[b nt]":  # noqa: F722
        list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
        text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
        return text

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

    def lens_to_mask(self, t: "int[b]", length: int | None = None) -> "bool[b n]":  # noqa: F722 F821
        if length is None:
            length = t.amax()

        seq = torch.arange(length, device=t.device)
        return seq[None, :] < t[:, None]

    def check_inputs(
        self,
        ref_audio: torch.Tensor | None,
        ref_text: Union[str, List[str]],
        gen_text: Union[str, List[str]],
        speed: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):
        if ref_audio is None:
            raise ValueError("`ref_audio` must be provided.")
        if not isinstance(ref_text, (str, list)):
            raise ValueError("`ref_text` must be a string or a list of strings.")
        if not isinstance(gen_text, (str, list)):
            raise ValueError("`gen_text` must be a string or a list of strings.")

        if not isinstance(ref_text, List):
            ref_text = [ref_text]

        if not isinstance(gen_text, List):
            gen_text = [gen_text]

        if len(ref_text) != len(gen_text):
            raise ValueError("`ref_text` and `gen_text` must have the same length.")

        # check if speed is non negative
        if speed is not None:
            if not isinstance(speed, torch.Tensor) and not isinstance(speed, List):
                raise ValueError("`speed` must be a torch.Tensor or a list of torch.Tensors.")
            if isinstance(speed, List):
                speed = torch.stack(speed)
                speed = speed.squeeze(-1)
            if (speed < 0).any():
                raise ValueError("`speed` must be non-negative.")
            if speed.ndim != 1:
                raise ValueError("`speed` must be a 1D tensor.")
            if speed.shape[0] != len(ref_text):
                raise ValueError("`speed` must have the same length as `ref_text` and `gen_text`.")

    def prepare_latents(
        self,
        ref_audio: torch.Tensor,
        ref_text: Union[str, List[str]],
        gen_text: Union[str, List[str]],
        speed: Optional[torch.Tensor] = None,
        guidance_scale=2.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # each text in text_list is a combination of ref_text and gen_text
        if isinstance(ref_text, str):
            ref_text = [ref_text]
        if isinstance(gen_text, str):
            gen_text = [gen_text]
        text_list = [f"{r} {g}" for r, g in zip(ref_text, gen_text)]
        ref_audio_len = ref_audio.shape[-1] // self.mel_spec.hop_length

        if isinstance(speed, List):
            speed = torch.stack(speed)
            speed = speed.squeeze(-1)

        if speed is None:
            speed = torch.ones(len(ref_text), device=ref_audio.device)
        # Calculate duration from speed
        duration_list = []

        for i in range(len(ref_text)):
            ref_text_len = len(ref_text[i].encode("utf-8"))
            gen_text_len = len(gen_text[i].encode("utf-8"))
            duration = ref_audio_len + int(
                (ref_audio_len * speed[i]) * ((ref_text_len + gen_text_len + 1) / ref_text_len)
            )
            duration_list.append(duration)
        duration = torch.tensor(duration_list, dtype=torch.long, device=ref_audio.device)

        cond = ref_audio
        if cond.ndim == 2:
            cond = cond.to("cpu")  # mel spec needs cpu
            cond = self.mel_spec(cond)
            cond = cond.to(self._execution_device)
            cond = cond.permute(0, 2, 1)
            if len(ref_text) > 1:
                # repeat cond for batch inference, TODO allow different conds in batch
                cond = cond.repeat(len(ref_text), 1, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text = self.list_str_to_idx(text_list, self.vocab_char_map).to(device)
        duration = duration.to(device)

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
        for i, dur in enumerate(duration):
            y0.append(
                randn_tensor(
                    (dur, self.num_channels),
                    device=device,
                    generator=generator[i] if isinstance(generator, list) else generator,
                )
            )
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        return y0, step_cond_input, text, cond, cond_mask, mask

    @torch.no_grad()
    def __call__(
        self,
        ref_audio: torch.Tensor | None = None,
        ref_text: Union[str, List[str]] = None,
        gen_text: Union[str, List[str]] = None,
        num_inference_steps=32,
        guidance_scale=2.0,
        sway_sampling_coef=-1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        speed: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # Check inputs
        self.check_inputs(ref_audio, ref_text, gen_text, speed)
        device = self._execution_device

        y0, step_cond_input, text, cond, cond_mask, mask = self.prepare_latents(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        sigmas = self.get_epss_timesteps(num_inference_steps, device, step_cond_input.dtype)
        if sway_sampling_coef is not None:
            sigmas = sigmas + sway_sampling_coef * (torch.cos(torch.pi / 2 * sigmas) - 1 + sigmas)
        timesteps = sigmas * (self.scheduler.num_train_timesteps - 1)
        timesteps = timesteps.round().long()
        self.scheduler.set_timesteps(
            num_inference_steps + 1, device=device, sigmas=sigmas.cpu(), timesteps=timesteps.cpu()
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(sigmas[:-1]):
                step_cond = self.conditioning_encoder(
                    y0, step_cond_input, text, drop_audio_cond=False, drop_text=False
                )
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
                step_uncond = self.conditioning_encoder(
                    y0, step_cond_input, text, drop_audio_cond=True, drop_text=True
                )
                step_cond = torch.cat((step_cond, step_uncond), dim=0)
                pred_cfg = self.transformer(
                    x=step_cond,
                    time=t,
                    mask=mask,
                    cache=True,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                pred = pred + (pred - null_pred) * guidance_scale

                y0 = self.scheduler.step(pred, t, y0, generator=generator).prev_sample

                progress_bar.update()

        sampled = y0
        out = sampled
        out = torch.where(cond_mask, cond, out)

        out = out.to(torch.float32)  # generated mel spectrogram
        audio = out.permute(0, 2, 1)

        # Offload all models
        self.maybe_free_model_hooks()
        if not return_dict:
            return (audio,)
        return AudioPipelineOutput(audios=audio)


if __name__ == "__main__":
    print("entering main funcitn")

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

    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)

    dit = F5DiTModel(**dit_config)
    print("DiT model initialized with config:", dit_config)

    conditioning_encoder_config = {
        "dim": 1024,
        "text_num_embeds": vocab_size,
        "text_dim": 512,
        "text_mask_padding": True,
        "conv_layers": 4,
        "mel_dim": mel_spec_config["n_mel_channels"],
    }
    conditioning_encoder = F5ConditioningEncoder(**conditioning_encoder_config)
    print("Conditioning Encoder initialized with config:", conditioning_encoder_config)

    scheduler = FlowMatchEulerDiscreteScheduler()

    f5_pipeline = F5FlowPipeline(
        transformer=dit, conditioning_encoder=conditioning_encoder, vocab_char_map=vocab_char_map, scheduler=scheduler
    )
    print("F5FlowPipeline initialized with DiT and Conditioning Encoder.")

    import torch

    ref_audio = torch.randn(2, 16000)  # Dummy reference audio
    duration = 250

    ref_text = "This is a test sentence."  # Dummy reference text
    gen_text = "This is a generated sentence."  # Dummy generated text

    ref_text = [ref_text] * 2
    gen_text = [gen_text] * 2

    x = f5_pipeline(
        ref_audio=ref_audio, ref_text=ref_text, gen_text=gen_text, num_inference_steps=2, return_dict=False
    )
    print("Generated output shape:", x[0].shape)
