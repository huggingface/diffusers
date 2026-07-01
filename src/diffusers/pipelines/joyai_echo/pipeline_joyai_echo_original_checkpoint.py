# Copyright 2026 JD.com and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ..pipeline_utils import DiffusionPipeline


class JoyAIEchoOriginalCheckpointPipeline(DiffusionPipeline):
    r"""
    Diffusers pipeline wrapper for running the original JoyAI-Echo release checkpoint.

    This class provides a diffusers entrypoint for the released JoyAI-Echo safetensors checkpoint while preserving the
    official inference math: Gemma prompt encoding is separated from generator loading, the distilled DMD sigma schedule
    predicts `x0`, and paired audio-video memory is chained across shots.

    Args:
        checkpoint_path (`str`):
            Path to the original JoyAI-Echo `.safetensors` checkpoint.
        gemma_path (`str`):
            Path to the Gemma text encoder directory.
        original_repo (`str`):
            Path to a JoyAI-Echo checkout containing `ltx-core`, `ltx-pipelines`, and `ltx-distillation`.
        device (`str`, defaults to `"cuda"`):
            Device used for inference.
        torch_dtype (`torch.dtype`, defaults to `torch.bfloat16`):
            Model dtype.
    """

    _optional_components = []

    def __init__(
        self,
        checkpoint_path: str,
        gemma_path: str,
        original_repo: str,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.gemma_path = str(Path(gemma_path).expanduser().resolve())
        self.original_repo = str(Path(original_repo).expanduser().resolve())
        self._joyai_echo_device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.register_to_config(
            checkpoint_path=self.checkpoint_path,
            gemma_path=self.gemma_path,
            original_repo=self.original_repo,
        )

        self._ensure_original_modules()
        self.generator = None
        self.video_vae = None
        self.audio_vae = None
        self.base_pipeline = None
        self.memory_pipeline = None
        self.audio_sample_rate = None

    @classmethod
    def from_original_checkpoint(
        cls,
        checkpoint_path: str,
        gemma_path: str,
        original_repo: str,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JoyAIEchoOriginalCheckpointPipeline":
        return cls(
            checkpoint_path=checkpoint_path,
            gemma_path=gemma_path,
            original_repo=original_repo,
            device=device,
            torch_dtype=torch_dtype,
        )

    def _ensure_original_modules(self) -> None:
        repo = Path(self.original_repo)
        for subpath in ["ltx-core/src", "ltx-pipelines/src", "ltx-distillation/src"]:
            path = str(repo / subpath)
            if path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _empty_cuda(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _move(module, target_device) -> None:
        if module is not None:
            module.to(target_device)

    def encode_prompts(self, prompts: list[str]) -> list[dict[str, Any]]:
        self._ensure_original_modules()
        from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper

        text_encoder = create_text_encoder_wrapper(
            checkpoint_path=self.checkpoint_path,
            gemma_path=self.gemma_path,
            device=self._joyai_echo_device,
            dtype=self.torch_dtype,
        )
        text_encoder.eval()

        cached = []
        for prompt in prompts:
            cond = text_encoder([prompt])
            cached.append({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in cond.items()})
            del cond

        del text_encoder
        gc.collect()
        self._empty_cuda(self._joyai_echo_device)
        return cached

    def load_generator(
        self,
        denoising_sigmas: list[float] | torch.Tensor,
        video_height: int,
        video_width: int,
        memory_downscale_factor: int = 1,
    ) -> None:
        self._ensure_original_modules()
        from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
        from ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers
        from ltx_distillation.utils import add_noise

        self.generator = create_ltx2_wrapper(
            checkpoint_path=self.checkpoint_path,
            gemma_path=self.gemma_path,
            device=self._joyai_echo_device,
            dtype=self.torch_dtype,
            video_height=int(video_height),
            video_width=int(video_width),
            loras=(),
        )
        self.generator.eval()

        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=self.checkpoint_path,
            device=torch.device("cpu"),
            dtype=self.torch_dtype,
            with_video_encoder=True,
            with_audio_encoder=True,
            decoder_device=torch.device("cpu"),
        )
        self.video_vae.eval()
        self.audio_vae.eval()

        denoising_sigmas = torch.as_tensor(denoising_sigmas, device=self._joyai_echo_device, dtype=torch.float32)
        self.base_pipeline = BidirectionalAVInferencePipeline(
            generator=self.generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
        )
        self.memory_pipeline = BidirectionalMemoryAVInferencePipeline(
            generator=self.generator,
            add_noise_fn=add_noise,
            denoising_sigmas=denoising_sigmas,
            memory_downscale_factor=int(memory_downscale_factor),
        )
        self.audio_sample_rate = self.audio_vae.get_output_sample_rate() or 24000

    def _stage_for_denoise(self) -> None:
        self._move(self.video_vae.encoder, "cpu")
        self._move(self.video_vae.decoder, "cpu")
        self._move(self.audio_vae.encoder, "cpu")
        self._move(self.audio_vae.decoder, "cpu")
        self._move(self.audio_vae.vocoder, "cpu")
        self._move(self.generator, self._joyai_echo_device)
        self._empty_cuda(self._joyai_echo_device)

    def _stage_for_video_encode(self) -> None:
        self._move(self.video_vae.encoder, self._joyai_echo_device)

    def _stage_after_video_encode(self) -> None:
        self._move(self.video_vae.encoder, "cpu")
        self._empty_cuda(self._joyai_echo_device)

    def _stage_for_decode(self) -> None:
        self._move(self.generator, "cpu")
        self._empty_cuda(self._joyai_echo_device)
        self._move(self.video_vae.decoder, self._joyai_echo_device)
        self._move(self.audio_vae.decoder, self._joyai_echo_device)
        self._move(self.audio_vae.vocoder, self._joyai_echo_device)

    @torch.no_grad()
    def __call__(
        self,
        prompts: list[str],
        output_dir: str | Path,
        cached_conds: list[dict[str, Any]] | None = None,
        num_frames: int = 241,
        height: int = 736,
        width: int = 1280,
        fps: int = 25,
        seed: int = 12345,
        memory_max_size: int = 7,
        num_fix_frames: int = 3,
        save_mode: str = "random_every_shot_frame",
        enable_audio_memory: bool = True,
        v2a_grad_scale: float = 2.0,
        memory_position_mode: str = "reference",
        audio_memory_window_size: int = 96,
        audio_memory_window_selection_mode: str = "max_response",
        video_memory_frame_selection_mode: str = "center",
        video_memory_clip_num_frames: int = 9,
        audio_memory_sample_rate: int = 16000,
        audio_memory_mel_bins: int = 128,
        audio_memory_mel_hop_length: int = 160,
        audio_memory_n_fft: int = 1024,
        audio_memory_downsample_factor: int = 4,
        audio_memory_is_causal: bool = True,
    ) -> dict[str, Any]:
        self._ensure_original_modules()
        import torchaudio
        from ltx_distillation.inference.memory_multishot import (
            PairedAudioVideoMemoryBank,
            audio_waveform_stats,
            build_paired_audio_memory_kwargs,
            video_uint8_to_pil_frames,
        )
        from ltx_distillation.utils import (
            compute_latent_shapes,
            concat_shot_audios,
            concat_shot_videos,
            decode_benchmark_sample,
            encode_memory_frames_batch,
            save_memory_bank_frames,
            write_benchmark_media,
        )

        if self.generator is None:
            raise RuntimeError("Call `load_generator(...)` before running the pipeline.")
        if cached_conds is None:
            cached_conds = self.encode_prompts(prompts)
        if len(cached_conds) != len(prompts):
            raise ValueError("`cached_conds` length must match `prompts` length.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_shape, audio_shape = compute_latent_shapes(
            num_frames=int(num_frames),
            video_height=int(height),
            video_width=int(width),
            batch_size=1,
            video_fps=float(fps),
        )
        memory_bank = PairedAudioVideoMemoryBank(
            max_size=int(memory_max_size),
            save_mode=str(save_mode),
            num_fix_frames=int(num_fix_frames),
        )

        shot_paths: list[Path] = []
        shot_audios: list[torch.Tensor] = []
        metadata: dict[str, Any] = {
            "checkpoint": self.checkpoint_path,
            "gemma_path": self.gemma_path,
            "num_prompts": len(prompts),
            "shots": [],
        }

        run_started = time.perf_counter()
        for shot_idx, prompt in enumerate(prompts):
            conditional_dict = {
                k: (v.to(self._joyai_echo_device) if isinstance(v, torch.Tensor) else v)
                for k, v in cached_conds[shot_idx].items()
            }
            prompt_seed = int(seed) + shot_idx
            memory_size_before = len(memory_bank)
            memory_video = None
            memory_audio_kwargs: dict[str, Any] = {}

            self._stage_for_denoise()
            with torch.random.fork_rng(devices=[self._joyai_echo_device]):
                torch.manual_seed(prompt_seed)
                if self._joyai_echo_device.type == "cuda":
                    torch.cuda.manual_seed(prompt_seed)

                if len(memory_bank) > 0:
                    self._stage_for_video_encode()
                    memory_video = encode_memory_frames_batch(
                        video_vae=self.video_vae,
                        batch_memory_frames=[memory_bank.get_memory_frames()],
                        target_h=int(height),
                        target_w=int(width),
                        device=self._joyai_echo_device,
                        dtype=self.torch_dtype,
                    )
                    self._stage_after_video_encode()

                    memory_audio_kwargs = build_paired_audio_memory_kwargs(
                        memory_bank,
                        enable_audio_memory=bool(enable_audio_memory),
                        v2a_grad_scale=float(v2a_grad_scale),
                        memory_position_mode=str(memory_position_mode),
                    )
                    video_latent, audio_latent = self.memory_pipeline.generate(
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        conditional_dict=conditional_dict,
                        memory_video=memory_video,
                        seed=prompt_seed,
                        **memory_audio_kwargs,
                    )
                else:
                    video_latent, audio_latent = self.base_pipeline.generate(
                        video_shape=tuple(video_shape),
                        audio_shape=tuple(audio_shape),
                        conditional_dict=conditional_dict,
                        seed=prompt_seed,
                    )

            del conditional_dict, memory_video, memory_audio_kwargs

            self._stage_for_decode()
            audio_memory_latent = (
                audio_latent.detach().cpu().contiguous()
                if (enable_audio_memory and audio_latent is not None)
                else None
            )
            video_uint8, audio_waveform = decode_benchmark_sample(
                self.video_vae, self.audio_vae, video_latent, audio_latent
            )
            memory_frames_for_bank = video_uint8_to_pil_frames(video_uint8)

            new_memory_metadata: dict[str, Any] = {}
            if audio_memory_latent is not None:
                new_memory_metadata = memory_bank.save_memory_slot(
                    memory_frames_for_bank,
                    audio_memory_latent,
                    audio_window_size=int(audio_memory_window_size),
                    video_clip_num_frames=int(video_memory_clip_num_frames),
                    audio_waveform=audio_waveform,
                    audio_sample_rate=int(audio_memory_sample_rate),
                    video_fps=float(fps),
                    audio_window_selection_mode=str(audio_memory_window_selection_mode),
                    video_frame_selection_mode=str(video_memory_frame_selection_mode),
                    audio_memory_mel_bins=int(audio_memory_mel_bins),
                    audio_memory_mel_hop_length=int(audio_memory_mel_hop_length),
                    audio_memory_n_fft=int(audio_memory_n_fft),
                    audio_memory_downsample_factor=int(audio_memory_downsample_factor),
                    audio_memory_is_causal=bool(audio_memory_is_causal),
                )

            save_memory_bank_frames(
                memory_bank.get_memory_frames(), output_dir / "memory_bank" / f"shot_{shot_idx:03d}"
            )

            shot_path = output_dir / f"shot_{shot_idx:03d}.mp4"
            write_result = write_benchmark_media(
                output_path=shot_path,
                video_uint8=video_uint8,
                audio_waveform=audio_waveform,
                fps=int(fps),
                audio_sr=int(self.audio_sample_rate),
            )
            shot_paths.append(shot_path)
            if audio_waveform is not None:
                shot_audios.append(audio_waveform.cpu())

            metadata["shots"].append(
                {
                    "shot_idx": int(shot_idx),
                    "prompt": prompt,
                    "output_path": str(shot_path),
                    "memory_size_before": int(memory_size_before),
                    "memory_size_after": int(len(memory_bank)),
                    "new_memory_entry": new_memory_metadata,
                    "audio_latent_shape": list(audio_latent.shape) if audio_latent is not None else None,
                    "wrote_audio_in_mp4": bool(write_result["wrote_audio_in_mp4"]),
                    "wrote_sidecar_wav": bool(write_result["wrote_sidecar_wav"]),
                    "audio_stats": write_result["audio_stats"],
                    "memory_entries": memory_bank.get_memory_metadata(),
                }
            )

            del video_latent, audio_latent, video_uint8, audio_waveform, audio_memory_latent, memory_frames_for_bank
            self._empty_cuda(self._joyai_echo_device)

        combined_path = output_dir / "combined_shots.mp4"
        concat_shot_videos(shot_paths, combined_path)
        combined_audio = concat_shot_audios(shot_audios)
        combined_audio_path = None
        if combined_audio is not None:
            combined_audio_path = output_dir / "combined_shots.wav"
            torchaudio.save(str(combined_audio_path), combined_audio, sample_rate=int(self.audio_sample_rate))

        metadata["combined_path"] = str(combined_path)
        metadata["combined_audio_path"] = str(combined_audio_path) if combined_audio_path else None
        metadata["combined_audio_stats"] = audio_waveform_stats(combined_audio)
        metadata["run_total_sec"] = round(time.perf_counter() - run_started, 3)
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metadata
