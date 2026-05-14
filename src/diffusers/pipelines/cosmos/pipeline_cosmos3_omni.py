# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer

from ...models.autoencoders.autoencoder_cosmos3_avae import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
)
from ...schedulers import UniPCMultistepScheduler
from ...utils import BaseOutput
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .sequence_packing import (
    GenerationDataClean,
    SequencePlan,
    build_packed_sequence,
    build_sequence_plans_from_data_batch,
    get_all_seq,
    pack_input_sequence,
)


_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."


@dataclass
class Cosmos3OmniPipelineOutput(BaseOutput):
    """Output dataclass for :class:`Cosmos3OmniDiffusersPipeline`.

    Attributes:
        video: List of decoded video tensors, one per generated sample,
            each of shape ``[C, T, H, W]`` in ``[0, 1]`` (or raw latents when
            ``output_type="latent"``).
        sound: List of decoded audio waveforms of shape ``[C, N]``, one per
            sample.  ``None`` when ``enable_sound=False``.
    """

    video: list
    sound: Optional[list] = None


def save_img_or_video(sample, save_fp_wo_ext, fps=24, quality=10, ffmpeg_params=None, **kwargs):
    # TODO: remove this function and use diffusers-style vidoe processor
    # However, it may cause numerical differences in the saved video, so we keep it for now for exact reproducibility of saved videos.
    import imageio
    from PIL import Image as PILImage

    assert sample.ndim == 4, "Only support 4D tensor"

    if torch.is_floating_point(sample):
        sample = sample.clamp(0, 1)
    else:
        assert sample.dtype == torch.uint8, "Only support uint8 tensor"
        sample = sample.float().div(255)

    if sample.shape[1] == 1:
        save_obj = PILImage.fromarray(
            rearrange((sample.cpu().float().numpy() * 255), "c 1 h w -> h w c").astype(np.uint8),
            mode="RGB",
        )
        save_obj.save(f"{save_fp_wo_ext}.jpg", format="JPEG", quality=85 if quality is None else quality)
    else:
        frames = rearrange((sample.cpu().float().numpy() * 255), "c t h w -> t h w c").astype(np.uint8)
        h, w = frames.shape[1], frames.shape[2]
        out_ffmpeg_params = ffmpeg_params if ffmpeg_params is not None else ["-s", f"{w}x{h}"]
        imageio.mimsave(
            f"{save_fp_wo_ext}.mp4",
            frames,
            fps=fps,
            quality=quality,
            macro_block_size=1,
            ffmpeg_params=out_ffmpeg_params,
            output_params=["-f", "mp4"],
        )


def save_wav(waveform: torch.Tensor, path, sample_rate: int) -> None:
    """Save a decoded waveform ``[C, N]`` or ``[N]`` as a WAV file.

    Args:
        waveform: Audio tensor of shape ``[C, N]`` (multi-channel) or ``[N]`` (mono).
        path: Destination file path (``str`` or :class:`~pathlib.Path`).  The ``.wav``
            extension is expected but not enforced.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf  # type: ignore[import-not-found]

    audio_np = waveform.clamp(-1.0, 1.0).to(dtype=torch.float32).cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # soundfile expects [N, C]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_np, sample_rate)


class DiffusersWan22VAE:
    """
    Drop-in replacement for Wan2pt2VAEInterface, backed by AutoencoderKLWan.

    Bridges the following interface differences:

    1. encode – AutoencoderKLWan returns AutoencoderKLOutput(latent_dist=
       DiagonalGaussianDistribution); we extract .mode() and apply the same
       (μ - mean) * inv_std normalization that WanVAE does internally.

    2. decode – AutoencoderKLWan expects un-normalized z and returns
       DecoderOutput(sample=…); we invert the normalization before calling
       decode and unwrap the result to a plain tensor.

    3. spatial/temporal_compression_factor properties – AutoencoderKLWan
       stores these as config.scale_factor_spatial / scale_factor_temporal
       and exposes spatial_compression_ratio (not *_factor).

    Note: AutoencoderKLWan._decode() clamps the output to [-1, 1];
    Wan2pt2VAEInterface does not.  The pipeline applies .clamp(0, 1) after
    decode so this difference does not affect saved videos.

    Numerical equivalence requirements (needed for bitwise-identical output
    vs Wan2pt2VAEInterface):

    - No torch.amp.autocast: Wan2pt2VAEInterface constructs WanVAE with
      is_amp=False, so the encoder/decoder run as pure bfloat16 with no
      autocast context.  Wrapping calls in autocast changes how ops such as
      F.normalize accumulate internally and breaks the match.

    - mean / inv_std must be initialised directly in `dtype` (bfloat16).
      WanVAE.__init__ does:
          self.std    = torch.tensor(std, dtype=bfloat16)
          self.scale  = [self.mean, 1.0 / self.std]  # division in bfloat16
      Computing 1/std in float32 and then casting to bfloat16 can yield
      different bit patterns, so we must perform the division in bfloat16
      from the start.
    """

    def __init__(self, vae: AutoencoderKLWan, dtype: torch.dtype = torch.bfloat16):
        self.vae = vae
        self.dtype = dtype
        # Initialise in `dtype` so 1/std is computed in bfloat16, matching WanVAE.
        mean = torch.tensor(vae.config.latents_mean, dtype=dtype)
        std = torch.tensor(vae.config.latents_std, dtype=dtype)
        self._mean = mean  # [z_dim]
        self._inv_std = 1.0 / std  # [z_dim]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] -> [B,z_dim,T//4,H//16,W//16]  (normalized μ, matching Wan2pt2VAEInterface)"""
        in_dtype = x.dtype
        device = x.device
        mean = self._mean.to(device=device, dtype=self.dtype)
        inv_std = self._inv_std.to(device=device, dtype=self.dtype)
        # No autocast — mirrors WanVAE(is_amp=False), pure bfloat16 forward pass.
        raw_mu = self.vae.encode(x.to(self.dtype)).latent_dist.mode()
        normalized = (raw_mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)
        return normalized.to(in_dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B,z_dim,T_lat,H_lat,W_lat] -> [B,3,T,H,W]"""
        in_dtype = z.dtype
        device = z.device
        mean = self._mean.to(device=device, dtype=self.dtype)
        inv_std = self._inv_std.to(device=device, dtype=self.dtype)
        z_raw = z.to(self.dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        # No autocast — mirrors WanVAE(is_amp=False), pure bfloat16 forward pass.
        out = self.vae.decode(z_raw).sample
        return out.to(in_dtype)

    @property
    def spatial_compression_factor(self) -> int:
        return self.vae.config.scale_factor_spatial

    @property
    def temporal_compression_factor(self) -> int:
        return self.vae.config.scale_factor_temporal


class Cosmos3OmniDiffusersPipeline(DiffusionPipeline):
    _optional_components = ["sound_tokenizer"]
    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: Cosmos3OmniTransformer,
        text_tokenizer: AutoTokenizer,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        sound_tokenizer: Optional[Cosmos3AVAEAudioTokenizer] = None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
        )
        # Plain attribute (not registered): registering the wrapper would cause save_pretrained to call
        # wrapper.save_pretrained(), which fails since DiffusersWan22VAE has no such method.
        self.vision_tokenizer = DiffusersWan22VAE(vae)

        self.llm_special_tokens = {
            "start_of_generation": text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "end_of_generation": text_tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            "eos_token_id": text_tokenizer.eos_token_id,
        }

    def tokenize_caption(
        self,
        caption: str,
        is_video: bool = False,
        use_system_prompt: bool = False,
    ) -> list[int]:
        """Tokenize a text caption into token IDs using the Qwen2 chat template.
        Returns:
            List of token IDs representing the full chat-formatted caption.
        """
        conversations = []
        # Optionally prepend a system prompt that tells the model whether it is generating
        # an image or a video. This changes the conditioning context for the LLM.
        if use_system_prompt:
            _system_prompt = _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE
            conversations.append({"role": "system", "content": _system_prompt})
        conversations.append({"role": "user", "content": caption})

        tokenizer_output = self.text_tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
        )
        return tokenizer_output

    def apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: List[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            assert noisy_indexes_i.numel() <= token_shape_i[0]
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            noisy_indexes_i = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
            noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())
            start_noisy_index += math.prod(token_shape_i)
        flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)
        assert packed_tokens.dim() == 2
        assert packed_timestep_embeds.dim() == 2
        assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
        assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
        assert flattened_noisy_frame_indexes.dim() == 1
        assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]
        flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
            -1,
            packed_tokens.shape[1],
        )
        return packed_tokens.scatter_add(
            dim=0,
            index=flattened_noisy_frame_indexes,
            src=packed_timestep_embeds,
        )

    def patchify_and_pack_latents(
        self,
        latent_patch_size: int,
        latent_channel: int,
        tokens_vision: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = latent_patch_size
        packed_latent = []
        original_latent_shapes = []
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C,T,H,W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        latent_patch_size: int,
        latent_channel: int,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        p = latent_patch_size
        unpatchified_latents = []
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            if original_latent_shapes is not None:
                _, h_orig, w_orig = original_latent_shapes[i]
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                h_orig, w_orig = h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def decode_vision(
        self,
        patch_latent_dim: int,
        latent_patch_size: int,
        latent_channel: int,
        packed_seq,
        last_hidden_state: torch.Tensor,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Decode vision predictions from last_hidden_state. Returns preds_vision list."""
        vision = packed_seq.vision
        has_noisy_vision = (
            vision is not None
            and vision.tokens is not None
            and isinstance(vision.mse_loss_indexes, torch.Tensor)
            and vision.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_vision:
            preds_vision = torch.zeros(
                [1, patch_latent_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            preds_vision = self.transformer.vae2llm(preds_vision)
            preds_vision = self.transformer.llm2vae(preds_vision)
            if vision is not None and vision.tokens is not None:
                preds_vision_list = [torch.zeros_like(tok) for tok in vision.tokens]
                preds_vision_list[0] = preds_vision_list[0] + 0.0 * preds_vision.sum()
            else:
                preds_vision_list = [preds_vision]
        else:
            assert vision is not None
            assert isinstance(vision.mse_loss_indexes, torch.Tensor)
            assert vision.noisy_frame_indexes is not None
            preds_vision = self.transformer.llm2vae(last_hidden_state[vision.mse_loss_indexes])
            preds_vision_list = self.unpatchify_and_unpack_latents(
                latent_patch_size,
                latent_channel,
                preds_vision,
                token_shapes_vision=vision.token_shapes,
                noisy_frame_indexes_vision=vision.noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )
        return preds_vision_list

    def _check_sound_enabled(self) -> None:
        """Fail-fast guard: raise if sound_tokenizer or sound_gen config is missing."""
        if self.sound_tokenizer is None:
            raise ValueError(
                "enable_sound=True requires a sound_tokenizer. "
                "Load a checkpoint that includes sound_tokenizer/ (e.g. the 6cd74411 checkpoint)."
            )
        if not getattr(self.transformer.config, "sound_gen", False):
            raise ValueError(
                "enable_sound=True but the transformer was not trained with sound_gen=True. "
                "Use a sound-capable checkpoint."
            )

    def _pack_sound_latents(
        self,
        tokens_sound: list,
        token_shapes_sound: list,
    ) -> torch.Tensor:
        """Pack per-sample sound latents into a single 2-D tensor.

        Args:
            tokens_sound: List of ``[C, T]`` tensors, one per sample.
            token_shapes_sound: List of ``(T, 1, 1)`` tuples (from packed_seq.sound).

        Returns:
            ``[total_T, C]`` packed tensor.
        """
        packed = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            packed.append(sound[:, :T].permute(1, 0))  # [C, T] → [T, C]
        return torch.cat(packed, dim=0)  # [total_T, C]

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list,
        noisy_frame_indexes_sound: list,
    ) -> list:
        """Unpack packed sound predictions back to per-sample ``[C, T]`` tensors.

        Args:
            packed_preds: ``[total_noisy_T, C]`` predictions at noisy positions.
            token_shapes_sound: List of ``(T, 1, 1)`` tuples per sample.
            noisy_frame_indexes_sound: List of ``[T_noisy]`` index tensors per sample.

        Returns:
            List of ``[C, T]`` tensors (zeros at conditioned positions).
        """
        sound_dim = self.transformer.config.sound_dim
        unpacked = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            output = torch.zeros(
                (sound_dim, T),
                device=packed_preds.device,
                dtype=packed_preds.dtype,
            )
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    def encode_sound_tokens(
        self,
        timestep_scale: float,
        packed_seq,
        hidden_states: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Project sound tokens into ``hidden_states`` in-place.

        Projects sound latents into ``hidden_states`` via ``sound2llm``, modality embed, and timestep embeddings.

        Steps:
        1. Pack latents: list of ``[C, T]`` → ``[total_T, C]``
        2. Project: ``sound2llm`` + ``sound_modality_embed``
        3. Add timestep embeddings to noisy frames
        4. Scatter into ``hidden_states`` at ``sound.sequence_indexes``
        """
        if packed_seq.sound is None or packed_seq.sound.tokens is None:
            return

        sound = packed_seq.sound
        assert sound.token_shapes is not None
        assert isinstance(sound.sequence_indexes, torch.Tensor)
        assert isinstance(sound.timesteps, torch.Tensor)
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)

        packed_tokens_sound = self._pack_sound_latents(sound.tokens, sound.token_shapes)
        packed_tokens_sound = packed_tokens_sound.to(target_dtype)

        packed_tokens_sound = self.transformer.sound2llm(packed_tokens_sound) + self.transformer.sound_modality_embed

        if sound.mse_loss_indexes.numel() > 0:
            timesteps_sound = sound.timesteps * timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_sound = self.transformer.time_embedder(timesteps_sound)
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self.apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )

        hidden_states[sound.sequence_indexes] = packed_tokens_sound

    def decode_sound_tokens(
        self,
        packed_seq,
        last_hidden_state: torch.Tensor,
    ) -> list:
        """Decode sound predictions from transformer hidden states.

        Extracts sound predictions from hidden states via ``llm2sound`` and unpacks them back to latent shape.
        Includes a dummy forward path for graph-consistency when no noisy tokens.

        Returns:
            List of ``[C, T]`` tensors, one per sample.
        """
        sound = packed_seq.sound
        has_noisy_sound = (
            sound is not None
            and sound.tokens is not None
            and isinstance(sound.mse_loss_indexes, torch.Tensor)
            and sound.mse_loss_indexes.numel() > 0
        )

        if not has_noisy_sound:
            sound_dim = self.transformer.config.sound_dim
            dummy = torch.zeros(
                [1, sound_dim],
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype,
            )
            dummy = self.transformer.sound2llm(dummy) + self.transformer.sound_modality_embed
            dummy = self.transformer.llm2sound(dummy)
            if sound is not None and sound.tokens is not None:
                preds = [torch.zeros_like(tok) for tok in sound.tokens]
                preds[0] = preds[0] + 0.0 * dummy.sum()
            else:
                preds = [dummy]
            return preds

        assert sound is not None
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)
        preds_packed = self.transformer.llm2sound(last_hidden_state[sound.mse_loss_indexes])
        return self._unpack_sound_latents(preds_packed, sound.token_shapes, sound.noisy_frame_indexes)

    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a sound latent ``[C, T]`` to a waveform ``[audio_ch, N]``.

        Adds/removes the batch dimension expected by the sound tokenizer decoder.
        """
        assert self.sound_tokenizer is not None
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        waveform = self.sound_tokenizer.decode(latent.unsqueeze(0).to(decoder_dtype))  # [1, audio_ch, N]
        return waveform.squeeze(0)  # [audio_ch, N]

    def normalize_video_databatch_inplace(
        self,
        input_video_key: str,
        data_batch: dict,
        input_key: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        input_key = input_video_key if input_key is None else input_key
        if input_key in data_batch:
            if data_batch.get("is_preprocessed", False) is True:
                for i in range(len(data_batch[input_key])):
                    assert torch.is_floating_point(data_batch[input_key][i])
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001))
            else:
                for i in range(len(data_batch[input_key])):
                    item = data_batch[input_key][i]
                    if isinstance(item, torch.Tensor):
                        item = [item]
                    assert item[0].dtype == torch.uint8
                    data_batch[input_key][i] = torch.stack(item).to(device=device, dtype=dtype) / 127.5 - 1.0
                data_batch["is_preprocessed"] = True

    def augment_image_dim_inplace(
        self,
        input_image_key: str,
        data_batch: dict,
        input_key: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        input_key = input_image_key if input_key is None else input_key
        if input_key in data_batch:
            if data_batch.get("is_preprocessed", False) is True:
                for i in range(len(data_batch[input_key])):
                    assert data_batch[input_key][i].shape[2] == 1
                return
            else:
                new_image_tensor_list = []
                for i in range(len(data_batch[input_key])):
                    for img_tensor in data_batch[input_key][i]:
                        img_tensor = rearrange(img_tensor, "c h w -> 1 c 1 h w").contiguous()
                        if img_tensor.dtype == torch.uint8:
                            img_tensor = img_tensor.to(device=device, dtype=dtype) / 127.5 - 1.0
                        new_image_tensor_list.append(img_tensor)
                data_batch[input_key] = new_image_tensor_list
                data_batch["is_preprocessed"] = True

    def remove_padding_from_latent(
        self,
        spatial_compression_factor: int,
        x0_tokens_vision: list[torch.Tensor],
        frame_size: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        cropped_latents = []
        for i in range(len(x0_tokens_vision)):
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())
            orig_h_latent = orig_h // spatial_compression_factor
            orig_w_latent = orig_w // spatial_compression_factor
            cropped_latents.append(x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous())
        return cropped_latents

    def get_data_and_condition(
        self,
        input_image_key: str,
        input_video_key: str,
        data_batch: dict,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> GenerationDataClean:
        assert (input_image_key in data_batch) != (input_video_key in data_batch)
        is_img = input_image_key in data_batch
        sample_vision_list = data_batch[input_image_key if is_img else input_video_key]

        if "num_vision_items_per_sample" not in data_batch:
            has_multiple_vision_per_sample = any(
                isinstance(v, (list, tuple)) and len(v) > 1 for v in sample_vision_list
            )
            num_vision_items_per_sample: list[int] | None = (
                [len(v) for v in sample_vision_list] if has_multiple_vision_per_sample else None
            )
            data_batch["num_vision_items_per_sample"] = num_vision_items_per_sample
            if has_multiple_vision_per_sample:
                media_key = input_video_key if not is_img else input_image_key
                data_batch[media_key] = [item.unsqueeze(0) for sublist in sample_vision_list for item in sublist]
                if data_batch[media_key][0].dtype == torch.float32 and not is_img:
                    data_batch["is_preprocessed"] = True
        else:
            num_vision_items_per_sample = data_batch["num_vision_items_per_sample"]

        batch_size = (
            len(sample_vision_list) if num_vision_items_per_sample is None else len(num_vision_items_per_sample)
        )

        self.normalize_video_databatch_inplace(input_video_key, data_batch, device=device, dtype=dtype)
        self.augment_image_dim_inplace(input_image_key, data_batch, device=device, dtype=dtype)
        raw_state_vision = data_batch[input_image_key if is_img else input_video_key]
        x0_tokens_vision = [
            self.vision_tokenizer.encode(raw_state_vision_i).contiguous().float()
            for raw_state_vision_i in raw_state_vision
        ]

        frame_size = data_batch.get("image_size", None)
        if frame_size is not None:
            x0_tokens_vision = self.remove_padding_from_latent(
                self.vision_tokenizer.spatial_compression_factor, x0_tokens_vision, frame_size
            )

        fps_raw = data_batch.get("conditioning_fps", None)
        if isinstance(fps_raw, list):
            fps_raw = torch.stack(fps_raw).flatten()
        fps_vision = fps_raw.to(device=device, dtype=dtype) if fps_raw is not None else None

        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=is_img,
            raw_state_vision=raw_state_vision,
            x0_tokens_vision=x0_tokens_vision,
            fps_vision=fps_vision,
            num_vision_items_per_sample=num_vision_items_per_sample,
        )

    def get_inference_text_tokens(
        self, use_system_prompt: bool, input_caption_key: str, data_batch: dict, has_negative_prompt: bool
    ) -> tuple[list[list[int]], list[list[int]]]:
        cond_tokens = [
            self.tokenize_caption(c, is_video=False, use_system_prompt=use_system_prompt)
            for c in data_batch[input_caption_key]
        ]
        if has_negative_prompt:
            neg_key = "neg_" + input_caption_key
            assert neg_key in data_batch, f"Negative prompt ({neg_key}) not found"
            uncond_captions = data_batch[neg_key]
        else:
            uncond_captions = [""] * len(cond_tokens)
        uncond_tokens = [
            self.tokenize_caption(c, is_video=False, use_system_prompt=use_system_prompt) for c in uncond_captions
        ]
        return cond_tokens, uncond_tokens

    def derive_include_end_of_generation_token(self, joint_attn_implementation: str) -> bool:
        assert joint_attn_implementation in ("flex", "two_way", "three_way")
        return joint_attn_implementation == "flex"

    def prepare_inference_data(
        self,
        use_system_prompt: bool,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        condition_frame_indexes: Optional[List[int]] = None,
        noises: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        input_caption_key: str = "ai_caption",
        input_video_key: str = "video",
        input_image_key: str = "images",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_sound: bool = False,
    ) -> tuple[
        list[SequencePlan],
        GenerationDataClean,
        list[list[int]],
        list[list[int]],
        torch.Tensor,
    ]:
        if enable_sound:
            self._check_sound_enabled()

        # Build data_batch
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompts)
        is_image = num_frames == 1

        conditioning_frames = None
        if image is not None:
            conditioning_frames = self._load_image_as_tensor(image, height, width)

        image_size = [
            torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)
            for _ in range(batch_size)
        ]

        if is_image:
            img_tensor = (
                conditioning_frames.unsqueeze(0).to(device=device, dtype=dtype)
                if conditioning_frames is not None
                else torch.zeros(1, 3, 1, height, width, dtype=dtype, device=device)
            )
            seq_plans = [
                SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])
                for _ in range(batch_size)
            ]
            data_batch = {
                input_image_key: [img_tensor] * batch_size,
                "image_size": image_size,
                "is_preprocessed": True,
                "fps": torch.full((batch_size,), float(fps), device=device),
                "conditioning_fps": torch.full((batch_size,), float(fps), device=device),
                "num_frames": torch.full((batch_size,), num_frames, device=device),
                "sequence_plan": seq_plans,
                input_caption_key: prompts,
            }
        else:
            cond_indexes = (
                condition_frame_indexes
                if condition_frame_indexes is not None
                else ([0] if conditioning_frames is not None else [])
            )
            if conditioning_frames is not None:
                video_data = torch.zeros(1, 3, num_frames, height, width, dtype=dtype)
                t_fill = min(conditioning_frames.shape[1], num_frames)
                video_data[0, :, :t_fill] = conditioning_frames[:, :t_fill].to(dtype=dtype)
                if t_fill < num_frames:
                    video_data[0, :, t_fill:] = video_data[0, :, t_fill - 1 : t_fill].expand(
                        -1, num_frames - t_fill, -1, -1
                    )
                video_tensor = video_data.to(device=device)
            else:
                video_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            seq_plans = [
                SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(cond_indexes))
                for _ in range(batch_size)
            ]
            data_batch = {
                input_video_key: [video_tensor] * batch_size,
                "image_size": image_size,
                "is_preprocessed": True,
                "fps": torch.full((batch_size,), float(fps), device=device),
                "conditioning_fps": torch.full((batch_size,), float(fps), device=device),
                "num_frames": torch.full((batch_size,), num_frames, device=device),
                "sequence_plan": seq_plans,
                input_caption_key: prompts,
            }

        has_negative_prompt = negative_prompt is not None
        if has_negative_prompt:
            neg_prompts = [negative_prompt] if isinstance(negative_prompt, str) else list(negative_prompt)
            data_batch["neg_" + input_caption_key] = neg_prompts

        # --- Inject sound into seq_plans and data_batch (before build_sequence_plans_from_data_batch) ---
        if enable_sound:
            sound_dim = self.transformer.config.sound_dim
            sound_latent_fps = float(self.transformer.config.sound_latent_fps)
            n_audio_samples = int(num_frames / fps * self.sound_tokenizer.sample_rate)
            hop_size = self.sound_tokenizer._hop_size
            T_sound = (n_audio_samples + hop_size - 1) // hop_size
            x0_tokens_sound = [torch.zeros(sound_dim, T_sound, device=device, dtype=dtype) for _ in range(batch_size)]
            fps_sound = torch.tensor([sound_latent_fps] * batch_size, device=device, dtype=dtype)
            # Upgrade each SequencePlan to include sound
            for sp in seq_plans:
                sp.has_sound = True
                sp.condition_frame_indexes_sound = []
            data_batch["sequence_plan"] = seq_plans

        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=input_video_key,
            input_image_key=input_image_key,
        )
        gen_data_clean = self.get_data_and_condition(
            input_image_key, input_video_key, data_batch, device=device, dtype=dtype
        )

        # Attach sound fields to gen_data_clean after construction
        if enable_sound:
            gen_data_clean.x0_tokens_sound = x0_tokens_sound
            gen_data_clean.fps_sound = fps_sound

        cond_text_tokens, uncond_text_tokens = self.get_inference_text_tokens(
            use_system_prompt, input_caption_key, data_batch, has_negative_prompt
        )

        mask_timesteps = torch.zeros((gen_data_clean.batch_size,), dtype=torch.float32)
        packed_seq = pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=cond_text_tokens,
            gen_data_clean=gen_data_clean,
            input_timesteps=mask_timesteps,
            special_tokens=self.llm_special_tokens,
            latent_patch_size=self.transformer.config.latent_patch_size,
            include_end_of_generation_token=self.derive_include_end_of_generation_token(
                self.transformer.config.joint_attn_implementation
            ),
            position_embedding_type=self.transformer.config.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.transformer.config.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.transformer.config.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.transformer.config.enable_fps_modulation,
            base_fps=float(self.transformer.config.base_fps),
            temporal_compression_factor=self.vision_tokenizer.temporal_compression_factor,
            video_temporal_causal=self.transformer.config.video_temporal_causal,
            action_dim=self.transformer.config.max_action_dim,
        )

        assert packed_seq.vision is not None
        assert packed_seq.vision.condition_mask is not None
        assert isinstance(packed_seq.vision.condition_mask, list)
        assert gen_data_clean.x0_tokens_vision is not None

        noise_vision_list: list[torch.Tensor] = []
        for i, (x0_token, cond_mask) in enumerate(
            zip(gen_data_clean.x0_tokens_vision, packed_seq.vision.condition_mask, strict=True)
        ):
            if noises is not None:
                pure_noise = noises[i].to(device=device, dtype=dtype)
            else:
                pure_noise = randn_tensor(tuple(x0_token.shape), generator=generator, device=device, dtype=dtype)
            noise_vision_list.append(
                cond_mask * x0_token.to(device=device, dtype=dtype) + (1.0 - cond_mask) * pure_noise
            )

        initial_noise = torch.cat([t.reshape(-1) for t in noise_vision_list])

        # Append sound noise (all noisy: cond_mask = 0 everywhere)
        if enable_sound and packed_seq.sound is not None:
            assert isinstance(packed_seq.sound.condition_mask, list)
            for x0_sound, cond_mask_sound in zip(x0_tokens_sound, packed_seq.sound.condition_mask):
                pure_noise_sound = randn_tensor(tuple(x0_sound.shape), generator=generator, device=device, dtype=dtype)
                noise_sound = cond_mask_sound.T * x0_sound + (1.0 - cond_mask_sound.T) * pure_noise_sound
                initial_noise = torch.cat([initial_noise, noise_sound.reshape(-1)])

        return sequence_plans, gen_data_clean, cond_text_tokens, uncond_text_tokens, initial_noise

    def encode_text(
        self,
        hidden_size: int,
        packed_seq,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens. Returns (hidden_states [N_total, H], target_dtype)."""
        packed_text_embedding = self.transformer.model.embed_tokens(packed_seq.text_ids)
        hidden_states = packed_text_embedding.new_zeros(size=(packed_seq.sequence_length, hidden_size))
        hidden_states[packed_seq.text_indexes] = packed_text_embedding
        return hidden_states, packed_text_embedding.dtype

    def encode_vision(
        self,
        timestep_scale: float,
        latent_patch_size: int,
        latent_channel: int,
        packed_seq,
        hidden_states: torch.Tensor,
        target_dtype: torch.dtype,
        fps: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, int, int]] | None:
        """Project vision tokens into hidden_states in-place. Returns original_latent_shapes."""
        if packed_seq.vision is None or packed_seq.vision.tokens is None:
            return None
        vision = packed_seq.vision
        assert vision.tokens is not None
        assert vision.token_shapes is not None
        assert isinstance(vision.sequence_indexes, torch.Tensor)
        assert isinstance(vision.timesteps, torch.Tensor)
        assert isinstance(vision.mse_loss_indexes, torch.Tensor)

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            latent_patch_size, latent_channel, vision.tokens, vision.token_shapes
        )
        packed_tokens_vision = self.transformer.vae2llm(packed_tokens_vision)

        if vision.mse_loss_indexes.numel() > 0:
            timesteps_vision = vision.timesteps * timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_vision = self.transformer.time_embedder(timesteps_vision)
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
            packed_tokens_vision = self.apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )

        hidden_states[vision.sequence_indexes] = packed_tokens_vision
        return original_latent_shapes

    @torch.no_grad()
    def run_single(
        self,
        packed_seq,
        noise_x_vision: list[torch.Tensor],
        hidden_size: int,
        latent_patch_size: int,
        latent_channel: int,
        patch_latent_dim: int,
        timestep_scale: float,
        use_moe: bool,
        fps_vision: Optional[torch.Tensor],
        noise_x_sound: Optional[list] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple:
        """Inlined forward pass from Cosmos3VFMNetworkSimple.forward().

        Returns:
            ``(preds_vision_list, preds_sound_list)`` — sound list is ``None``
            when ``noise_x_sound`` is ``None``.
        """
        if packed_seq.vision is not None:
            packed_seq.vision.tokens = [x.to(device=device, dtype=dtype) for x in noise_x_vision]

        # Set sound tokens before packing
        if noise_x_sound is not None and packed_seq.sound is not None:
            packed_seq.sound.tokens = [x.to(device=device, dtype=dtype) for x in noise_x_sound]

        packed_seq.to_cuda()

        # 1. Encode text
        hidden_states, target_dtype = self.encode_text(hidden_size, packed_seq)

        # 2. Encode vision
        original_latent_shapes = self.encode_vision(
            timestep_scale,
            latent_patch_size,
            latent_channel,
            packed_seq,
            hidden_states,
            target_dtype,
            fps=fps_vision,
        )

        # 3. Encode sound (in-place into hidden_states)
        if noise_x_sound is not None:
            self.encode_sound_tokens(timestep_scale, packed_seq, hidden_states, target_dtype)

        # 4. Build attention metadata
        assert use_moe
        input_pack, attention_meta = build_packed_sequence(
            packed_sequence=hidden_states,
            attn_modes=packed_seq.attn_modes,
            split_lens=packed_seq.split_lens,
            sample_lens=packed_seq.sample_lens,
        )

        # 5. Run transformer
        packed_outputs, _ = self.transformer(
            input_pack,
            attention_mask=attention_meta,
            position_ids=packed_seq.position_ids,
            dual_kv_cache=None,
            frame_idx=None,
            natten_metadata_list=None,
        )
        last_hidden_state = get_all_seq(packed_outputs)

        # 6. Decode vision
        preds_vision = self.decode_vision(
            patch_latent_dim,
            latent_patch_size,
            latent_channel,
            packed_seq,
            last_hidden_state,
            original_latent_shapes,
        )

        # 7. Decode sound
        preds_sound = None
        if noise_x_sound is not None:
            preds_sound = self.decode_sound_tokens(packed_seq, last_hidden_state)

        return preds_vision, preds_sound

    @torch.no_grad()
    def get_cfg_velocity(
        self,
        noise_x: torch.Tensor,
        timestep: torch.Tensor,
        guidance: float,
        gen_data_clean: GenerationDataClean,
        sequence_plans: list[SequencePlan],
        cond_tokens: list[list[int]],
        uncond_tokens: list[list[int]],
        include_eog: bool,
        hidden_size: int,
        latent_patch_size: int,
        latent_channel: int,
        patch_latent_dim: int,
        timestep_scale: float,
        use_moe: bool,
        skip_text_tokens_for_cfg: bool = False,
        normalize_cfg: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        assert timestep.ndim == 2 and timestep.shape == (1, 1)

        has_sound = gen_data_clean.x0_tokens_sound is not None and any(sp.has_sound for sp in sequence_plans)

        num_items = gen_data_clean.num_vision_items_per_sample
        num_vis = num_items[0] if num_items is not None else 1

        # Split flat noise_x → vision part
        noise_x_vision: list[torch.Tensor] = []
        offset = 0
        for j in range(num_vis):
            vision_shape = gen_data_clean.x0_tokens_vision[j].shape
            vision_dim = int(torch.prod(torch.tensor(vision_shape)))
            noise_x_vision.append(noise_x[offset : offset + vision_dim].reshape(vision_shape))
            offset += vision_dim

        # Split flat noise_x → sound part
        noise_x_sound: Optional[list] = None
        if has_sound:
            noise_x_sound = []
            assert gen_data_clean.x0_tokens_sound is not None
            sound_shape = gen_data_clean.x0_tokens_sound[0].shape
            sound_dim_flat = int(torch.prod(torch.tensor(sound_shape)))
            noise_x_sound.append(noise_x[offset : offset + sound_dim_flat].reshape(sound_shape))

        gen_data_for_packing = GenerationDataClean(
            batch_size=1,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=noise_x_vision,
            fps_vision=gen_data_clean.fps_vision,
            num_vision_items_per_sample=num_items,
            x0_tokens_sound=noise_x_sound,
            fps_sound=gen_data_clean.fps_sound if has_sound else None,
        )

        def _run_cond(text_tokens: list[list[int]], skip_text: bool) -> torch.Tensor:
            packed_seq = pack_input_sequence(
                sequence_plans=sequence_plans,
                input_text_indexes=text_tokens,
                gen_data_clean=gen_data_for_packing,
                input_timesteps=timestep.cpu(),
                special_tokens=self.llm_special_tokens,
                latent_patch_size=self.transformer.config.latent_patch_size,
                include_end_of_generation_token=include_eog,
                skip_text_tokens=skip_text,
                position_embedding_type=self.transformer.config.position_embedding_type,
                unified_3d_mrope_reset_spatial_ids=self.transformer.config.unified_3d_mrope_reset_spatial_ids,
                unified_3d_mrope_temporal_modality_margin=self.transformer.config.unified_3d_mrope_temporal_modality_margin,
                enable_fps_modulation=self.transformer.config.enable_fps_modulation,
                base_fps=float(self.transformer.config.base_fps),
                temporal_compression_factor=self.vision_tokenizer.temporal_compression_factor,
                video_temporal_causal=self.transformer.config.video_temporal_causal,
                action_dim=self.transformer.config.max_action_dim,
            )
            preds_vision, preds_sound = self.run_single(
                packed_seq,
                noise_x_vision,
                hidden_size,
                latent_patch_size,
                latent_channel,
                patch_latent_dim,
                timestep_scale,
                use_moe,
                fps_vision=gen_data_clean.fps_vision,
                noise_x_sound=noise_x_sound,
                device=device,
                dtype=dtype,
            )

            assert packed_seq.vision is not None
            assert packed_seq.vision.condition_mask is not None
            assert isinstance(packed_seq.vision.condition_mask, list)
            velocity_vision = [
                pred * (1.0 - m).to(dtype=pred.dtype, device=pred.device)
                if (1.0 - m).sum() > 0
                else torch.zeros_like(pred)
                for pred, m in zip(preds_vision, packed_seq.vision.condition_mask)
            ]
            parts = [v.reshape(-1) for v in velocity_vision]

            if preds_sound is not None and packed_seq.sound is not None:
                assert isinstance(packed_seq.sound.condition_mask, list)
                for pred_s, cond_mask_s in zip(preds_sound, packed_seq.sound.condition_mask):
                    noisy_mask_s = (1.0 - cond_mask_s).T.to(dtype=pred_s.dtype, device=pred_s.device)
                    v_sound = pred_s * noisy_mask_s if noisy_mask_s.sum() > 0 else torch.zeros_like(pred_s)
                    parts.append(v_sound.reshape(-1))

            return torch.cat(parts)

        cond_v = _run_cond(cond_tokens, False)
        uncond_v = _run_cond(uncond_tokens, skip_text_tokens_for_cfg)

        v_pred = uncond_v + guidance * (cond_v - uncond_v)

        if normalize_cfg:
            v_pred = v_pred * (torch.norm(cond_v) / (torch.norm(v_pred) + 1e-8)).clamp(min=0.0, max=1.0)
        return v_pred

    def _load_image_as_tensor(self, image, target_h: int, target_w: int) -> torch.Tensor:
        """Load image from PIL, path, URL, or tensor; returns [3, 1, H, W] in [-1, 1]."""
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image_str = str(image)
            if image_str.startswith("http://") or image_str.startswith("https://"):
                import io
                import urllib.request

                with urllib.request.urlopen(image_str) as resp:
                    img_bytes = resp.read()
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                with open(image_str, "rb") as f:
                    pil_img = PILImage.open(f).convert("RGB")
            img_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()
        elif hasattr(image, "convert"):  # PIL.Image
            img_t = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float()
        elif isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.dim() == 4:
                img_t = img_t.squeeze(0)
            # if already normalized to [-1, 1], skip the /127.5-1 step below
            if img_t.max() <= 1.1:
                img_4d = img_t.unsqueeze(0)
                orig_h, orig_w = img_4d.shape[2], img_4d.shape[3]
                scale = max(target_w / orig_w, target_h / orig_h)
                resize_h = int(math.ceil(scale * orig_h))
                resize_w = int(math.ceil(scale * orig_w))
                img_4d = TF.resize(img_4d, [resize_h, resize_w])
                img_4d = TF.center_crop(img_4d, [target_h, target_w])
                return img_4d.squeeze(0).unsqueeze(1)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        img_4d = img_t.unsqueeze(0)  # [1, 3, H, W]  (uint8-range [0, 255])
        orig_h, orig_w = img_4d.shape[2], img_4d.shape[3]
        scale = max(target_w / orig_w, target_h / orig_h)
        resize_h = int(math.ceil(scale * orig_h))
        resize_w = int(math.ceil(scale * orig_w))
        img_4d = TF.resize(img_4d, [resize_h, resize_w])
        img_4d = TF.center_crop(img_4d, [target_h, target_w])
        img_4d = img_4d / 127.5 - 1.0  # normalize after resize, matching load_conditioning_image
        return img_4d.squeeze(0).unsqueeze(1)  # [3, 1, H, W]

    def _resolve_defaults_and_prompts(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]],
        image,
        num_frames: int,
        fps: float,
        height: int,
        width: int,
    ) -> tuple[float, int, float, Union[str, List[str]], Union[str, List[str]]]:
        """Load modality defaults and apply duration/resolution templates to prompts.

        Returns (guidance, num_steps, shift, formatted_prompt, formatted_negative_prompt).
        """
        if image is not None:
            model_mode = "image2video"
        elif num_frames == 1:
            model_mode = "text2image"
        else:
            model_mode = "text2video"
        # Defaults live under examples/cosmos3/sample_args/ in the source checkout
        # (parents[4] = repo root from this file).
        defaults_path = Path(__file__).parents[4] / "examples" / "cosmos3" / "sample_args" / f"{model_mode}.json"
        defaults = json.loads(defaults_path.read_text())

        guidance = float(defaults["guidance"])
        num_steps = int(defaults["num_steps"])
        shift = float(defaults["shift"])
        print(f"model_mode={model_mode!r}: guidance={guidance}, num_steps={num_steps}, shift={shift}")

        duration_template = defaults.get("duration_template")
        resolution_template = defaults.get("resolution_template")
        negative_prompt_base = defaults.get("negative_prompt", "")
        keep_metadata = defaults.get("negative_prompt_keep_metadata", False)

        def _apply_templates(text: str) -> str:
            if duration_template and num_frames > 1:
                text = text.rstrip(".") + ". " + duration_template.format(duration=num_frames / fps, fps=fps)
            if resolution_template:
                text = text.rstrip(".") + ". " + resolution_template.format(height=height, width=width)
            return text

        if isinstance(prompt, str):
            prompt = _apply_templates(prompt)
        else:
            prompt = [_apply_templates(p) for p in prompt]

        if negative_prompt is None:
            negative_prompt = _apply_templates(negative_prompt_base) if keep_metadata else negative_prompt_base

        return guidance, num_steps, shift, prompt, negative_prompt

    @torch.no_grad()
    def decode_latents(self, vision_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode latents to pixel tensors of shape [C, T, H, W] in [0, 1]."""
        frames = []
        for vision_latent in vision_list:
            vision = self.vision_tokenizer.decode(vision_latent.cuda())  # [1, C, T, H, W]
            frames.append(((1.0 + vision) / 2).clamp(0, 1).squeeze(0))
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image=None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        condition_frame_indexes: Optional[List[int]] = None,
        noises: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        use_system_prompt: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        output_type: str = "video",
        enable_sound: bool = False,
    ) -> Cosmos3OmniPipelineOutput:
        latent_patch_size = self.transformer.config.latent_patch_size
        latent_channel = self.transformer.config.latent_channel
        patch_latent_dim = self.transformer.config.patch_latent_dim
        timestep_scale = self.transformer.config.timestep_scale
        hidden_size = self.transformer.config.hidden_size
        use_moe = self.transformer.config.use_moe
        joint_attn_implementation = self.transformer.config.joint_attn_implementation

        guidance, num_steps, shift, prompt, negative_prompt = self._resolve_defaults_and_prompts(
            prompt, negative_prompt, image, num_frames, fps, height, width
        )

        if enable_sound:
            self._check_sound_enabled()

        sequence_plans, gen_data_clean, cond_tokens, uncond_tokens, initial_noise = self.prepare_inference_data(
            use_system_prompt,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            condition_frame_indexes=condition_frame_indexes,
            noises=noises,
            generator=generator,
            device=device,
            dtype=dtype,
            enable_sound=enable_sound,
        )

        assert guidance != 1.0, "Guidance weight must be != 1.0"

        device = initial_noise.device
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        # print(f"sigmas:    first={self.scheduler.sigmas[0].item():.4f}  last={self.scheduler.sigmas[-2].item():.4f}")
        # print(f"timesteps: first={timesteps[0].item():.2f}  last={timesteps[-1].item():.2f}")
        # print(f"timestep_scale: {timestep_scale}")
        # breakpoint()
        latent = initial_noise
        include_eog = self.derive_include_end_of_generation_token(joint_attn_implementation)

        # --- Denoising loop ---
        print("Running generate_samples_from_batch …")
        for timestep in tqdm(timesteps, desc="Denoising"):
            velocity_pred = self.get_cfg_velocity(
                latent,
                timestep.reshape(1, 1),
                guidance,
                gen_data_clean,
                sequence_plans,
                cond_tokens,
                uncond_tokens,
                include_eog,
                hidden_size,
                latent_patch_size,
                latent_channel,
                patch_latent_dim,
                timestep_scale,
                use_moe,
                device=device,
                dtype=dtype,
            )
            latent = self.scheduler.step(
                model_output=velocity_pred,
                timestep=timestep,
                sample=latent.unsqueeze(0),
                return_dict=False,
            )[0].squeeze(0)

        # --- Extract vision results ---
        num_vision_items = gen_data_clean.num_vision_items_per_sample
        n_vis = num_vision_items[0] if num_vision_items is not None else 1
        result_vision: list[torch.Tensor] = []
        offset = 0
        for j in range(n_vis):
            vision_shape = gen_data_clean.x0_tokens_vision[j].shape
            vision_dim = int(torch.prod(torch.tensor(vision_shape)))
            if j == n_vis - 1:
                result_vision.append(latent[offset : offset + vision_dim].reshape(vision_shape))
            offset += vision_dim

        # --- Extract and decode sound result ---
        result_sound: Optional[list] = None
        if enable_sound and gen_data_clean.x0_tokens_sound is not None:
            sound_shape = gen_data_clean.x0_tokens_sound[0].shape  # [sound_dim, T_sound]
            sound_dim_flat = int(torch.prod(torch.tensor(sound_shape)))
            sound_latent = latent[offset : offset + sound_dim_flat].reshape(sound_shape)
            result_sound = [self.decode_sound(sound_latent)]

        if output_type == "latent":
            return Cosmos3OmniPipelineOutput(video=result_vision, sound=result_sound)

        frames = self.decode_latents(result_vision)
        return Cosmos3OmniPipelineOutput(video=frames, sound=result_sound)
