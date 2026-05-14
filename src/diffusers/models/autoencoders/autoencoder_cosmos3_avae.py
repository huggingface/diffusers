# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos3 AVAE Audio Tokenizer — decoder-only implementation.

Supports two latent normalisation modes controlled by ``normalization_type``:

- ``"tanh"``: latents were tanh-normalised during
  encoding; decode applies the inverse ``atanh`` before the Oobleck decoder.
  Constants: ``tanh_input_scale=1.5``, ``tanh_output_scale=3.5``, ``tanh_clamp=0.995``.

- ``"none"``: no latent normalisation; latents are passed directly to the decoder.
"""

import math

import torch

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from .autoencoder_oobleck import OobleckDecoder


class Cosmos3AVAEAudioTokenizer(ModelMixin, ConfigMixin):
    """Decoder-only audio tokenizer for Cosmos3 sound generation.

    Wraps the Oobleck decoder used in the AVAE (Audio VAE) component of the Cosmos3
    omni model.  Provides the interface expected by ``Cosmos3OmniDiffusersPipeline``
    when ``enable_sound=True``.

    Parameters:
        sampling_rate (`int`, defaults to `48000`): Audio sample rate in Hz.
        enc_latent_dim (`int`, defaults to `128`): Latent channels (``sound_dim``).
        dec_dim (`int`, defaults to `320`): Base decoder channel count.
        dec_c_mults (`list[int]`, defaults to `[1, 2, 4, 8, 16]`): Channel multipliers.
        dec_strides (`list[int]`, defaults to `[2, 4, 5, 6, 8]`): Upsampling strides.
        dec_out_channels (`int`, defaults to `2`): Output audio channels (2 = stereo).
        normalization_type (`str`, defaults to `"none"`): Latent normalisation mode.
            ``"tanh"`` applies inverse-tanh before decoding; ``"none"`` skips it.
        normalize_latents (`bool`, defaults to `True`): Legacy flag. When
            ``normalization_type="none"`` and this is ``True``, promotes the type to
            ``"tanh"``
        tanh_input_scale (`float`, defaults to `1.5`): Scale applied before ``atanh``
            during latent denormalisation (only used when ``normalization_type="tanh"``).
        tanh_output_scale (`float`, defaults to `3.5`): Scale applied after encoding
            tanh (only used when ``normalization_type="tanh"``).
        tanh_clamp (`float`, defaults to `0.995`): Clamp limit for ``atanh`` stability
            (only used when ``normalization_type="tanh"``).
    """

    _supports_gradient_checkpointing = False
    _supports_group_offloading = False

    @register_to_config
    def __init__(
        self,
        sampling_rate: int = 48000,
        enc_latent_dim: int = 128,
        vocoder_input_dim: int = 64,
        dec_dim: int = 320,
        dec_c_mults: list = None,
        dec_strides: list = None,
        dec_out_channels: int = 2,
        normalization_type: str = "none",
        normalize_latents: bool = False,
        tanh_input_scale: float = 1.5,
        tanh_output_scale: float = 3.5,
        tanh_clamp: float = 0.995,
        **kwargs,
    ):
        super().__init__()

        if dec_c_mults is None:
            dec_c_mults = [1, 2, 4, 8, 16]
        if dec_strides is None:
            dec_strides = [2, 4, 5, 6, 8]

        if normalization_type == "none" and normalize_latents:
            normalization_type = "tanh"
        self._normalization_type = normalization_type

        self.decoder = OobleckDecoder(
            channels=dec_dim,
            input_channels=vocoder_input_dim,
            audio_channels=dec_out_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=dec_c_mults,
        )

        self._hop_size: int = math.prod(dec_strides)

    # ------------------------------------------------------------------
    # Interface expected by Cosmos3OmniDiffusersPipeline
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self.config.sampling_rate

    @property
    def latent_ch(self) -> int:
        """Number of latent channels (== transformer config ``sound_dim`` == vocoder_input_dim)."""
        return self.config.vocoder_input_dim

    def get_latent_num_samples(self, n_audio_samples: int) -> int:
        """Return the number of latent frames for ``n_audio_samples`` raw samples."""
        return n_audio_samples // self._hop_size

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Invert the tanh normalisation applied during AVAE encoding.

        z = atanh(clamp(latent / tanh_output_scale, -tanh_clamp, tanh_clamp))
            * tanh_input_scale
        """
        in_dtype = latent.dtype
        z = torch.clamp(
            latent.float() / self.config.tanh_output_scale,
            -self.config.tanh_clamp,
            self.config.tanh_clamp,
        )
        return (torch.atanh(z) * self.config.tanh_input_scale).to(in_dtype)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sound latents into an audio waveform.

        Applies ``_denormalize_latent`` (inverse tanh) before the Oobleck decoder
        only when ``normalization_type="tanh"``.  When ``normalization_type="none"``
        (no latent normalisation) latents are passed directly.

        Args:
            latents: ``[B, C, T]`` or ``[C, T]`` tensor of diffusion-model latents.

        Returns:
            Waveform tensor ``[B, audio_channels, N]`` or ``[audio_channels, N]``.
        """
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        z = self._denormalize_latent(latents) if self._normalization_type == "tanh" else latents
        audio = self.decoder(z)
        audio = audio.clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio

    # ------------------------------------------------------------------
    # Weight loading: handle non-standard filename and silently ignore
    # encoder/bottleneck keys from source checkpoint.
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Override to support an upstream ``model.safetensors`` filename.

        ``ModelMixin.from_pretrained`` hardcodes the weight filename as
        ``diffusion_pytorch_model.safetensors`` and provides no public parameter to
        override it.  When the standard file is absent but ``model.safetensors`` is
        present in a local directory, we create a relative symlink so that diffusers'
        loader finds it transparently.  The symlink persists and speeds up future loads.
        """
        import os

        from diffusers.utils import SAFETENSORS_WEIGHTS_NAME

        model_dir = str(pretrained_model_name_or_path)
        std_sf = os.path.join(model_dir, SAFETENSORS_WEIGHTS_NAME)
        alt_sf = os.path.join(model_dir, "model.safetensors")

        if os.path.isdir(model_dir) and not os.path.exists(std_sf) and os.path.exists(alt_sf):
            # Create a relative symlink so diffusers' loader finds it.
            os.symlink("model.safetensors", std_sf)

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def _remap_checkpoint_keys(state_dict: dict) -> dict:
        """Remap flat ``decoder.layers.N.*`` keys (upstream ``nn.Sequential`` layout)
        to the named-attribute layout of ``OobleckDecoder``.

        Mapping::

            decoder.layers.0.*                          → decoder.conv1.*
            decoder.layers.N.layers.0.*  (N=1..5)       → decoder.block.{N-1}.snake1.*
            decoder.layers.N.layers.1.*  (N=1..5)       → decoder.block.{N-1}.conv_t1.*
            decoder.layers.N.layers.K.layers.0.*        → decoder.block.{N-1}.res_unit{K-1}.snake1.*
            decoder.layers.N.layers.K.layers.1.*        → decoder.block.{N-1}.res_unit{K-1}.conv1.*
            decoder.layers.N.layers.K.layers.2.*        → decoder.block.{N-1}.res_unit{K-1}.snake2.*
            decoder.layers.N.layers.K.layers.3.*        → decoder.block.{N-1}.res_unit{K-1}.conv2.*
            decoder.layers.6.*                          → decoder.snake1.*
            decoder.layers.7.*                          → decoder.conv2.*
        """
        import re

        _RES_SUB = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}

        def _remap(key: str) -> str:
            if not key.startswith("decoder.layers."):
                return key
            suffix = key[len("decoder.") :]  # strip leading "decoder."

            m = re.fullmatch(r"layers\.0\.(.+)", suffix)
            if m:
                return f"decoder.conv1.{m.group(1)}"

            m = re.fullmatch(r"layers\.6\.(.+)", suffix)
            if m:
                return f"decoder.snake1.{m.group(1)}"

            m = re.fullmatch(r"layers\.7\.(.+)", suffix)
            if m:
                return f"decoder.conv2.{m.group(1)}"

            # decoder block: layers.N.layers.M.* (N=1..5)
            m = re.fullmatch(r"layers\.(\d+)\.layers\.(\d+)\.(.+)", suffix)
            if m:
                block_n, sub_m, rest = int(m.group(1)), int(m.group(2)), m.group(3)
                bi = block_n - 1
                if sub_m == 0:
                    return f"decoder.block.{bi}.snake1.{rest}"
                if sub_m == 1:
                    return f"decoder.block.{bi}.conv_t1.{rest}"
                # sub_m in {2,3,4} → res_unit{sub_m-1}
                res_name = f"res_unit{sub_m - 1}"
                mm = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
                if mm:
                    sub_k, sub_rest = int(mm.group(1)), mm.group(2)
                    sub_name = _RES_SUB.get(sub_k, str(sub_k))
                    return f"decoder.block.{bi}.{res_name}.{sub_name}.{sub_rest}"

            return key

        return {_remap(k): v for k, v in state_dict.items()}

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        resolved_archive_file,
        pretrained_model_name_or_path,
        loaded_keys=None,
        **kwargs,
    ):
        # Strip common per-key prefixes produced by DDP ("module."), full-model saves
        # ("model.", "generator.") or upstream training exports. Iterate until keys
        # stabilise so that stacked prefixes (e.g. "module.model.") are fully removed.
        _prefixes = ("module.", "model.", "generator.", "state_dict.")
        if state_dict and not any(k.startswith("decoder.") for k in state_dict):
            changed = True
            while changed:
                changed = False
                for prefix in _prefixes:
                    if any(k.startswith(prefix) for k in state_dict):
                        state_dict = {
                            (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()
                        }
                        changed = True
                        break
                if any(k.startswith("decoder.") for k in state_dict):
                    break

        # Remap flat nn.Sequential key names to our OobleckDecoder attribute names.
        state_dict = cls._remap_checkpoint_keys(state_dict)

        # Reshape Snake1d alpha/beta: checkpoint stores [C], our Snake1d expects [1, C, 1].
        for key, val in state_dict.items():
            if (key.endswith(".alpha") or key.endswith(".beta")) and val.ndim == 1:
                state_dict[key] = val.unsqueeze(0).unsqueeze(-1)

        # Drop encoder and bottleneck keys — not needed for decode-only inference.
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("decoder.")}

        # Keep loaded_keys in sync so diffusers doesn't flag non-decoder keys as missing.
        if loaded_keys is not None:
            remapped_dummy = cls._remap_checkpoint_keys({k: None for k in loaded_keys})
            loaded_keys = [k for k in remapped_dummy if k.startswith("decoder.")]

        return super()._load_pretrained_model(
            model,
            state_dict,
            resolved_archive_file,
            pretrained_model_name_or_path,
            loaded_keys,
            **kwargs,
        )
