import math
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import torchaudio

from transformers.audio_utils import spectrogram, mel_filter_bank, optimal_fft_length, window_function
from transformers import UnivNetFeatureExtractor

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.embeddings import TimestepEmbedding, Timesteps
from ...models.resnet import AdaGroupNorm, Downsample2D, Upsample2D, downsample_2d, partial, upsample_2d
from ...utils import BaseOutput, logging

from .modeling_common import TortoiseTTSAttention, TortoiseTTSSelfAttention, TacotronSTFT

from scipy.signal import get_window
import librosa.util as librosa_util


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




def pad_or_truncate(t, length: int, random_start: bool = False):
    t = torch.from_numpy(t) if isinstance(t, numpy.ndarray) else t

    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length - t.shape[-1]))
    else:
        return t[..., :length]

def check_and_resample(audio, audio_sr, target_sr):
    if audio_sr!=target_sr:
        audio = torchaudio.functional.resample(audio, audio_sr, target_sr)

    return audio

def compute_groupnorm_groups(channels: int, groups: int = 32):
    """
    Calculates the value of `num_groups` for nn.GroupNorm. This logic is taken from the official tortoise repository. link :
    https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/models/arch_util.py#L26
    """
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)

    if groups <= 2:
        raise ValueError(
            f"Number of groups for the GroupNorm must be greater than 2, but it is {groups}."
            f"Please consider using a different `hidden_size`"
        )

    return groups


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


@dataclass
class DiffusionConditioningEncoderOutput(BaseOutput):
    """
    The output of [`DiffusionConditioningEncoder`].

    Args:
        TODO: fix
        embedding (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    embedding: torch.FloatTensor


class DiffusionConditioningEncoder(ModelMixin, ConfigMixin):
    """
    Conditioning encoder for the Tortoise TTS diffusion model.
    """

    @register_to_config
    def __init__(
        self,
        audio_in_channels: int = 100,
        audio_attention_layers: int = 5,
        latent_in_channels: int = 1024,
        latent_attention_layers: int = 4,
        hidden_channels: int = 1024,
        num_attention_heads: int = 16,
        chunk_size: int = 102400  # DURS_CONST in original
    ):
        super().__init__()

        # Class to map audio waveforms to log mel spectrograms
        self.stft = UnivNetFeatureExtractor(
            sampling_rate=24000,
            num_mel_bins=100,
            hop_length=256,
            win_length=1024,
            filter_length=1024,
            fmin=0.0,
            fmax=12000.0,
        )

        # Define the contextual embedder, which maps the audio waveforms into the diffusion audio embedding.
        self.contextual_embedder_conv = nn.Sequential(
            nn.Conv1d(audio_in_channels, hidden_channels, 3, padding=1, stride=2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, 3, padding=1, stride=2),
        )
        self.contextual_embedder_attention = nn.Sequential(*[
            TortoiseTTSSelfAttention(
                hidden_channels * 2,
                n_heads=num_attention_heads,
                dim_head=(hidden_channels * 2) // num_attention_heads,
                relative_attention_num_buckets=32,
                relative_attention_max_distance=64,
            )
            for _ in range(audio_attention_layers)
        ])

        # Define the latent conditioner, which maps diffusion audio embeddings and autoregressive latents to the final
        # diffusion conditioning embedding.
        self.latent_conditioner_conv = nn.Conv1d(latent_in_channels, hidden_channels, 3, padding=1)

        latent_attentions = [
            TortoiseTTSSelfAttention(
                hidden_channels,
                n_heads=num_attention_heads,
                dim_head=hidden_channels // num_attention_heads,
            )
            for _ in range(latent_attention_layers)
        ]
        self.latent_conditioner_attn_blocks = nn.ModuleList(latent_attentions)

        # The unconditional embedding used for Tortoise TTS spectrogram diffusion classifier-free guidance.
        self.unconditional_embedding = nn.Parameter(torch.randn(1, hidden_channels, 1))

    def get_mel_spectrogram(self, audio):
        stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000)
        stft = stft.to("cuda")
        mel_spectrogram = stft.mel_spectrogram(audio)

        return mel_spectrogram

    def convert_and_average_audio_samples(
        self,
        audio,
        latent_averaging_mode: int = 0,
        chunk_size: Optional[int] = None,
    ):
        audio = [audio] if not isinstance(audio[0], list) else audio

        chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size
        audio_spectrograms = []
        for audio_sample in audio:
            if latent_averaging_mode == 0:
                # Average across all samples (original Tortoise TTS behavior)
                audio_sample = pad_or_truncate(audio_sample, chunk_size).to("cuda")
                # spectrogram = self.get_mel_spectrogram(audio_sample[None])
                spectrogram = torch.load("/home/susnato/PycharmProjects/tortoise/check/mel_spec.pth") # use this until the Feature Extractor problem is solved.
                audio_spectrograms.append(spectrogram)
            else:
                if latent_averaging_mode == 2:
                    sample_audio_spectrograms = []
                for chunk in range(math.ceil(audio_sample.shape[1] / chunk_size)):
                    current_chunk = audio_sample[:, chunk * chunk_size : (chunk + 1) * chunk_size]
                    current_chunk = pad_or_truncate(current_chunk, chunk_size).to("cuda")
                    # chunk_spectrogram = self.get_mel_spectrogram(current_chunk[None])
                    chunk_spectrogram = torch.load(
                        "/home/susnato/PycharmProjects/tortoise/check/mel_spec.pth")  # use this until the Feature Extractor problem is solved.

                    if latent_averaging_mode == 1:
                        # Average across all chunks of all samples
                        audio_spectrograms.append(chunk_spectrogram)
                    elif latent_averaging_mode == 2:
                        # Double average: average across all chunks for each sample, then average among all samples
                        sample_audio_spectrograms.append(chunk_spectrogram)
                if latent_averaging_mode == 2:
                    averaged_sample_spectrogram = torch.stack(sample_audio_spectrograms).mean(0)
                    audio_spectrograms.append(averaged_sample_spectrogram)
        audio_spectrograms = torch.stack(audio_spectrograms, dim=1).to("cuda" if torch.cuda.is_available() else "cpu")
        audio_spectrograms = audio_spectrograms.type(torch.float32)
        return audio_spectrograms

    def diffusion_cond_audio_embedding(self, audio, audio_sr, target_sr, latent_averaging_mode: int = 0, chunk_size: Optional[int] = None):
        # the diffusion model expects the audio to be at 24 kHz so resample it.
        audio = check_and_resample(torch.from_numpy(audio), audio_sr, target_sr)
        audio_spectrograms = self.convert_and_average_audio_samples(audio, latent_averaging_mode, chunk_size)
        audio_spectrograms = audio_spectrograms[0, ...]

        audio_embedding = self.contextual_embedder_conv(audio_spectrograms)
        audio_embedding = self.contextual_embedder_attention(audio_embedding.transpose(1, 2))

        audio_embedding = audio_embedding.mean(dim=1)
        return audio_embedding

    def diffusion_cond_embedding(
        self,
        audio_embedding,
        autoregressive_latents,
        attention_mask,
        unconditional: bool = False,
        batch_size: int = 1,
        target_size: Optional[int] = None,
    ):
        if unconditional:
            cond_embedding = self.unconditional_embedding
            if target_size is not None:
                cond_embedding = cond_embedding.repeat(batch_size, 1, target_size)
        else:
            cond_scale, cond_shift = torch.chunk(audio_embedding, 2, dim=1)

            # apply the conv layer first and carefully handle the attention mask
            if attention_mask is not None:
                autoregressive_latents = torch.masked_fill(autoregressive_latents, attention_mask[..., None].bool().logical_not(), 0.0)

            autoregressive_latents = autoregressive_latents[:, 0, :, :]
            cond_embedding = self.latent_conditioner_conv(autoregressive_latents.transpose(1, 2))
            cond_embedding = cond_embedding.transpose(1, 2)

            # then apply the attention layers.
            # Note that because the previous conv layer had kernel_size=3 and padding=1, we don't need to change the
            # attention_mask to make sure the shapes match.
            for attn_block in self.latent_conditioner_attn_blocks:
                cond_embedding = attn_block(cond_embedding, attention_mask)

            cond_embedding = cond_embedding.transpose(1, 2)
            cond_embedding = (1 + cond_scale.unsqueeze(-1)) * cond_embedding + cond_shift.unsqueeze(-1)
            if target_size is not None:
                cond_embedding = F.interpolate(cond_embedding, size=target_size, mode="nearest")
        return cond_embedding

    def forward(
        self,
        audio,
        autoregressive_latents,
        attention_mask = None,
        latent_averaging_mode: int = 0,
        chunk_size: Optional[int] = None,
        unconditional: bool = False,
        batch_size: int = 1,
        target_size: Optional[int] = None,
        return_dict: bool = True,
    ):
        diffusion_audio_embedding = self.diffusion_cond_audio_embedding(audio, latent_averaging_mode, chunk_size)
        diffusion_embedding = self.diffusion_cond_embedding(
            diffusion_audio_embedding, autoregressive_latents, attention_mask, unconditional, batch_size, target_size
        )

        if not return_dict:
            output = (diffusion_embedding,)
            return output

        return DiffusionConditioningEncoderOutput(embedding=diffusion_embedding)


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-5,
        non_linearity="silu",
        time_embedding_norm="scale_shift",  # default, scale_shift, ada_group
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        # changing the Conv2d to Conv1d
        # changing kernel_size=1 from 3 and padding=0
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        # changing the Conv2d to Conv1d
        self.conv2 = torch.nn.Conv1d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            # changing the Conv2d to Conv1d
            self.conv_shortcut = torch.nn.Conv1d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            # change this line too since we are now dealing with Conv1d
            # temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class UNetBlock1D(nn.Module):
    """
    1D U-Net style block with architecture (no down/upsampling)

    ResnetBlock1d => TortoiseTTSAttention
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        num_heads: int = 16,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        relative_pos_embeddings: bool = True,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                TortoiseTTSAttention(
                    query_dim=in_channels,
                    n_heads=num_heads,
                    dim_head=in_channels//num_heads,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, upsample_size=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


@dataclass
class TortoiseTTSDenoisingModelOutput(BaseOutput):
    """
    The output of [`TortoiseTTSDenoisingModel`].

    Args:
        TODO: fix
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class TortoiseTTSDenoisingModel(ModelMixin, ConfigMixin):
    """
    The denoising model used in the diffusion portion of the Tortoise TTS model.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 100,
        out_channels: int = 200,
        hidden_channels: int = 1024,
        num_layers: int = 10,
        num_timestep_integrator_layers: int = 3,
        num_post_res_blocks: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        num_heads=16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # TODO: make sure all the blocks are initialized the same way as original code

        # 1. Define conditioning timestep integrator, which combines the diffusion conditioning embedding from the
        # audio samples and autoregressive model with the timestep embedding
        self.conditioning_timestep_integrator = UNetBlock1D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            temb_channels=hidden_channels,
            dropout=dropout,
            num_layers=num_timestep_integrator_layers,
            num_heads=num_heads,
            resnet_time_scale_shift="scale_shift",
            resnet_act_fn="silu",
        )

        # 2. Define the timestep embedding.
        self.time_proj = Timesteps(hidden_channels, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift)
        self.time_embedding = TimestepEmbedding(in_channels=hidden_channels, time_embed_dim=hidden_channels)

        # 3. Define the inital Conv1d layers
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv_add_cond_emb_to_hidden = nn.Conv1d(2 * hidden_channels, hidden_channels, 1)

        # 4. Define the trunk of the denoising model
        self.blocks = UNetBlock1D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            temb_channels=hidden_channels,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            resnet_time_scale_shift="scale_shift",
            resnet_act_fn="silu",
        )

        self.post_res_blocks = nn.ModuleList(
            [
                ResnetBlock1D(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    temb_channels=hidden_channels,
                    dropout=dropout,
                    time_embedding_norm="scale_shift",
                )
                for _ in range(num_post_res_blocks)
            ]
        )

        # 7. Define the output layers
        self.norm_out = nn.GroupNorm(compute_groupnorm_groups(hidden_channels), hidden_channels)
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning_embedding: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        TODO
        """
        # 1. Handle timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. Combine conditioning embedding with timestep embedding
        conditioning_embedding = self.conditioning_timestep_integrator(conditioning_embedding, temb=emb)[0]

        # 3. Map inital sample to hidden states
        hidden_states = self.conv_in(sample)

        # 4. Concatenate initial hidden states with conditioning embedding and process
        hidden_states = torch.cat([hidden_states, conditioning_embedding], dim=1)
        hidden_states = self.conv_add_cond_emb_to_hidden(hidden_states)

        # 5. Run the hidden states through the trunk of the denoising model
        for unet_block in self.blocks:
            hidden_states = unet_block(hidden_states, temb=emb)[0]
        for post_res_block in self.post_res_blocks:
            hidden_states = post_res_block(hidden_states, emb)

        # 6. Map hidden states out to a denoised sample
        hidden_states = F.silu(self.norm_out(hidden_states))
        denoised_sample = self.conv_out(hidden_states)

        if not return_dict:
            return (denoised_sample,)

        return TortoiseTTSDenoisingModelOutput(sample=denoised_sample)
