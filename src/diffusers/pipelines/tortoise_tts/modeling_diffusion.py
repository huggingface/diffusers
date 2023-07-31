from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.embeddings import TimestepEmbedding, Timesteps
from ...models.resnet import AdaGroupNorm, Upsample2D, Downsample2D, upsample_2d, downsample_2d, partial
from ...utils import BaseOutput, logging

from .modeling_common import AttentionBlock, ConditioningEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


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
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",  # default, scale_shift, ada_group
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


class AttnEncoderBlock1D(nn.Module):
    """
    1D U-Net style block with architecture (no down/upsampling)

    ResnetBlock1d => AttentionBlock
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim = 1,
        relative_pos_embeddings: bool = False,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnets = []
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

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
                AttentionBlock(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                    relative_pos_embeddings=relative_pos_embeddings,
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
        in_latent_channels: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 8,
        num_latent_cond_layers: int = 4,
        num_timestep_integrator_layers: int = 3,
        num_post_res_blocks: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        attention_head_dim: int = 32,  # hidden_channels / num_heads = 512 / 16 = 32
        dropout: float = 0.0,
    ):
        super().__init__()

        # TODO: make sure all the blocks are initialized the same way as original code

        # 1. Define latent conditioner, which processes the latent conditioning information
        # from the autoregressive model
        self.latent_conditioner = ConditioningEncoder(
            in_channels=in_latent_channels,
            out_channels=hidden_channels,
            num_layers=num_latent_cond_layers,
            attention_head_dim=attention_head_dim,
            relative_pos_embeddings=True,
            input_transform="conv",
            input_conv_kernel_size=3,
            input_conv_stride=1,
            input_conv_padding=1,
            output_transform="groupnorm",
            output_num_groups=32,  # TODO: get accurate num_groups
        )

        # 2. Define unconditioned embedding (TODO: add more information)
        self.unconditioned_embedding = nn.Parameter(torch.randn(1, hidden_channels, 1))

        # 3. Define conditioning timestep integrator, which combines the conditioning embedding from the
        # autoregressive model with the time embedding
        self.conditioning_timestep_integrator = AttnEncoderBlock1D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            temb_channels=hidden_channels,
            dropout=dropout,
            num_layers=num_timestep_integrator_layers,
        )

        # 4. Define the timestep embedding. Only support positional embeddings for now.
        time_embed_dim = hidden_channels
        self.time_proj = Timesteps(hidden_channels, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift)
        self.time_embedding = TimestepEmbedding(hidden_channels, time_embed_dim)

        # 5. Define the inital Conv1d layers
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv_add_cond_emb_to_hidden = nn.Conv1d(2 * hidden_channels, hidden_channels, 1)

        # 6. Define the trunk of the denoising model
        self.blocks = AttnEncoderBlock1D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            temb_channels=hidden_channels,
            dropout=dropout,
            num_layers=num_layers,
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
        self.norm_out = nn.GroupNorm(32, hidden_channels)  # TODO: get right number of groups
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        autoregressive_latents: torch.FloatTensor,
        conditioning_audio_latents: torch.FloatTensor,
        unconditional: bool = False,
        return_dict: bool = True
    ):
        """
        TODO
        """
        # 1. Handle the conditioning embedding
        if unconditional:
            cond_embedding = self.unconditioned_embedding.repeat(sample.shape[0], 1, sample.shape[-1])
        else:
            cond_scale, cond_shift = torch.chunk(conditioning_audio_latents, 2, dim=1)
            cond_embedding = self.latent_conditioner(autoregressive_latents).embedding
            cond_embedding = cond_embedding * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)
            # Interpolate conditional embeddings...?
            cond_embedding = F.interpolate(cond_embedding, size=sample.shape[-1], mode="nearest")
        
        # 2. Handle timestep embedding
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

        # 3. Combine conditioning embedding with timestep embedding
        cond_embedding = self.conditioning_timestep_integrator(cond_embedding, temb=emb)[0]

        # 4. Map inital sample to hidden states
        sample = self.conv_in(sample)

        # 5. Concatenate initial hidden states with conditioning embedding and process
        sample = torch.cat([sample, cond_embedding], dim=1)
        sample = self.conv_add_cond_emb_to_hidden(sample)

        # 6. Run the hidden states through the trunk of the denoising model
        sample = self.blocks(sample, temb=emb)[0]
        sample = self.post_res_blocks(sample, emb)

        # 7. Map hidden states out to a denoised sample
        sample = F.silu(self.norm_out(sample))
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        
        return TortoiseTTSDenoisingModelOutput(sample=sample)
