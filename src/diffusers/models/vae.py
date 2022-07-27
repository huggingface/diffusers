import numpy as np
import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from .attention import AttentionBlock
from .resnet import Downsample2D, ResnetBlock2D, Upsample2D
from .unet_blocks import UNetMidBlock2D, get_up_block, get_down_block


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        act_fn="silu",
        double_z=True,
        # To delete
#        ch=None,
#        ch_mult=(1, 2, 4, 8),
#        num_res_blocks=None,
#        attn_resolutions=None,
#        dropout=0.0,
#        resamp_with_conv=True,
#        resolution=None,
#        z_channels=None,
#        **ignore_kwargs,
    ):
        super().__init__()
#        self.ch = ch
#        self.temb_ch = 0
#        self.num_resolutions = len(ch_mult)
#        self.num_res_blocks = num_res_blocks
#        self.resolution = resolution
#        self.in_channels = in_channels
        self.layers_per_block = layers_per_block

        if True:
            ch = block_out_channels[0]
            ch_mult = [x // ch for x in block_out_channels]
            resolution = None
            z_channels = out_channels
            attn_resolutions = ()
            num_res_blocks = layers_per_block
            resamp_with_conv = True

            self.init_orig(
                ch=ch,
                ch_mult=ch_mult,
                resolution=resolution,
                z_channels=z_channels,
                dropout=0.0,
                attn_resolutions=attn_resolutions,
                resamp_with_conv=resamp_with_conv,
                num_res_blocks=num_res_blocks,
            )
            self.weights_is_set = False

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=32,
            temb_channels=None,
        )

        # out
        num_groups_out = 32
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=num_groups_out, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def init_orig(self, ch, ch_mult, resolution, z_channels, dropout, attn_resolutions, resamp_with_conv, num_res_blocks):
        # downsampling
#        curr_res = resolution
        curr_res = 32
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        num_resolutions = len(ch_mult)
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in, out_channels=block_out, temb_channels=0, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttentionBlock(block_in, overwrite_qkv=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != num_resolutions - 1:
                down.downsample = Downsample2D(block_in, use_conv=resamp_with_conv, padding=0)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout
        )
        self.mid.attn_1 = AttentionBlock(block_in, overwrite_qkv=True)
        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        num_resolutions = len(self.down)
        num_res_blocks = self.layers_per_block
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        out_h = self.forward_2(x)
        print("Diff", (h - out_h).abs().sum())

        return out_h

    def forward_2(self, x):
        self.set_weight()
        # z to block_in
        sample = x
        sample = self.conv_in(sample)
        print("sample", sample.abs().sum())

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)
            print("sample up", sample.abs().sum())

        # middle
        sample = self.mid_block(sample)
        print("sample", sample.abs().sum())

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def set_weight(self):
        if self.weights_is_set:
            return
        self.weights_is_set = True

        self.mid_block.resnets[0].set_weight(self.mid.block_1)
        self.mid_block.resnets[1].set_weight(self.mid.block_2)
        self.mid_block.attentions[0].set_weight(self.mid.attn_1)

        for i, block in enumerate(self.down):
            k = i
            if hasattr(block, "downsample"):
                self.down_blocks[k].downsamplers[0].conv.weight.data = block.downsample.conv.weight.data
                self.down_blocks[k].downsamplers[0].conv.bias.data = block.downsample.conv.bias.data
            if hasattr(block, "block") and len(block.block) > 0:
                for j in range(self.layers_per_block):
                    self.down_blocks[k].resnets[j].set_weight(block.block[j])
            if hasattr(block, "attn") and len(block.attn) > 0:
                for j in range(self.layers_per_block):
                    self.down_blocks[k].attentions[j].set_weight(block.attn[j])

        self.conv_norm_out.weight.data = self.norm_out.weight.data
        self.conv_norm_out.bias.data = self.norm_out.bias.data


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        act_fn="silu",
        # To delete
#        ch=None,
#        out_ch=None,
#        ch_mult=(1, 2, 4, 8),
#        num_res_blocks=None,
#        attn_resolutions=None,
#        dropout=0.0,
#        resamp_with_conv=True,
#        resolution=None,
#        z_channels=None,
#        give_pre_end=False,
#        **ignorekwargs,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        if True:
            ch = block_out_channels[0]
            ch_mult = [x // ch for x in block_out_channels]
            resolution = None
            z_channels = in_channels
            attn_resolutions = ()
            num_res_blocks = layers_per_block
            resamp_with_conv = True

            self.init_orig(
                ch=ch,
                ch_mult=ch_mult,
                resolution=resolution,
                z_channels=z_channels,
                dropout=0.0,
                attn_resolutions=attn_resolutions,
                resamp_with_conv=resamp_with_conv,
                out_ch=out_channels,
                num_res_blocks=num_res_blocks,
            )
            self.weights_is_set = False

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=32,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = 32
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def init_orig(
        self, ch, ch_mult, resolution, z_channels, dropout, attn_resolutions, resamp_with_conv, out_ch, num_res_blocks
    ):
        # compute in_ch_mult, block_in and curr_res at lowest res
        resolution = 32
        block_in = ch * ch_mult[len(ch_mult) - 1]
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = AttentionBlock(block_in, overwrite_qkv=True)
        self.mid.block_2 = ResnetBlock2D(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock2D(in_channels=block_in, out_channels=block_out, temb_channels=0, dropout=dropout)
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttentionBlock(block_in, overwrite_qkv=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2D(block_in, use_conv=resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        print("h", h.abs().sum())

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        print("h", h.abs().sum())

        # upsampling
        for i_level in reversed(range(len(self.up))):
            for i_block in range(self.layers_per_block + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            print("h", h.abs().sum())

        # end
        self.give_pre_end = False
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        out_h = self.forward_2(z)
        print("Diff", (h - out_h).abs().sum())

        return out_h

    def forward_2(self, z):
        self.set_weight()
        # z to block_in
        sample = z
        sample = self.conv_in(sample)
        print("sample", sample.abs().sum())

        # middle
        sample = self.mid_block(sample)
        print("sample", sample.abs().sum())

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)
            print("sample up", sample.abs().sum())

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def set_weight(self):
        if self.weights_is_set:
            return
        self.weights_is_set = True

        self.mid_block.resnets[0].set_weight(self.mid.block_1)
        self.mid_block.resnets[1].set_weight(self.mid.block_2)
        self.mid_block.attentions[0].set_weight(self.mid.attn_1)

        for i, block in enumerate(self.up):
            k = len(self.up) - 1 - i
            if hasattr(block, "upsample"):
                self.up_blocks[k].upsamplers[0].conv.weight.data = block.upsample.conv.weight.data
                self.up_blocks[k].upsamplers[0].conv.bias.data = block.upsample.conv.bias.data
            if hasattr(block, "block") and len(block.block) > 0:
                for j in range(self.layers_per_block + 1):
                    self.up_blocks[k].resnets[j].set_weight(block.block[j])
            if hasattr(block, "attn") and len(block.attn) > 0:
                for j in range(self.layers_per_block + 1):
                    self.up_blocks[k].attentions[j].set_weight(block.attn[j])

        self.conv_norm_out.weight.data = self.norm_out.weight.data
        self.conv_norm_out.bias.data = self.norm_out.bias.data


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class VQModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        act_fn="silu",
        # to delete
        ch=None,
        out_ch=None,
        num_res_blocks=None,
        attn_resolutions=None,
        resolution=None,
        z_channels=None,
        n_embed=None,
        embed_dim=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        double_z=True,
        resamp_with_conv=True,
        give_pre_end=False,
    ):
        super().__init__()

        if True:
            block_out_channels = [ch * c for c in ch_mult]
            down_block_types = [down_block_types[0] for _ in range(len(block_out_channels))]
            up_block_types = [up_block_types[0] for _ in range(len(block_out_channels))]
            layers_per_block = num_res_blocks
            latent_channels = z_channels

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            double_z=False,
        )

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, sample):
        x = sample
        h = self.encode(x)
        dec = self.decode(h)
        return dec


class AutoencoderKL(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        act_fn="silu",
        # to delete
        ch=None,
        out_ch=None,
        num_res_blocks=None,
        attn_resolutions=None,
        resolution=None,
        z_channels=None,
        embed_dim=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
        double_z=True,
        resamp_with_conv=True,
        give_pre_end=False,
    ):
        super().__init__()

        if True:
            block_out_channels = [ch * c for c in ch_mult]
            down_block_types = [down_block_types[0] for _ in range(len(block_out_channels))]
            up_block_types = [up_block_types[0] for _ in range(len(block_out_channels))]
            layers_per_block = num_res_blocks
            latent_channels = z_channels

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            double_z=True,
        )
#        self.encoder = Encoder(
#            ch=ch,
#            out_ch=out_ch,
#            num_res_blocks=num_res_blocks,
#            attn_resolutions=attn_resolutions,
#            in_channels=in_channels,
#            resolution=resolution,
#            z_channels=z_channels,
#            ch_mult=ch_mult,
#            dropout=dropout,
#            resamp_with_conv=resamp_with_conv,
#            give_pre_end=give_pre_end,
#            double_z=True,
#        )
#
        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
        )
        # pass init params to Decoder
#        self.decoder = Decoder(
#            ch=ch,
#            out_ch=out_ch,
#            num_res_blocks=num_res_blocks,
#            attn_resolutions=attn_resolutions,
#            in_channels=in_channels,
#            resolution=resolution,
#            z_channels=z_channels,
#            ch_mult=ch_mult,
#            dropout=dropout,
#            resamp_with_conv=resamp_with_conv,
#            give_pre_end=give_pre_end,
#        )

        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, sample, sample_posterior=False):
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec
