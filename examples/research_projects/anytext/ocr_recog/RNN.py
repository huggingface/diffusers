import torch
from torch import nn

from .RecSVTR import Block


class Swish(nn.Module):
    def __int__(self):
        super(Swish, self).__int__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Im2Im(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        return x


class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == 1
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))
        return x


class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(EncoderWithRNN, self).__init__()
        hidden_size = kwargs.get("hidden_size", 256)
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size, bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type="rnn", **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {"reshape": Im2Seq, "rnn": EncoderWithRNN, "svtr": EncoderWithSVTR}
            assert encoder_type in support_encoder_dict, "{} must in {}".format(
                encoder_type, support_encoder_dict.keys()
            )

            self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x


class ConvBNLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias_attr=False, groups=1, act=nn.GELU
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            # weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = Swish()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        dims=64,  # XS
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        qk_scale=None,
    ):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, padding=1, act="swish")
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act="swish")

        self.svtr_block = nn.ModuleList(
            [
                Block(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer="swish",
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer="nn.LayerNorm",
                    epsilon=1e-05,
                    prenorm=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act="swish")
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, padding=1, act="swish")

        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act="swish")
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)

        for blk in self.svtr_block:
            z = blk(z)

        z = self.norm(z)
        # last stage
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))

        return z


if __name__ == "__main__":
    svtrRNN = EncoderWithSVTR(56)
    print(svtrRNN)
