from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin, Sideloads
from .resnet import Downsample2D


class ResnetBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c, down, ksize=3, sk=False, use_conv=True, proj_ksize=1):
        super().__init__()
        ps = ksize // 2
        proj_pad = proj_ksize // 2

        if in_c != mid_c or sk is False:
            self.in_conv = nn.Conv2d(in_c, mid_c, proj_ksize, 1, proj_pad)
        else:
            self.in_conv = None

        if out_c != mid_c:
            self.out_conv = nn.Conv2d(mid_c, out_c, proj_ksize, 1, proj_pad)
        else:
            self.out_conv = None

        self.block1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(mid_c, mid_c, ksize, 1, ps)

        if sk is False:
            self.skep = nn.Conv2d(in_c, mid_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down is True:
            self.down_opt = Downsample2D(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down is True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            h = h + self.skep(x)
        else:
            h = h + x

        if self.out_conv is not None:
            h = self.out_conv(h)
        return h


class Adapter(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes, depth, and others, and
    generates multiple feature maps that can be injected into `UNet2DConditionModel` by `SideloadProcessor`. The
    model's architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        block_out_channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channel of each downsample block's output hidden state. The `len(block_out_channels)` will
            also determine the number of downsample blocks in the Adapter.
        block_mid_channels (`List[int]`, *optional*, defaults to `block_out_channels` if not provided):
            The number of channels ResNet blocks in each downsample blocks will have, a downsample block will insert a
             projection layer in the last ResNet block when having different "mid_channel" and "out_channel".
        num_res_blocks (`int`, *optional*, defaults to 3):
            Number of ResNet blocks in each downsample block
        channels_in (`int`, *optional*, defaults to 3):
            Number of channels of Aapter's input(*control image*). Set this parameter to 1 if you're using gray scale
            image as *control image*.
        kerenl_size (`int`, *optional*, defaults to 3):
            Kernel size of conv-2d layers inside ResNet blocks.
        proj_kerenl_size (`int`, *optional*, defaults to 3):
            Kernel size of conv-2d projection layers located at the start and end of a downsample block.
        res_block_skip (`bool`, *optional*, defaults to True):
            If set to `True`, ResNet block will using a regular residual connect that add layer's input to its output.
            If set to `False`, ResNet block will create a additional conv-2d layer in residual connect before adding
            residual back.
        use_conv (`bool`, *optional*, defaults to False):
            Whether to use a conv-2d layer for down sample feature map or a average pooling layer.
        target_layers (`List[int]`, *optional*, defaults to `Adapter.DEFAULT_TARGET`):
            The names of layers from `UNet2DConditionModel` that adapter's outputs will be fusing to.
        input_scale_factor (`int`, *optional*, defaults to 8):
            The down scaling factor will be apply to input image when it is frist deliver to Adapter. Which should be
            equal to the down scaling factor of the VAE of your choice.
    """

    DEFAULT_TARGET = [
        "down_blocks.0.attentions.1",
        "down_blocks.1.attentions.1",
        "down_blocks.2.attentions.1",
        "down_blocks.3.resnets.1",
    ]

    @register_to_config
    def __init__(
        self,
        block_out_channels: List[int] = [320, 640, 1280, 1280],
        block_mid_channels: Optional[List[int]] = None,
        num_res_blocks: int = 3,
        channels_in: int = 3,
        kerenl_size: int = 3,
        proj_kerenl_size: int = 1,
        res_block_skip: bool = True,
        use_conv: bool = False,
        target_layers: List[str] = DEFAULT_TARGET,
        input_scale_factor: int = 8,
    ):
        super(Adapter, self).__init__()

        self.num_downsample_blocks = len(block_out_channels)
        self.unshuffle = nn.PixelUnshuffle(input_scale_factor)
        self.target_layers = target_layers
        self.num_res_blocks = num_res_blocks
        self.body = []

        if block_mid_channels is None:
            block_mid_channels = block_out_channels

        for i in range(self.num_downsample_blocks):
            for j in range(num_res_blocks):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(
                            block_out_channels[i - 1],
                            block_mid_channels[i],
                            block_mid_channels[i],
                            down=True,
                            ksize=kerenl_size,
                            proj_ksize=proj_kerenl_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
                        )
                    )
                elif j == num_res_blocks - 1:
                    self.body.append(
                        ResnetBlock(
                            block_mid_channels[i],
                            block_mid_channels[i],
                            block_out_channels[i],
                            down=False,
                            ksize=kerenl_size,
                            proj_ksize=proj_kerenl_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
                        )
                    )
                else:
                    self.body.append(
                        ResnetBlock(
                            block_mid_channels[i],
                            block_mid_channels[i],
                            block_mid_channels[i],
                            down=False,
                            ksize=kerenl_size,
                            proj_ksize=proj_kerenl_size,
                            sk=res_block_skip,
                            use_conv=use_conv,
                        )
                    )
        self.body = nn.ModuleList(self.body)

        if block_mid_channels[0] == block_out_channels[0]:
            # follow standar adapter schema, using fix kernel size 3 for stem conv layer
            self.conv_in = nn.Conv2d(channels_in * input_scale_factor**2, block_mid_channels[0], 3, 1, 1)
        else:
            # if block_mid_channels[i] < block_out_channels[i](bottleneck downsample block), using light weight adapter schema instead
            self.conv_in = nn.Conv2d(
                channels_in * input_scale_factor**2,
                block_mid_channels[0],
                proj_kerenl_size,
                1,
                proj_kerenl_size // 2,
            )

    def forward(self, x: torch.Tensor) -> Sideloads:
        r"""
        Args:
            x (`torch.Tensor`):
                (batch, channel, height, width) input images for adapter model, `channel` should equal to
                `channels_in`.
        """
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)
        for i in range(self.num_downsample_blocks):
            for j in range(self.num_res_blocks):
                idx = i * self.num_res_blocks + j
                x = self.body[idx](x)
            features.append(x)

        return Sideloads({layer_name: h for layer_name, h in zip(self.target_layers, features)})


class MultiAdapter(ModelMixin, ConfigMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        num_adapter (`int`): The number of `Adapter` models this MultiAdapter will create or contains.
        adapters_kwargs (`List[dict]`, defaults to `MultiAdapter.default_adapter_kwargs`):
            A list of keyword arguments for the `Adapter` constructor. The length of `adapters_kwargs` should equal to
            `num_adapter`.
        adapters (`List[Adapter]`, *optional*, defaults to None):
            A list of `Adapter` instances. If this parameter is provided, `MultiAdapter` uses these adapters instead of
            creating new ones.
        adapter_weights (`List[float]`, *optional*, defaults to None):
            List of floats representing the weight which will be multiply to each adapter's output before adding them
            together.
    """

    ignore_for_config = ["adapters"]
    default_adapter_kwargs = {
        "block_out_channels": [320, 640, 1280, 1280],
        "num_res_blocks": 3,
        "channels_in": 3,
        "kerenl_size": 3,
        "res_block_skip": False,
        "use_conv": False,
        "target_layers": Adapter.DEFAULT_TARGET,
        "input_scale_factor": 8,
    }

    @register_to_config
    def __init__(
        self,
        num_adapter: int = 2,
        adapters_kwargs: List[Dict[str, Any]] = [default_adapter_kwargs] * 2,
        adapters: Optional[List[Adapter]] = None,
        adapter_weights: Optional[List[float]] = None,
    ):
        super(MultiAdapter, self).__init__()

        self.num_adapter = num_adapter
        if adapters is None:
            self.adapters = nn.ModuleList([Adapter(**kwargs) for kwargs in adapters_kwargs])
        else:
            self._check_adapter_config(adapters_kwargs, adapters)
            self.adapters = nn.ModuleList(adapters)
        if adapter_weights is None:
            self.adapter_weights = nn.Parameter(torch.tensor([1 / num_adapter] * num_adapter))
        else:
            self.adapter_weights = nn.Parameter(torch.tensor(adapter_weights))

    def _check_adapter_config(self, adapters_kwargs: List[Dict[str, Any]], adapters: List[Adapter]):
        for i, (init_kwargs, adapter) in enumerate(zip(adapters_kwargs, adapters)):
            config = adapter.config
            for k, v in init_kwargs.items():
                if v != config[k]:
                    raise ValueError(
                        f"keyword argument \"{k}\" from adapters_kwargs of {i}'th adapter dont match the Adapter instance's config!"
                        f"  {v} != {config[k]}"
                    )

    @classmethod
    def from_adapters(cls, adapters: List[Adapter], adapter_weights: Optional[List[float]] = None) -> "MultiAdapter":
        r"""
        Create a MultiAdapter model with existing Adapter instances

        Args:
            adapters (`List[Adapter]`):
                List of `Adapter` instances `MultiAdapter` will use.
            adapter_weights (`List[float]`, *optional*, defaults to None):
                List of floats representing the weight which will be multiply to each adapter's output before adding
                them together.
        """

        def get_public_kwargs(kwargs):
            return {k: v for k, v in kwargs.items() if not k.startswith("_")}

        adapters_kwargs = [get_public_kwargs(adapter.config) for adapter in adapters]
        multi_adapter = cls(
            num_adapter=len(adapters),
            adapters_kwargs=adapters_kwargs,
            adapters=adapters,
            adapter_weights=adapter_weights,
        )
        return multi_adapter

    def forward(self, xs: torch.Tensor) -> Sideloads:
        r"""
        Args:
            xs (`torch.Tensor`):
                (batch, channel, height, width) input images for multiple adapter models concated along dimension 1,
                `channel` should equal to `num_adapter` * "number of channel of image".
        """
        if xs.shape[1] % self.num_adapter != 0:
            raise ValueError(
                f"Expecting multi-adapter's input have number of channel that cab be evenly divisible "
                f"by num_adapter: {xs.shape[1]} % {self.num_adapter} != 0"
            )
        x_list = torch.chunk(xs, self.num_adapter, dim=1)
        accume_state = None
        for x, w, adapter in zip(x_list, self.adapter_weights, self.adapters):
            sideload = adapter(x)
            if accume_state is None:
                accume_state = Sideloads({layer_name: h * w for layer_name, h in sideload.items()})
            else:
                for layer_name in sideload.keys():
                    accume_state[layer_name] += w * sideload[layer_name]
        return accume_state
