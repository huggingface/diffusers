import torch
from torch import nn
from torch.nn import functional as F
from .activations import get_activation

def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = False, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
    bias=None,
    stride=1,
    dilation=1
):
    """
    https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/training/networks_stylegan3.py
    """
#     with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = (w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)) # [NOIkk]
    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)

    x = torch.nn.functional.conv2d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding, dilation=dilation, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

class UniControlTaskMOEEmbedding(nn.Module):
    def __init__(
        self,
        time_embed_dim,
        all_tasks_num = 13,
        hint_channels=3, #conditioning_channels
        model_channels=320, #conditioning_embedding_channels
        conditioning_embedding_out_channels=(16,32,96,256)
        ):
        super().__init__()

        ## MOE with multiple task specific layers. So num of layers = num of tasks
        ## where each layer is UniControlNetTaskConditioningEmbedding
        in_channels = conditioning_embedding_out_channels[0]
        out_channels = conditioning_embedding_out_channels[1]
        self.input_hint_block_list_moe = nn.ModuleList([])
        for _ in range(all_tasks_num):
            self.input_hint_block_list_moe.append(
                UniControlTaskConditioningEmbedding(
                    hint_channels=hint_channels,
                    in_channels = in_channels,
                    out_channels = out_channels
                    )
                )

        # self.input_hint_block_share = nn.ModuleList([
        #     nn.Conv2d(conditioning_embedding_out_channels[1], conditioning_embedding_out_channels[1], 3, padding=1),
        #     nn.Conv2d(conditioning_embedding_out_channels[1], conditioning_embedding_out_channels[2], 3, padding=1, stride=2),
        #     nn.Conv2d(conditioning_embedding_out_channels[2], conditioning_embedding_out_channels[2], 3, padding=1),
        #     nn.Conv2d(conditioning_embedding_out_channels[2],  conditioning_embedding_out_channels[3], 3, padding=1, stride=2),
        # ])
        
        blocks = []
        for i in range(1, len(conditioning_embedding_out_channels) - 1):
            channel_in = conditioning_embedding_out_channels[i]
            channel_out = conditioning_embedding_out_channels[i + 1]
            conv1 = nn.Conv2d(
                channel_in,
                channel_in,
                kernel_size=3,
                padding=1,
            )
            blocks.append(conv1)
            conv2 = nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=3,
                padding=1,
                stride=2
            )
            blocks.append(conv2)
        self.input_hint_block_share = nn.ModuleList(blocks)

        if len(conditioning_embedding_out_channels) < 2:
            raise ValueError("`conditioning_embedding_out_channels` must have atleast two channels in the list")
        self.task_id_layernet_zeroconv_0 = nn.Linear(time_embed_dim, conditioning_embedding_out_channels[1])

        #The second zero conv is redundant
        self.input_hint_block_zeroconv_0 = nn.ModuleList([
            zero_module(nn.Conv2d(conditioning_embedding_out_channels[1], conditioning_embedding_out_channels[1], 3, padding=1)),
            zero_module(nn.Conv2d(conditioning_embedding_out_channels[1], conditioning_embedding_out_channels[1], 3, padding=1))]
        )

        #The second zero conv is redundant.
        self.input_hint_block_zeroconv_1 = nn.ModuleList([
            zero_module(nn.Conv2d(conditioning_embedding_out_channels[-1], model_channels, 3, padding=1)),
            zero_module(nn.Conv2d(conditioning_embedding_out_channels[-1], model_channels, 3, padding=1))]
        )
        self.task_id_layernet_zeroconv_1 = nn.Linear(time_embed_dim, conditioning_embedding_out_channels[-1])

    def forward(self, x, hint, context, task_id_emb, task_id):

        batch_size = x.shape[0]

        guided_hint = self.input_hint_block_list_moe[task_id](hint)
        
        guided_hint = modulated_conv2d(guided_hint, self.input_hint_block_zeroconv_0[0].weight, self.task_id_layernet_zeroconv_0(task_id_emb).repeat(batch_size, 1).detach(), padding=1)

        guided_hint += self.input_hint_block_zeroconv_0[0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)


        for layer in self.input_hint_block_share:
            act = nn.SiLU()
            guided_hint = act(layer(guided_hint))
        

        guided_hint = modulated_conv2d(guided_hint, self.input_hint_block_zeroconv_1[0].weight, self.task_id_layernet_zeroconv_1(task_id_emb).repeat(batch_size, 1).detach(), padding=1)


        guided_hint += self.input_hint_block_zeroconv_1[0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return guided_hint

class UniControlTaskConditioningEmbedding(nn.Module):

    def __init__(self, hint_channels, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(hint_channels, in_channels, kernel_size = 3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1, stride=2),
        ])

    def forward(self, conditioning):
        embedding = conditioning
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        return embedding

class UniControlTaskIDHypernet(nn.Module):

    def __init__(
        self,
        time_embed_dim,
        cross_attention_dim=768,
        act_fn='silu',
        post_act_fn='silu'
    ):
        super().__init__()
        self.linear1 = nn.Linear(cross_attention_dim, time_embed_dim)
        self.act = get_activation(act_fn)
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.post_act_fn = get_activation(post_act_fn)

    def forward(self, in_feature):
        out_embedding = self.linear1(in_feature)
        out_embedding = self.act(out_embedding)
        out_embedding = self.linear2(out_embedding)
        out_embedding = self.post_act_fn(out_embedding)
        return out_embedding

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module