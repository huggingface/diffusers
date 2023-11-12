# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple

from itertools import zip_longest

import torch
from torch import nn
from torch.nn.modules.normalization import GroupNorm
import torch.utils.checkpoint

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging
from .embeddings import get_timestep_embedding
from .modeling_utils import ModelMixin
from .lora import LoRACompatibleConv
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    CrossAttnUpBlock2D,
    UpBlock2D,
    ResnetBlock2D,
    Transformer2DModel,
    Downsample2D,
    Upsample2D,
)
from .unet_2d_condition import UNet2DConditionModel
from ..umer_debug_logger import udl


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# todo Umer later: add attention_bias to relevant docs

@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None


class ControlNetXSModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    """A ControlNet-XS model."""

    # to delete later
    @classmethod
    def create_as_in_paper(cls, base_model=None):
        if base_model is None:
            # todo: load sdxl instead
            base_model = UNet2DConditionModel(
                block_out_channels=(320, 640, 1280),
                down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
                up_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
                transformer_layers_per_block=(0,2,10),
                cross_attention_dim=2048,
            )

        def class_names(modules): return [m.__class__.__name__ for m in modules]
        def get_time_emb_dim(unet: UNet2DConditionModel): return unet.time_embedding.linear_2.out_features
        def get_time_emb_input_dim(unet: UNet2DConditionModel):return unet.time_embedding.linear_1.in_features

        base_model_channel_sizes = ControlNetXSModel.gather_base_model_sizes(base_model, base_or_control='base')

        cnxs_model = cls(
            model_channels=320,
            out_channels=4,
            hint_channels=3,
            block_out_channels=(32,64,128),
            down_block_types=class_names(base_model.down_blocks),
            up_block_types=class_names(base_model.up_blocks),
            time_embedding_dim=get_time_emb_dim(base_model),
            time_embedding_input_dim=get_time_emb_input_dim(base_model),
            transformer_layers_per_block=(0,2,10),
            cross_attention_dim=2048,
            learn_embedding=True,
            control_model_ratio=0.1,
            base_model_channel_sizes=base_model_channel_sizes,
            control_scale=0.95,
            addition_embed_type='text_time',
        )
        cnxs_model.base_model = base_model
        return cnxs_model

    @classmethod
    def gather_base_model_sizes(cls, unet: UNet2DConditionModel, base_or_control):
        if base_or_control not in ['base', 'control']:
            raise ValueError(f"`base_or_control` needs to be either `base` or `control`")

        channel_sizes = {'enc': [], 'mid': [], 'dec': []}

        # input convolution
        channel_sizes['enc'].append((unet.conv_in.in_channels, unet.conv_in.out_channels))

        # encoder blocks
        for module in unet.down_blocks:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                for r in module.resnets:
                    channel_sizes['enc'].append((r.in_channels, r.out_channels))
                if module.downsamplers:
                    channel_sizes['enc'].append((module.downsamplers[0].channels, module.downsamplers[0].out_channels))
            else:
                raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        # middle block
        channel_sizes['mid'].append((unet.mid_block.resnets[0].in_channels, unet.mid_block.resnets[0].out_channels))

        # decoder blocks
        if base_or_control == 'base':
            for module in unet.up_blocks:
                if isinstance(module, (CrossAttnUpBlock2D, UpBlock2D)):
                    for r in module.resnets:
                        channel_sizes['dec'].append((r.in_channels, r.out_channels))
                else:
                   raise ValueError(f'Encountered unknown module of type {type(module)} while creating ControlNet-XS.')

        return channel_sizes

    @register_to_config
    def __init__(
            self,
            model_channels=320,
            out_channels=4,
            hint_channels=3,
            block_out_channels=(32,64,128),
            down_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
            up_block_types=("DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
            time_embedding_dim=1280,
            time_embedding_input_dim=320,
            transformer_layers_per_block=(0,2,10),
            cross_attention_dim: Union[int, Tuple[int]] = 2048,#1280,
            learn_embedding=False,
            control_model_ratio=1.0,
            base_model_channel_sizes={
                'enc': [(4, 320), (320, 320), (320, 320), (320, 320), (320, 640), (640, 640), (640, 640), (640, 1280), (1280, 1280)],
                'mid': [(1280, 1280)],
                'dec': [(2560, 1280), (2560, 1280), (1920, 1280), (1920, 640), (1280, 640), (960, 640), (960, 320), (640, 320), (640, 320)]
            },
            global_pool_conditions: bool = False, # Todo Umer: Needed by SDXL pipeline, but what is this?,
            control_scale=1,
            time_control_scale=1,
            addition_embed_type: Optional[str] = None,
        ):
        super().__init__()

        # 1 - Save parameters
        # TODO make variables
        self.in_ch_factor = 1 if "cat" == 'add' else 2
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.hint_model = None
        self.no_control = False
        self.learn_embedding = learn_embedding

        # 1 - Create controller
        self.control_model = UNet2DConditionModel(
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            time_embedding_dim=time_embedding_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            cross_attention_dim=cross_attention_dim,
        )

        # 2 - Do model surgery on control model
        # 2.1 - Allow to use the same time information as the base model
        adjust_time_input_dim(self.control_model, time_embedding_input_dim)
        # 2.2 - Allow for information infusion from base model
        # todo: make variable (sth like zip(block_out_channels[:-1],block_out_channels[1:]))
        for i, extra_channels in enumerate(((320, 320), (320,640), (640,1280))):
            e1,e2=extra_channels
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=0, by=e1)
            increase_block_input_in_encoder_resnet(self.control_model, block_no=i, resnet_idx=1, by=e2)
            if self.control_model.down_blocks[i].downsamplers: increase_block_input_in_encoder_downsampler(self.control_model, block_no=i, by=e2)
        increase_block_input_in_mid_resnet(self.control_model, by=1280) # todo: make var

        # 3 - Gather Channel Sizes
        self.ch_inout_ctrl = ControlNetXSModel.gather_base_model_sizes(self.control_model, base_or_control='control')
        self.ch_inout_base = base_model_channel_sizes

        # 4 - Build connections between base and control model
        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])
        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])
        self.dec_zero_convs_out = nn.ModuleList([])
        self.dec_zero_convs_in = nn.ModuleList([])

        for ch_io_base in self.ch_inout_base['enc']:
            self.enc_zero_convs_in.append(self.make_zero_conv(
                in_channels=ch_io_base[1], out_channels=ch_io_base[1])
            )
        for i in range(len(self.ch_inout_ctrl['enc'])):
            self.enc_zero_convs_out.append(
                self.make_zero_conv(self.ch_inout_ctrl['enc'][i][1], self.ch_inout_base['enc'][i][1])
            )       
 
        self.middle_block_out = self.make_zero_conv(self.ch_inout_ctrl['mid'][-1][1], self.ch_inout_base['mid'][-1][1])
        
        self.dec_zero_convs_out.append(
            self.make_zero_conv(self.ch_inout_ctrl['enc'][-1][1], self.ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(self.ch_inout_ctrl['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(self.ch_inout_ctrl['enc'][-(i + 1)][1], self.ch_inout_base['dec'][i - 1][1])
            )

        # 5 - Create conditioning hint embedding
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, int(model_channels * self.control_model_ratio), 3, padding=1))
        )
    
        # 6 - Create time embedding
        pass
        self.flip_sin_to_cos = True # default params
        self.freq_shift = 0
        # !! TODO !! : learn_embedding is True, so we need our own embedding
        # Edit: That's already part of the ctrl model, even thought it's not used
        # Todo: Only when `learn_embedding = False` can we just use the base model's time embedding, otherwise we need to create our own 
        
        # Text embedding
        # info: I deleted the encoder_hid_proj as it's not given by the Heidelberg CVL weights

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list) * control_scale)

        # in the mininal implementation setting, we only need the control model up to the mid block
        # note: these can only be deleted after  has to be `gather_base_model_sizes(self.control_mode, 'control')` has been called
        del self.control_model.up_blocks
        del self.control_model.conv_norm_out
        del self.control_model.conv_out

    def toggle_control(self, to):
        if not hasattr(self, 'do_control'): self.do_control = True
        if not hasattr(self, 'scale_back_up'): self.back_up = None
        if self.do_control == to:
            print(f'Model already set to control mode == {to}')
            return
        if not to:
            self.scale_back_up = self.scale_list[0].clone()
            self.scale_list = self.scale_list * 0.
            self.do_control = False
        else:
            self.scale_list = self.scale_list * 0. + self.scale_back_up 
            self.scale_back_up = None
            self.do_control = True
        assert self.do_control == to
        print(f'Model set to control mode == {self.do_control}')


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        hint: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        no_control=False,
    ):
        if self.base_model is None:
            raise RuntimeError("To use `forward`, first set the base model for this ControlNetXSModel by `cnxs_model.base_model = the_base_model`")

        print('control_scale:',self.scale_list)

        """ Params from unet_2d_condition.UNet2DConditionModel.forward:
        # self,
        # sample: torch.FloatTensor,
        # timestep: Union[torch.Tensor, float, int],
        # encoder_hidden_states: torch.Tensor,
        # class_labels: Optional[torch.Tensor] = None,
        # timestep_cond: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        # down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        # mid_block_additional_residual: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        # return_dict: bool = True,
        """

        #x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if x.size(0) // 2 == hint.size(0): hint = torch.cat([hint, hint], dim=0) # for classifier free guidance
        
        timesteps=t
        
        if no_control or self.no_control:
            return self.base_model(x, timesteps, encoder_hidden_states,cross_attention_kwargs=cross_attention_kwargs,added_cond_kwargs=added_cond_kwargs)

        # time embeddings
        timesteps = timesteps[None]

        udl.log_if('timestep', timesteps, condition='TIME', print_=True)
        t_emb = get_timestep_embedding(
            timesteps, 
            self.model_channels,
            # # TODO: Undetrstand flip_sin_to_cos / (downscale_)freq_shift
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.freq_shift,
        )
        udl.log_if('time_emb', t_emb, condition='TIME', print_=True)
        if self.learn_embedding:
            udl.log_if('time_proj_ctrl',self.control_model.time_embedding(t_emb), condition='TIME', print_=True)
            udl.log_if('time_proj_ctrl_scaled',self.control_model.time_embedding(t_emb) * self.config.time_control_scale ** 0.3, condition='TIME', print_=True)
            udl.log_if('time_proj_base',self.base_model.time_embedding(t_emb), condition='TIME', print_=True)
            udl.log_if('time_proj_base_scaled',self.base_model.time_embedding(t_emb) * (1 - self.config.time_control_scale ** 0.3), condition='TIME', print_=True)
            temb = self.control_model.time_embedding(t_emb) * self.config.time_control_scale ** 0.3 + self.base_model.time_embedding(t_emb) * (1 - self.config.time_control_scale ** 0.3)
        else:
            temb = self.base_model.time_embedding(t_emb)
        udl.log_if('time_proj',temb, condition='TIME', print_=True)

        aug_emb = None

        # text embeddings
        cemb = encoder_hidden_states

        # added time & text embeddings
        if self.config.addition_embed_type == "text":
            raise NotImplementedError()
        elif self.config.addition_embed_type == "text_image":
            raise NotImplementedError()
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            udl.log_if('text_embeds',text_embeds, condition='TIME', print_=True)
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            udl.log_if('add_input',time_ids.flatten(), condition='TIME', print_=True)
            time_embeds = self.base_model.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            udl.log_if('add_emb',time_embeds, condition='TIME', print_=True)
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(temb.dtype)
            aug_emb = self.base_model.add_embedding(add_embeds)
            udl.log_if('add_proj',aug_emb, condition='TIME', print_=True)

        elif self.config.addition_embed_type == "image":
            raise NotImplementedError()
        elif self.config.addition_embed_type == "image_hint":
            raise NotImplementedError()

        temb = temb + aug_emb if aug_emb is not None else temb
        udl.log_if('final_temb',temb,condition='TIME', print_=True)
        udl.stop_if(condition='TIME', funny_msg='Time to analyze time!')

        ###
        guided_hint = self.input_hint_block(hint)

        h_ctrl = h_base = x
        hs_base, hs_ctrl = [], []
        it_enc_convs_in, it_enc_convs_out, it_dec_convs_in, it_dec_convs_out = map(iter, (self.enc_zero_convs_in, self.enc_zero_convs_out, self.dec_zero_convs_in, self.dec_zero_convs_out))
        scales = iter(self.scale_list)

        base_down_subblocks = to_sub_blocks(self.base_model.down_blocks)
        ctrl_down_subblocks = to_sub_blocks(self.control_model.down_blocks)
        base_mid_subblocks = to_sub_blocks([self.base_model.mid_block])
        ctrl_mid_subblocks = to_sub_blocks([self.control_model.mid_block])
        base_up_subblocks = to_sub_blocks(self.base_model.up_blocks)

        # Debug Umer -- to delete later on
        # this is for a global view, ie on subblock level
        udl.log_if('prep.x',            x,              condition='SUBBLOCK')
        udl.log_if('prep.temb',         temb,           condition='SUBBLOCK')
        udl.log_if('prep.context',      cemb,           condition='SUBBLOCK')
        udl.log_if('prep.raw_hint',     hint,           condition='SUBBLOCK')
        udl.log_if('prep.guided_hint',  guided_hint,    condition='SUBBLOCK')

        # Debug Umer - another one!
        # this is for a detail view, ie below subblock level

        # Cross Control
        # 0 - conv in
        h_base = self.base_model.conv_in(h_base)
        udl.log_if('enc.h_base', h_base, condition='SUBBLOCK')

        h_ctrl = self.control_model.conv_in(h_ctrl)
        udl.log_if('enc.h_ctrl', h_ctrl, condition='SUBBLOCK')
        
        if guided_hint is not None: h_ctrl += guided_hint
        udl.log_if('enc.h_ctrl', h_ctrl, condition='SUBBLOCK')
        
        h_base = h_base + next(it_enc_convs_out)(h_ctrl) * next(scales)
        udl.log_if('enc.h_base', h_base, condition='SUBBLOCK')

        hs_base.append(h_base)
        hs_ctrl.append(h_ctrl)
        # 1 - input blocks (encoder)
        RUN_ONCE = ('SUBBLOCK', 'SUBBLOCK-MINUS-1')
        udl.print_if('------ enc ------', conditions=RUN_ONCE)
        for i, (m_base, m_ctrl)  in enumerate(zip(base_down_subblocks, ctrl_down_subblocks)):
            # A - concat base -> ctrl
            cat_to_ctrl = next(it_enc_convs_in)(h_base)
            h_ctrl = torch.cat([h_ctrl, cat_to_ctrl], dim=1)
            udl.log_if('enc.h_ctrl', h_ctrl, condition='SUBBLOCK')
            # B - apply base subblock
            udl.print_if('>> Applying base block\t', end='', conditions=RUN_ONCE)
            h_base = m_base(h_base, temb, cemb)
            udl.log_if('enc.h_base', h_base, condition='SUBBLOCK')
            # C - apply ctrl subblock
            udl.print_if('>> Applying ctrl block\t', end='', conditions=RUN_ONCE)
            h_ctrl = m_ctrl(h_ctrl, temb, cemb)
            udl.log_if('enc.h_ctrl', h_ctrl, condition='SUBBLOCK')
            udl.print_if('', conditions=RUN_ONCE)
            # D - add ctrl -> base
            add_to_base = next(it_enc_convs_out)(h_ctrl)
            scale = next(scales)
            h_base = h_base + add_to_base * scale
            udl.log_if('enc.h_base', h_base, condition='SUBBLOCK')
            hs_base.append(h_base)
            hs_ctrl.append(h_ctrl)
        h_ctrl = torch.concat([h_ctrl, h_base], dim=1)
        udl.log_if('enc.h_ctrl', h_ctrl, condition='SUBBLOCK')
        # 2 - mid blocks (bottleneck)
        udl.print_if('------ mid ------', conditions=RUN_ONCE)
        # Because Heidelberg treats the R/A/R as one block, they first execute the full base mid block,
        # then the full ctrl mid block; while I execute them interlaced.
        # This doesn't change the computation, but messes up parts of the logs.
        # So let's, while debugging, first execute full base mid block and then full ctrl mid block.

        #for m_base, m_ctrl in zip(base_mid_subblocks, ctrl_mid_subblocks):
        #    udl.print_if('>> Applying base block\t', end='', conditions=RUN_ONCE)
        #    h_base = m_base(h_base, temb, cemb)
        #    udl.print_if('>> Applying ctrl block\t', end='', conditions=RUN_ONCE)
        #    h_ctrl = m_ctrl(h_ctrl, temb, cemb)
        #    udl.print_if('', conditions=RUN_ONCE)

        for m_base in base_mid_subblocks:
            udl.print_if('>> Applying base block\t', end='', conditions=RUN_ONCE)
            h_base = m_base(h_base, temb, cemb)
            udl.print_if('', conditions=RUN_ONCE)

        for m_ctrl in ctrl_mid_subblocks:
            udl.print_if('>> Applying ctrl block\t', end='', conditions=RUN_ONCE)
            h_ctrl = m_ctrl(h_ctrl, temb, cemb)
            udl.print_if('', conditions=RUN_ONCE)

        # Heidelberg treats the R/A/R as one block, while I treat is as 2 subblocks
        # Let's therefore only log after the mid section
        udl.log_if('mid.h_base', h_base, condition='SUBBLOCK')
        udl.log_if('mid.h_ctrl', h_ctrl, condition='SUBBLOCK')
    
        h_base = h_base + self.middle_block_out(h_ctrl) * next(scales)
        udl.log_if('mid.h_base', h_base, condition='SUBBLOCK')

        # 3 - output blocks (decoder)
        udl.print_if('------ dec ------', conditions=RUN_ONCE)
        for m_base in base_up_subblocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctrl.pop()) * next(scales) # add info from ctrl encoder 
            udl.log_if('dec.h_base', h_base, condition='SUBBLOCK')
            h_base = torch.cat([h_base, hs_base.pop()], dim=1) # concat info from base encoder+ctrl encoder
            udl.log_if('dec.h_base', h_base, condition='SUBBLOCK')
            udl.print_if('>> Applying base block\t', end='', conditions=RUN_ONCE)
            h_base = m_base(h_base, temb, cemb)
            udl.log_if('dec.h_base', h_base, condition='SUBBLOCK')
            udl.print_if('',conditions=RUN_ONCE)

        h_base = self.base_model.conv_norm_out(h_base)
        h_base = self.base_model.conv_act(h_base)
        h_base = self.base_model.conv_out(h_base)

        result = h_base

        udl.log_if('conv_out.h_base', result, condition='SUBBLOCK')
        udl.print_if('',conditions=RUN_ONCE)

        udl.stop_if('SUBBLOCK', 'The subblocks are cought. Let us gaze into their soul, their very essence.')
        udl.stop_if('SUBBLOCK-MINUS-1', 'Alright captain. Look at all these tensors we caught. Time to do some real analysis.')

        return UNet2DConditionOutput(sample=result)

    def make_zero_conv(self, in_channels, out_channels=None):
        # keep running track # todo: better comment
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))

    def debug_print(self, s):
        if hasattr(self, 'debug') and self.debug:
            print(s)


def adjust_time_input_dim(unet: UNet2DConditionModel, dim: int):
    time_emb = unet.time_embedding
    time_emb.linear_1 = nn.Linear(dim, time_emb.linear_1.out_features)

def increase_block_input_in_encoder_resnet(unet:UNet2DConditionModel, block_no, resnet_idx, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    r=unet.down_blocks[block_no].resnets[resnet_idx]
    old_norm1, old_conv1, old_conv_shortcut = r.norm1,r.conv1,r.conv_shortcut
    # norm
    norm_args = 'num_groups num_channels eps affine'.split(' ')
    for a in norm_args: assert hasattr(old_norm1, a)
    norm_kwargs = { a: getattr(old_norm1, a) for a in norm_args }
    norm_kwargs['num_channels'] += by  # surgery done here
    # conv1
    conv1_args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in conv1_args: assert hasattr(old_conv1, a)
    conv1_kwargs = { a: getattr(old_conv1, a) for a in conv1_args }
    conv1_kwargs['bias'] = 'bias' in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs['in_channels'] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work) 
    conv_shortcut_args_kwargs = { 
        'in_channels': conv1_kwargs['in_channels'],
        'out_channels': conv1_kwargs['out_channels'],
        # default arguments from resnet.__init__
        'kernel_size':1, 
        'stride':1, 
        'padding':0,
        'bias':True
    }
    # swap old with new modules
    unet.down_blocks[block_no].resnets[resnet_idx].norm1 = GroupNorm(**norm_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.down_blocks[block_no].resnets[resnet_idx].in_channels += by  # surgery done here


def increase_block_input_in_encoder_downsampler(unet:UNet2DConditionModel, block_no, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    old_down=unet.down_blocks[block_no].downsamplers[0].conv
    # conv1
    args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in args: assert hasattr(old_down, a)
    kwargs = { a: getattr(old_down, a) for a in args}
    kwargs['bias'] = 'bias' in kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    kwargs['in_channels'] += by  # surgery done here
    # swap old with new modules
    unet.down_blocks[block_no].downsamplers[0].conv = LoRACompatibleConv(**kwargs)
    unet.down_blocks[block_no].downsamplers[0].channels += by  # surgery done here


def increase_block_input_in_mid_resnet(unet:UNet2DConditionModel, by):
    """Increase channels sizes to allow for additional concatted information from base model"""
    m=unet.mid_block.resnets[0]
    old_norm1, old_conv1, old_conv_shortcut = m.norm1,m.conv1,m.conv_shortcut
    # norm
    norm_args = 'num_groups num_channels eps affine'.split(' ')
    for a in norm_args: assert hasattr(old_norm1, a)
    norm_kwargs = { a: getattr(old_norm1, a) for a in norm_args }
    norm_kwargs['num_channels'] += by  # surgery done here
    # conv1
    conv1_args = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode lora_layer'.split(' ')
    for a in conv1_args: assert hasattr(old_conv1, a)
    conv1_kwargs = { a: getattr(old_conv1, a) for a in conv1_args }
    conv1_kwargs['bias'] = 'bias' in conv1_kwargs  # as param, bias is a boolean, but as attr, it's a tensor.
    conv1_kwargs['in_channels'] += by  # surgery done here
    # conv_shortcut
    # as we changed the input size of the block, the input and output sizes are likely different,
    # therefore we need a conv_shortcut (simply adding won't work) 
    conv_shortcut_args_kwargs = { 
        'in_channels': conv1_kwargs['in_channels'],
        'out_channels': conv1_kwargs['out_channels'],
        # default arguments from resnet.__init__
        'kernel_size':1, 
        'stride':1, 
        'padding':0,
        'bias':True
    }
    # swap old with new modules
    unet.mid_block.resnets[0].norm1 = GroupNorm(**norm_kwargs)
    unet.mid_block.resnets[0].conv1 = LoRACompatibleConv(**conv1_kwargs)
    unet.mid_block.resnets[0].conv_shortcut = LoRACompatibleConv(**conv_shortcut_args_kwargs)
    unet.mid_block.resnets[0].in_channels += by  # surgery done here


class EmbedSequential(nn.ModuleList):
    """Sequential module passing embeddings (time and conditioning) to children if they support it."""
    def __init__(self,ms,*args,**kwargs):
        if not is_iterable(ms): ms = [ms]
        super().__init__(ms,*args,**kwargs)
    
    def forward(self,x,temb,cemb):
        def cls_name(x): return str(type(x)).split('.')[-1].replace("'>",'')
        content = ' '.join(cls_name(m) for m in self)
        udl.print_if(f'EmbedSequential.forward with content {content}', conditions='SUBBLOCK-MINUS-1')
        for m in self:
            if isinstance(m,ResnetBlock2D):
                x = m(x,temb)
            elif isinstance(m,Transformer2DModel):
                x = m(x,cemb).sample
            elif isinstance(m,Downsample2D):
                x = m(x)
            elif isinstance(m,Upsample2D):
                x = m(x)
            else: raise ValueError(f'Type of m is {type(m)} but should be `ResnetBlock2D`, `Transformer2DModel`,  `Downsample2D` or `Upsample2D`')

        return x


def is_iterable(o):
    if isinstance(o, str): return False
    try:
        iter(o)
        return True
    except TypeError:
        return False


def to_sub_blocks(blocks):
    if not is_iterable(blocks): blocks = [blocks]
    sub_blocks = []
    for b in blocks:
        current_subblocks = []
        if hasattr(b, 'resnets'):
            if hasattr(b, 'attentions') and b.attentions is not None:
                current_subblocks = list(zip_longest(b.resnets, b.attentions))
                 # if we have 1 more resnets than attentions, let the last subblock only be the resnet, not (resnet, None)
                if current_subblocks[-1][1] is None:
                    current_subblocks[-1] = current_subblocks[-1][0]
            else:
                current_subblocks = list(b.resnets)
        # upsamplers are part of the same block # q: what if we have multiple upsamplers?
        if hasattr(b, 'upsamplers') and b.upsamplers is not None: current_subblocks[-1] = list(current_subblocks[-1]) + list(b.upsamplers)
        # downsamplers are own block
        if hasattr(b, 'downsamplers') and b.downsamplers is not None: current_subblocks.append(list(b.downsamplers))   
        sub_blocks += current_subblocks
    return list(map(EmbedSequential, sub_blocks))


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
