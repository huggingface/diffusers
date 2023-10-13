import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


class HypernetworkModule(torch.nn.Module):
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    # Add all activations from torch.nn.modules.activation.__all__
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in torch.nn.modules.activation.__dict__.items() if cls_name in torch.nn.modules.activation.__all__})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, activate_output=False, dropout_structure=None):
        super().__init__()

        self.multiplier = 1.0

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected _layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last _layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add _layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Everything should be now parsed into dropout structure, and applied here.
            # Since we only have dropouts after layers, dropout structure should start with 0 and end with 0.
            if dropout_structure is not None and dropout_structure[i+1] > 0:
                assert 0 < dropout_structure[i+1] < 1, "Dropout probability should be 0 or float between 0 and 1!"
                linears.append(torch.nn.Dropout(p=dropout_structure[i+1]))
            # Code explanation : [1, 2, 1] -> dropout is missing when last_layer_dropout is false. [1, 2, 2, 1] -> [0, 0.3, 0, 0], when its True, [0, 0.3, 0.3, 0].

        self.linear = torch.nn.Sequential(*linears)

        # Define a dictionary mapping weight initialization methods to their functions
        weight_init_functions = {
            "Normal": (init.normal_, {'mean': 0.0, 'std': 0.01}),
            "XavierUniform": (init.xavier_uniform_, {}),
            "XavierNormal": (init.xavier_normal_, {}),
            "KaimingUniform": (init.kaiming_uniform_, {'nonlinearity': 'leaky_relu' if 'leakyrelu' == activation_func else 'relu'}),
            "KaimingNormal": (init.kaiming_normal_, {'nonlinearity': 'leaky_relu' if 'leakyrelu' == activation_func else 'relu'}),
        }

        def initialize_weights(_layer, _weight_init):
            if type(_layer) == torch.nn.Linear or type(_layer) == torch.nn.LayerNorm:
                w, b = _layer.weight.data, _layer.bias.data
                weight_init_fn, kwargs = weight_init_functions.get(_weight_init, None)
                if weight_init_fn is None:
                    raise KeyError(f"Key {_weight_init} is not defined as initialization!")
                weight_init_fn(w, **kwargs)
                init.zeros_(b)

        if state_dict is None:
            for layer in self.linear:
                initialize_weights(layer, weight_init)
        else:
            self.load_state_dict(state_dict)

    def forward(self, x):
        return x + self.linear(x) * (self.multiplier if not self.training else 1)


def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout, default_dropout_p=0.3):
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    if not use_dropout:
        return [0] * len(layer_structure)
    dropout_values = [0]
    dropout_values.extend([default_dropout_p] * (len(layer_structure) - 3))
    if last_layer_dropout:
        dropout_values.append(default_dropout_p)
    else:
        dropout_values.append(0)
    dropout_values.append(0)
    return dropout_values


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, activate_output=False, **kwargs):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.weight_init = weight_init
        self.add_layer_norm = add_layer_norm
        self.use_dropout = use_dropout
        self.activate_output = activate_output
        self.last_layer_dropout = kwargs.get('last_layer_dropout', True)
        self.dropout_structure = kwargs.get('dropout_structure', None)
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        self.optimizer_name = None
        self.optimizer_state_dict = None
        self.optional_info = None

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
            )

    def weights(self):
        res = []
        for layers in self.layers.values():
            for layer in layers:
                res += layer.parameters()
        return res

    def train(self, mode=True):
        for layers in self.layers.values():
            for layer in layers:
                layer.train(mode=mode)
                for param in layer.parameters():
                    param.requires_grad = mode

    def to(self, device):
        for layers in self.layers.values():
            for layer in layers:
                layer.to(device)

        return self

    def set_multiplier(self, multiplier):
        for layers in self.layers.values():
            for layer in layers:
                layer.multiplier = multiplier
        return self

    def eval(self):
        for layers in self.layers.values():
            for layer in layers:
                layer.eval()
        return self

    def load_state_dict(self, state_dict):
        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        self.optional_info = state_dict.get('optional_info', None)
        self.activation_func = state_dict.get('activation_func', None)
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.dropout_structure = state_dict.get('dropout_structure', None)
        self.use_dropout = True if self.dropout_structure is not None and any(self.dropout_structure) else state_dict.get('use_dropout', False)
        self.activate_output = state_dict.get('activate_output', True)
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)
        # Dropout structure should have same length as layer structure, Every digits should be in [0,1), and last digit must be 0.
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)

        for size, hypernet_state_dict in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, hypernet_state_dict[0], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                    HypernetworkModule(size, hypernet_state_dict[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)
        self.eval()


def apply_single_hypernetwork(hypernetwork, context_k, context_v, layer=None):
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context_k.shape[2], None)

    if hypernetwork_layers is None:
        return context_k, context_v

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = hypernetwork_layers[0](context_k)
    context_v = hypernetwork_layers[1](context_v)
    return context_k, context_v


def apply_hypernetworks(hypernetworks, context, layer=None):
    context_k = context
    context_v = context
    for hypernetwork in hypernetworks:
        context_k, context_v = apply_single_hypernetwork(hypernetwork, context_k, context_v, layer)

    return context_k, context_v



class HyperAttnProcessor(nn.Module):
    r"""
    Processor for implementing the hypernet attention mechanism.

    Args:
        hypernets (`list`, *optional*):
            List of hypernetworks to use
    """

    def __init__(self, hypernets=[]):
        super().__init__()
        self.hypernetworks = hypernets
   
    def __call__(
        self,
        attn: 'Attention',
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        context = encoder_hidden_states
        context_k, context_v = apply_hypernetworks(self.hypernetworks, context, self)

        key = attn.to_k(context_k)
        value = attn.to_v(context_v)

        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def add_hypernet(unet: 'UNet2DConditionModel', hypernet: Hypernetwork) -> None:
    """
    Add a hypernetwork to an unet.
    """
    # fill attn processors
    attn_processors = unet.attn_processors
    keys = unet.attn_processors.keys()
    for key in keys:
        if isinstance(attn_processors[key], HyperAttnProcessor):
            attn_processors[key].hypernetworks.append(hypernet)
        else:
            attn_processors[key] = HyperAttnProcessor([hypernet])
    unet.set_attn_processor(attn_processors)


def clear_hypernets(unet: 'UNet2DConditionModel') -> None:
    """
    Remove all hypernetworks from an unet.
    """
    # fill attn processors
    attn_processors = unet.attn_processors
    keys = unet.attn_processors.keys()
    for key in keys:
        if isinstance(attn_processors[key], HyperAttnProcessor):
            attn_processors[key].hypernetworks = []
        else:
            pass
