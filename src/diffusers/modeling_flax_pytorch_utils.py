# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
""" PyTorch - Flax general utilities."""
import re

import jax.numpy as jnp
import numpy as onp #Original numpy to avoid confusion with jax numpy
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from collections import OrderedDict

from .utils import logging


logger = logging.get_logger(__name__)


def rename_key(key):
    regex = r"\w+[.]\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key


#####################
# PyTorch => Flax #
#####################

# Adapted from https://github.com/huggingface/transformers/blob/c603c80f46881ae18b2ca50770ef65fa4033eacd/src/transformers/modeling_flax_pytorch_utils.py#L69
# and https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/convert_diffusers_to_jax.py
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""

    # conv norm or layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if (
        any("norm" in str_ for str_ in pt_tuple_key)
        and (pt_tuple_key[-1] == "bias")
        and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)
        and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor
    elif pt_tuple_key[-1] in ["weight", "gamma"] and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict:
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict:
        pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor, "transpose(2, 3, 1, 0)"

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight":
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor, "T"

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor


def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42, return_mapping_dict=False):
    mapping_dict={}
    # Step 1: Convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    # Step 2: Since the model is stateless, get random Flax params
    random_flax_params = flax_model.init_weights(PRNGKey(init_key))

    random_flax_state_dict = flatten_dict(random_flax_params)
    flax_state_dict = {}

    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():
        renamed_pt_key = rename_key(pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        # Correctly rename weight parameters
        return_tuple = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict)
        if len(return_tuple)==2:
            flax_key, flax_tensor=return_tuple
            tensor_change=None
        elif len(return_tuple)==3:
            flax_key, flax_tensor,tensor_change=return_tuple

        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)
        mapping_dict[flax_key]={
            "original_name": pt_key,
            "tensor_change": tensor_change
        }
    if return_mapping_dict == True:
        return unflatten_dict(flax_state_dict), mapping_dict
    
    return unflatten_dict(flax_state_dict)

##########################################
# Reversing Flax back to Pytorch #
##########################################
def reverse_flax_into_pytorch(params, mapping_dict):
    #This reverse a set of params for flax back into pytorch format. 
    #mapping_dict is generated by from_pretrained (from_pt=True, return_mapping_dict=True)
    if is_torch_available():
        import torch
    else:
        raise EnvironmentError(
            "Can't save the model in PyTorch format because PyTorch is not installed. "
            "Please, install PyTorch or save as native Flax weights."
            )
        
    state_dict=OrderedDict()
    flatten_params=flatten_dict(params)
    
    for flax_key in flatten_params.keys():
        if flax_key in mapping_dict :
            tensor=flatten_params[flax_key]
            original_name=mapping_dict[flax_key]['original_name']
            if mapping_dict[flax_key]['tensor_change'] == "T":
                tensor=tensor.T
            elif mapping_dict[flax_key]['tensor_change'] == "transpose(2, 3, 1, 0)":
                tensor=tensor.transpose(3, 2, 0, 1)
            else: 
                #Unsure which is better, failing silently or error. Feels like this deserves an error, as it's better than saving broken weights
                assert mapping_dict[flax_key]['tensor_change'] is None, "Unsupported tensor_change found in mapping_dict."
            state_dict[original_name]=torch.from_numpy( onp.array (tensor) )
                
        else:
            raise ValueError(
                    f"flax model weight have key {flax_key}, but isn't in mapping_dict. Adding new weights and reconverting back to pytorch isn't supported. Do not delete key from mapping_dict"
                )
    return state_dict
