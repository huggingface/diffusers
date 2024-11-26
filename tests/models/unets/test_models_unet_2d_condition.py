# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import copy
import gc
import os
import tempfile
import unittest
from collections import OrderedDict

import torch
from huggingface_hub import snapshot_download
from parameterized import parameterized
from pytest import mark

from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import ImageProjection, IPAdapterFaceIDImageProjection, IPAdapterPlusImageProjection
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    is_peft_available,
    load_hf_numpy,
    require_peft_backend,
    require_torch_accelerator,
    require_torch_accelerator_with_fp16,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_all_close,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


if is_peft_available():
    from peft import LoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer


logger = logging.get_logger(__name__)

enable_full_determinism()


def get_unet_lora_config():
    rank = 4
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=False,
        use_dora=False,
    )
    return unet_lora_config


def check_if_lora_correctly_set(model) -> bool:
    """
    Checks if the LoRA layers are correctly set with peft
    """
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return True
    return False


def create_ip_adapter_state_dict(model):
    # "ip_adapter" (cross-attention weights)
    ip_cross_attn_state_dict = {}
    key_id = 1

    for name in model.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") or "motion_module" in name else model.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]

        if cross_attention_dim is not None:
            sd = IPAdapterAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
            ).state_dict()
            ip_cross_attn_state_dict.update(
                {
                    f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                    f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                }
            )

            key_id += 2

    # "image_proj" (ImageProjection layer weights)
    cross_attention_dim = model.config["cross_attention_dim"]
    image_projection = ImageProjection(
        cross_attention_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, num_image_text_embeds=4
    )

    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.weight": sd["image_embeds.weight"],
            "proj.bias": sd["image_embeds.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    ip_state_dict = {}
    ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
    return ip_state_dict


def create_ip_adapter_plus_state_dict(model):
    # "ip_adapter" (cross-attention weights)
    ip_cross_attn_state_dict = {}
    key_id = 1

    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        if cross_attention_dim is not None:
            sd = IPAdapterAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
            ).state_dict()
            ip_cross_attn_state_dict.update(
                {
                    f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                    f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                }
            )

            key_id += 2

    # "image_proj" (ImageProjection layer weights)
    cross_attention_dim = model.config["cross_attention_dim"]
    image_projection = IPAdapterPlusImageProjection(
        embed_dims=cross_attention_dim, output_dims=cross_attention_dim, dim_head=32, heads=2, num_queries=4
    )

    ip_image_projection_state_dict = OrderedDict()
    keys = [k for k in image_projection.state_dict() if "layers." in k]
    print(keys)
    for k, v in image_projection.state_dict().items():
        if "2.to" in k:
            k = k.replace("2.to", "0.to")
        elif "layers.0.ln0" in k:
            k = k.replace("layers.0.ln0", "layers.0.0.norm1")
        elif "layers.0.ln1" in k:
            k = k.replace("layers.0.ln1", "layers.0.0.norm2")
        elif "layers.1.ln0" in k:
            k = k.replace("layers.1.ln0", "layers.1.0.norm1")
        elif "layers.1.ln1" in k:
            k = k.replace("layers.1.ln1", "layers.1.0.norm2")
        elif "layers.2.ln0" in k:
            k = k.replace("layers.2.ln0", "layers.2.0.norm1")
        elif "layers.2.ln1" in k:
            k = k.replace("layers.2.ln1", "layers.2.0.norm2")
        elif "layers.3.ln0" in k:
            k = k.replace("layers.3.ln0", "layers.3.0.norm1")
        elif "layers.3.ln1" in k:
            k = k.replace("layers.3.ln1", "layers.3.0.norm2")
        elif "to_q" in k:
            parts = k.split(".")
            parts[2] = "attn"
            k = ".".join(parts)
        elif "to_out.0" in k:
            parts = k.split(".")
            parts[2] = "attn"
            k = ".".join(parts)
            k = k.replace("to_out.0", "to_out")
        else:
            k = k.replace("0.ff.0", "0.1.0")
            k = k.replace("0.ff.1.net.0.proj", "0.1.1")
            k = k.replace("0.ff.1.net.2", "0.1.3")

            k = k.replace("1.ff.0", "1.1.0")
            k = k.replace("1.ff.1.net.0.proj", "1.1.1")
            k = k.replace("1.ff.1.net.2", "1.1.3")

            k = k.replace("2.ff.0", "2.1.0")
            k = k.replace("2.ff.1.net.0.proj", "2.1.1")
            k = k.replace("2.ff.1.net.2", "2.1.3")

            k = k.replace("3.ff.0", "3.1.0")
            k = k.replace("3.ff.1.net.0.proj", "3.1.1")
            k = k.replace("3.ff.1.net.2", "3.1.3")

        # if "norm_cross" in k:
        #     ip_image_projection_state_dict[k.replace("norm_cross", "norm1")] = v
        # elif "layer_norm" in k:
        #     ip_image_projection_state_dict[k.replace("layer_norm", "norm2")] = v
        if "to_k" in k:
            parts = k.split(".")
            parts[2] = "attn"
            k = ".".join(parts)
            ip_image_projection_state_dict[k.replace("to_k", "to_kv")] = torch.cat([v, v], dim=0)
        elif "to_v" in k:
            continue
        else:
            ip_image_projection_state_dict[k] = v

    ip_state_dict = {}
    ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
    return ip_state_dict


def create_ip_adapter_faceid_state_dict(model):
    # "ip_adapter" (cross-attention weights)
    # no LoRA weights
    ip_cross_attn_state_dict = {}
    key_id = 1

    for name in model.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") or "motion_module" in name else model.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]

        if cross_attention_dim is not None:
            sd = IPAdapterAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
            ).state_dict()
            ip_cross_attn_state_dict.update(
                {
                    f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                    f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                }
            )

            key_id += 2

    # "image_proj" (ImageProjection layer weights)
    cross_attention_dim = model.config["cross_attention_dim"]
    image_projection = IPAdapterFaceIDImageProjection(
        cross_attention_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, mult=2, num_tokens=4
    )

    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.0.weight": sd["ff.net.0.proj.weight"],
            "proj.0.bias": sd["ff.net.0.proj.bias"],
            "proj.2.weight": sd["ff.net.2.weight"],
            "proj.2.bias": sd["ff.net.2.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    ip_state_dict = {}
    ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
    return ip_state_dict


def create_custom_diffusion_layers(model, mock_weights: bool = True):
    train_kv = True
    train_q_out = True
    custom_diffusion_attn_procs = {}

    st = model.state_dict()
    for name, _ in model.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
            "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(model.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
            if mock_weights:
                # add 1 to weights to mock trained weights
                with torch.no_grad():
                    custom_diffusion_attn_procs[name].to_k_custom_diffusion.weight += 1
                    custom_diffusion_attn_procs[name].to_v_custom_diffusion.weight += 1
        else:
            custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    del st
    return custom_diffusion_attn_procs


class UNet2DConditionModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet2DConditionModel
    main_input_name = "sample"
    # We override the items here because the unet under consideration is small.
    model_split_percents = [0.5, 0.3, 0.4]

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (16, 16)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        time_step = torch.tensor([10]).to(torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 16, 16)

    @property
    def output_shape(self):
        return (4, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (4, 8),
            "norm_num_groups": 4,
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 8,
            "attention_head_dim": 2,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 16,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_enable_works(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        model.enable_xformers_memory_efficient_attention()

        assert (
            model.mid_block.attentions[0].transformer_blocks[0].attn1.processor.__class__.__name__
            == "XFormersAttnProcessor"
        ), "xformers is not enabled"

    def test_model_with_attention_head_dim_tuple(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_use_linear_projection(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["use_linear_projection"] = True

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_cross_attention_dim_tuple(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["cross_attention_dim"] = (8, 8)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_simple_projection(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        batch_size, _, _, sample_size = inputs_dict["sample"].shape

        init_dict["class_embed_type"] = "simple_projection"
        init_dict["projection_class_embeddings_input_dim"] = sample_size

        inputs_dict["class_labels"] = floats_tensor((batch_size, sample_size)).to(torch_device)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_class_embeddings_concat(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        batch_size, _, _, sample_size = inputs_dict["sample"].shape

        init_dict["class_embed_type"] = "simple_projection"
        init_dict["projection_class_embeddings_input_dim"] = sample_size
        init_dict["class_embeddings_concat"] = True

        inputs_dict["class_labels"] = floats_tensor((batch_size, sample_size)).to(torch_device)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_attention_slicing(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        model.set_attention_slice("auto")
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

        model.set_attention_slice("max")
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

        model.set_attention_slice(2)
        with torch.no_grad():
            output = model(**inputs_dict)
        assert output is not None

    def test_model_sliceable_head_dim(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)

        def check_sliceable_dim_attr(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                assert isinstance(module.sliceable_head_dim, int)

            for child in module.children():
                check_sliceable_dim_attr(child)

        # retrieve number of attention layers
        for module in model.children():
            check_sliceable_dim_attr(module)

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "CrossAttnUpBlock2D",
            "CrossAttnDownBlock2D",
            "UNetMidBlock2DCrossAttn",
            "UpBlock2D",
            "Transformer2DModel",
            "DownBlock2D",
        }
        attention_head_dim = (8, 16)
        block_out_channels = (16, 32)
        super().test_gradient_checkpointing_is_applied(
            expected_set=expected_set, attention_head_dim=attention_head_dim, block_out_channels=block_out_channels
        )

    def test_special_attn_proc(self):
        class AttnEasyProc(torch.nn.Module):
            def __init__(self, num):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(num))
                self.is_run = False
                self.number = 0
                self.counter = 0

            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, number=None):
                batch_size, sequence_length, _ = hidden_states.shape
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                query = attn.to_q(hidden_states)

                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                hidden_states += self.weight

                self.is_run = True
                self.counter += 1
                self.number = number

                return hidden_states

        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        processor = AttnEasyProc(5.0)

        model.set_attn_processor(processor)
        model(**inputs_dict, cross_attention_kwargs={"number": 123}).sample

        assert processor.counter == 8
        assert processor.is_run
        assert processor.number == 123

    @parameterized.expand(
        [
            # fmt: off
            [torch.bool],
            [torch.long],
            [torch.float],
            # fmt: on
        ]
    )
    def test_model_xattn_mask(self, mask_dtype):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**{**init_dict, "attention_head_dim": (8, 16), "block_out_channels": (16, 32)})
        model.to(torch_device)
        model.eval()

        cond = inputs_dict["encoder_hidden_states"]
        with torch.no_grad():
            full_cond_out = model(**inputs_dict).sample
            assert full_cond_out is not None

            keepall_mask = torch.ones(*cond.shape[:-1], device=cond.device, dtype=mask_dtype)
            full_cond_keepallmask_out = model(**{**inputs_dict, "encoder_attention_mask": keepall_mask}).sample
            assert full_cond_keepallmask_out.allclose(
                full_cond_out, rtol=1e-05, atol=1e-05
            ), "a 'keep all' mask should give the same result as no mask"

            trunc_cond = cond[:, :-1, :]
            trunc_cond_out = model(**{**inputs_dict, "encoder_hidden_states": trunc_cond}).sample
            assert not trunc_cond_out.allclose(
                full_cond_out, rtol=1e-05, atol=1e-05
            ), "discarding the last token from our cond should change the result"

            batch, tokens, _ = cond.shape
            mask_last = (torch.arange(tokens) < tokens - 1).expand(batch, -1).to(cond.device, mask_dtype)
            masked_cond_out = model(**{**inputs_dict, "encoder_attention_mask": mask_last}).sample
            assert masked_cond_out.allclose(
                trunc_cond_out, rtol=1e-05, atol=1e-05
            ), "masking the last token from our cond should be equivalent to truncating that token out of the condition"

    # see diffusers.models.attention_processor::Attention#prepare_attention_mask
    # note: we may not need to fix mask padding to work for stable-diffusion cross-attn masks.
    # since the use-case (somebody passes in a too-short cross-attn mask) is pretty esoteric.
    # maybe it's fine that this only works for the unclip use-case.
    @mark.skip(
        reason="we currently pad mask by target_length tokens (what unclip needs), whereas stable-diffusion's cross-attn needs to instead pad by remaining_length."
    )
    def test_model_xattn_padding(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**{**init_dict, "attention_head_dim": (8, 16)})
        model.to(torch_device)
        model.eval()

        cond = inputs_dict["encoder_hidden_states"]
        with torch.no_grad():
            full_cond_out = model(**inputs_dict).sample
            assert full_cond_out is not None

            batch, tokens, _ = cond.shape
            keeplast_mask = (torch.arange(tokens) == tokens - 1).expand(batch, -1).to(cond.device, torch.bool)
            keeplast_out = model(**{**inputs_dict, "encoder_attention_mask": keeplast_mask}).sample
            assert not keeplast_out.allclose(full_cond_out), "a 'keep last token' mask should change the result"

            trunc_mask = torch.zeros(batch, tokens - 1, device=cond.device, dtype=torch.bool)
            trunc_mask_out = model(**{**inputs_dict, "encoder_attention_mask": trunc_mask}).sample
            assert trunc_mask_out.allclose(
                keeplast_out
            ), "a mask with fewer tokens than condition, will be padded with 'keep' tokens. a 'discard-all' mask missing the final token is thus equivalent to a 'keep last' mask."

    def test_custom_diffusion_processors(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        custom_diffusion_attn_procs = create_custom_diffusion_layers(model, mock_weights=False)

        # make sure we can set a list of attention processors
        model.set_attn_processor(custom_diffusion_attn_procs)
        model.to(torch_device)

        # test that attn processors can be set to itself
        model.set_attn_processor(model.attn_processors)

        with torch.no_grad():
            sample2 = model(**inputs_dict).sample

        assert (sample1 - sample2).abs().max() < 3e-3

    def test_custom_diffusion_save_load(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            old_sample = model(**inputs_dict).sample

        custom_diffusion_attn_procs = create_custom_diffusion_layers(model, mock_weights=False)
        model.set_attn_processor(custom_diffusion_attn_procs)

        with torch.no_grad():
            sample = model(**inputs_dict).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname, safe_serialization=False)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_custom_diffusion_weights.bin")))
            torch.manual_seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname, weight_name="pytorch_custom_diffusion_weights.bin")
            new_model.to(torch_device)

        with torch.no_grad():
            new_sample = new_model(**inputs_dict).sample

        assert (sample - new_sample).abs().max() < 1e-4

        # custom diffusion and no custom diffusion should be the same
        assert (sample - old_sample).abs().max() < 3e-3

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_custom_diffusion_xformers_on_off(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)
        custom_diffusion_attn_procs = create_custom_diffusion_layers(model, mock_weights=False)
        model.set_attn_processor(custom_diffusion_attn_procs)

        # default
        with torch.no_grad():
            sample = model(**inputs_dict).sample

            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample

            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample

        assert (sample - on_sample).abs().max() < 1e-4
        assert (sample - off_sample).abs().max() < 1e-4

    def test_pickle(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample = model(**inputs_dict).sample

        sample_copy = copy.copy(sample)

        assert (sample - sample_copy).abs().max() < 1e-4

    def test_asymmetrical_unet(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        # Add asymmetry to configs
        init_dict["transformer_layers_per_block"] = [[3, 2], 1]
        init_dict["reverse_transformer_layers_per_block"] = [[3, 4], 1]

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        output = model(**inputs_dict).sample
        expected_shape = inputs_dict["sample"].shape

        # Check if input and output shapes are the same
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_ip_adapter(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        # forward pass without ip-adapter
        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        # update inputs_dict for ip-adapter
        batch_size = inputs_dict["encoder_hidden_states"].shape[0]
        # for ip-adapter image_embeds has shape [batch_size, num_image, embed_dim]
        image_embeds = floats_tensor((batch_size, 1, model.config.cross_attention_dim)).to(torch_device)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds]}

        # make ip_adapter_1 and ip_adapter_2
        ip_adapter_1 = create_ip_adapter_state_dict(model)

        image_proj_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["image_proj"].items()}
        cross_attn_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["ip_adapter"].items()}
        ip_adapter_2 = {}
        ip_adapter_2.update({"image_proj": image_proj_state_dict_2, "ip_adapter": cross_attn_state_dict_2})

        # forward pass ip_adapter_1
        model._load_ip_adapter_weights([ip_adapter_1])
        assert model.config.encoder_hid_dim_type == "ip_image_proj"
        assert model.encoder_hid_proj is not None
        assert model.down_blocks[0].attentions[0].transformer_blocks[0].attn2.processor.__class__.__name__ in (
            "IPAdapterAttnProcessor",
            "IPAdapterAttnProcessor2_0",
        )
        with torch.no_grad():
            sample2 = model(**inputs_dict).sample

        # forward pass with ip_adapter_2
        model._load_ip_adapter_weights([ip_adapter_2])
        with torch.no_grad():
            sample3 = model(**inputs_dict).sample

        # forward pass with ip_adapter_1 again
        model._load_ip_adapter_weights([ip_adapter_1])
        with torch.no_grad():
            sample4 = model(**inputs_dict).sample

        # forward pass with multiple ip-adapters and multiple images
        model._load_ip_adapter_weights([ip_adapter_1, ip_adapter_2])
        # set the scale for ip_adapter_2 to 0 so that result should be same as only load ip_adapter_1
        for attn_processor in model.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                attn_processor.scale = [1, 0]
        image_embeds_multi = image_embeds.repeat(1, 2, 1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds_multi, image_embeds_multi]}
        with torch.no_grad():
            sample5 = model(**inputs_dict).sample

        # forward pass with single ip-adapter & single image when image_embeds is not a list and a 2-d tensor
        image_embeds = image_embeds.squeeze(1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": image_embeds}

        model._load_ip_adapter_weights(ip_adapter_1)
        with torch.no_grad():
            sample6 = model(**inputs_dict).sample

        assert not sample1.allclose(sample2, atol=1e-4, rtol=1e-4)
        assert not sample2.allclose(sample3, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample4, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample5, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample6, atol=1e-4, rtol=1e-4)

    def test_ip_adapter_plus(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        # forward pass without ip-adapter
        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        # update inputs_dict for ip-adapter
        batch_size = inputs_dict["encoder_hidden_states"].shape[0]
        # for ip-adapter-plus image_embeds has shape [batch_size, num_image, sequence_length, embed_dim]
        image_embeds = floats_tensor((batch_size, 1, 1, model.config.cross_attention_dim)).to(torch_device)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds]}

        # make ip_adapter_1 and ip_adapter_2
        ip_adapter_1 = create_ip_adapter_plus_state_dict(model)

        image_proj_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["image_proj"].items()}
        cross_attn_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["ip_adapter"].items()}
        ip_adapter_2 = {}
        ip_adapter_2.update({"image_proj": image_proj_state_dict_2, "ip_adapter": cross_attn_state_dict_2})

        # forward pass ip_adapter_1
        model._load_ip_adapter_weights([ip_adapter_1])
        assert model.config.encoder_hid_dim_type == "ip_image_proj"
        assert model.encoder_hid_proj is not None
        assert model.down_blocks[0].attentions[0].transformer_blocks[0].attn2.processor.__class__.__name__ in (
            "IPAdapterAttnProcessor",
            "IPAdapterAttnProcessor2_0",
        )
        with torch.no_grad():
            sample2 = model(**inputs_dict).sample

        # forward pass with ip_adapter_2
        model._load_ip_adapter_weights([ip_adapter_2])
        with torch.no_grad():
            sample3 = model(**inputs_dict).sample

        # forward pass with ip_adapter_1 again
        model._load_ip_adapter_weights([ip_adapter_1])
        with torch.no_grad():
            sample4 = model(**inputs_dict).sample

        # forward pass with multiple ip-adapters and multiple images
        model._load_ip_adapter_weights([ip_adapter_1, ip_adapter_2])
        # set the scale for ip_adapter_2 to 0 so that result should be same as only load ip_adapter_1
        for attn_processor in model.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                attn_processor.scale = [1, 0]
        image_embeds_multi = image_embeds.repeat(1, 2, 1, 1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds_multi, image_embeds_multi]}
        with torch.no_grad():
            sample5 = model(**inputs_dict).sample

        # forward pass with single ip-adapter & single image when image_embeds is a 3-d tensor
        image_embeds = image_embeds[:,].squeeze(1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": image_embeds}

        model._load_ip_adapter_weights(ip_adapter_1)
        with torch.no_grad():
            sample6 = model(**inputs_dict).sample

        assert not sample1.allclose(sample2, atol=1e-4, rtol=1e-4)
        assert not sample2.allclose(sample3, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample4, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample5, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample6, atol=1e-4, rtol=1e-4)

    @require_torch_gpu
    @parameterized.expand(
        [
            ("hf-internal-testing/unet2d-sharded-dummy", None),
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format", "fp16"),
        ]
    )
    def test_load_sharded_checkpoint_from_hub(self, repo_id, variant):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        loaded_model = self.model_class.from_pretrained(repo_id, variant=variant)
        loaded_model = loaded_model.to(torch_device)
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    @parameterized.expand(
        [
            ("hf-internal-testing/unet2d-sharded-dummy-subfolder", None),
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format-subfolder", "fp16"),
        ]
    )
    def test_load_sharded_checkpoint_from_hub_subfolder(self, repo_id, variant):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        loaded_model = self.model_class.from_pretrained(repo_id, subfolder="unet", variant=variant)
        loaded_model = loaded_model.to(torch_device)
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    def test_load_sharded_checkpoint_from_hub_local(self):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        ckpt_path = snapshot_download("hf-internal-testing/unet2d-sharded-dummy")
        loaded_model = self.model_class.from_pretrained(ckpt_path, local_files_only=True)
        loaded_model = loaded_model.to(torch_device)
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    def test_load_sharded_checkpoint_from_hub_local_subfolder(self):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        ckpt_path = snapshot_download("hf-internal-testing/unet2d-sharded-dummy-subfolder")
        loaded_model = self.model_class.from_pretrained(ckpt_path, subfolder="unet", local_files_only=True)
        loaded_model = loaded_model.to(torch_device)
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    @parameterized.expand(
        [
            ("hf-internal-testing/unet2d-sharded-dummy", None),
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format", "fp16"),
        ]
    )
    def test_load_sharded_checkpoint_device_map_from_hub(self, repo_id, variant):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        loaded_model = self.model_class.from_pretrained(repo_id, variant=variant, device_map="auto")
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    @parameterized.expand(
        [
            ("hf-internal-testing/unet2d-sharded-dummy-subfolder", None),
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format-subfolder", "fp16"),
        ]
    )
    def test_load_sharded_checkpoint_device_map_from_hub_subfolder(self, repo_id, variant):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        loaded_model = self.model_class.from_pretrained(repo_id, variant=variant, subfolder="unet", device_map="auto")
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    def test_load_sharded_checkpoint_device_map_from_hub_local(self):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        ckpt_path = snapshot_download("hf-internal-testing/unet2d-sharded-dummy")
        loaded_model = self.model_class.from_pretrained(ckpt_path, local_files_only=True, device_map="auto")
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_torch_gpu
    def test_load_sharded_checkpoint_device_map_from_hub_local_subfolder(self):
        _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        ckpt_path = snapshot_download("hf-internal-testing/unet2d-sharded-dummy-subfolder")
        loaded_model = self.model_class.from_pretrained(
            ckpt_path, local_files_only=True, subfolder="unet", device_map="auto"
        )
        new_output = loaded_model(**inputs_dict)

        assert loaded_model
        assert new_output.sample.shape == (4, 4, 16, 16)

    @require_peft_backend
    def test_load_attn_procs_raise_warning(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        # forward pass without LoRA
        with torch.no_grad():
            non_lora_sample = model(**inputs_dict).sample

        unet_lora_config = get_unet_lora_config()
        model.add_adapter(unet_lora_config)

        assert check_if_lora_correctly_set(model), "Lora not correctly set in UNet."

        # forward pass with LoRA
        with torch.no_grad():
            lora_sample_1 = model(**inputs_dict).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname)
            model.unload_lora()

            with self.assertWarns(FutureWarning) as warning:
                model.load_attn_procs(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

            warning_message = str(warning.warnings[0].message)
            assert "Using the `load_attn_procs()` method has been deprecated" in warning_message

            # import to still check for the rest of the stuff.
            assert check_if_lora_correctly_set(model), "Lora not correctly set in UNet."

            with torch.no_grad():
                lora_sample_2 = model(**inputs_dict).sample

        assert not torch.allclose(
            non_lora_sample, lora_sample_1, atol=1e-4, rtol=1e-4
        ), "LoRA injected UNet should produce different results."
        assert torch.allclose(
            lora_sample_1, lora_sample_2, atol=1e-4, rtol=1e-4
        ), "Loading from a saved checkpoint should produce identical results."


@slow
class UNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_unet_model(self, fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
        variant = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch_dtype, variant=variant
        )
        model.to(torch_device).eval()

        return model

    @require_torch_gpu
    def test_set_attention_slice_auto(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        unet = self.get_unet_model()
        unet.set_attention_slice("auto")

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    @require_torch_gpu
    def test_set_attention_slice_max(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        unet = self.get_unet_model()
        unet.set_attention_slice("max")

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    @require_torch_gpu
    def test_set_attention_slice_int(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        unet = self.get_unet_model()
        unet.set_attention_slice(2)

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    @require_torch_gpu
    def test_set_attention_slice_list(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # there are 32 sliceable layers
        slice_list = 16 * [2, 3]
        unet = self.get_unet_model()
        unet.set_attention_slice(slice_list)

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        hidden_states = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return hidden_states

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4424, 0.1510, -0.1937, 0.2118, 0.3746, -0.3957, 0.0160, -0.0435]],
            [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.2540, 0.3633, -0.0821, 0.1719, -0.0207]],
            [21, 0.89, [-0.6479, 0.6364, -0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]],
            [9, 1000, [0.8888, -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_v1_4(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919, -0.1571, -0.1125, -0.5806]],
            [17, 0.55, [-0.0831, -0.2443, 0.0901, -0.0919, 0.3396, 0.0103, -0.3743, 0.0701]],
            [8, 0.89, [-0.4863, 0.0859, 0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]],
            [3, 1000, [-0.5649, 0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_v1_4_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4430, 0.1570, -0.1867, 0.2376, 0.3205, -0.3681, 0.0525, -0.0722]],
            [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257, 0.3430, -0.0536, 0.2114, -0.0436]],
            [21, 0.89, [-0.7091, 0.6664, -0.3643, 0.9032, 0.4499, -0.6541, 0.0139, 0.1750]],
            [9, 1000, [0.8878, -0.5659, 0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_compvis_sd_v1_5(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2695, -0.1669, 0.0073, -0.3181, -0.1187, -0.1676, -0.1395, -0.5972]],
            [17, 0.55, [-0.1290, -0.2588, 0.0551, -0.0916, 0.3286, 0.0238, -0.3669, 0.0322]],
            [8, 0.89, [-0.5283, 0.1198, 0.0870, -0.1141, 0.9189, -0.0150, 0.5474, 0.4319]],
            [3, 1000, [-0.5601, 0.2411, -0.5435, 0.1268, 1.1338, -0.2427, -0.0280, -1.0020]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_v1_5_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423, -0.7972, 0.0085, -0.4858]],
            [47, 0.55, [-0.6564, 0.0795, -1.9026, -0.6258, 1.8235, 1.2056, 1.2169, 0.9073]],
            [21, 0.89, [0.0327, 0.4399, -0.6358, 0.3417, 0.4120, -0.5621, -0.0397, -1.0430]],
            [9, 1000, [0.1600, 0.7303, -1.0556, -0.3515, -0.7440, -1.2037, -1.8149, -1.8931]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_compvis_sd_inpaint(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stable-diffusion-v1-5/stable-diffusion-inpainting")
        latents = self.get_latents(seed, shape=(4, 9, 64, 64))
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == (4, 4, 64, 64)

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.1047, -1.7227, 0.1067, 0.0164, -0.5698, -0.4172, -0.1388, 1.1387]],
            [17, 0.55, [0.0975, -0.2856, -0.3508, -0.4600, 0.3376, 0.2930, -0.2747, -0.7026]],
            [8, 0.89, [-0.0952, 0.0183, -0.5825, -0.1981, 0.1131, 0.4668, -0.0395, -0.3486]],
            [3, 1000, [0.4790, 0.4949, -1.0732, -0.7158, 0.7959, -0.9478, 0.1105, -0.9741]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_inpaint_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stable-diffusion-v1-5/stable-diffusion-inpainting", fp16=True)
        latents = self.get_latents(seed, shape=(4, 9, 64, 64), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == (4, 4, 64, 64)

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [0.1514, 0.0807, 0.1624, 0.1016, -0.1896, 0.0263, 0.0677, 0.2310]],
            [17, 0.55, [0.1164, -0.0216, 0.0170, 0.1589, -0.3120, 0.1005, -0.0581, -0.1458]],
            [8, 0.89, [-0.1758, -0.0169, 0.1004, -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]],
            [3, 1000, [0.1214, 0.0352, -0.0731, -0.1562, -0.0994, -0.0906, -0.2340, -0.0539]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_stabilityai_sd_v2_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stabilityai/stable-diffusion-2", fp16=True)
        latents = self.get_latents(seed, shape=(4, 4, 96, 96), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, shape=(4, 77, 1024), fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)
