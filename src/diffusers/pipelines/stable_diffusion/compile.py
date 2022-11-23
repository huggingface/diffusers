#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
from collections import OrderedDict

import click
import numpy as np
import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target

from .modeling.clip import CLIPTextTransformer as ait_CLIPTextTransformer
from .modeling.unet_2d_condition import UNet2DConditionModel as ait_UNet2DConditionModel
from .modeling.vae import AutoencoderKL as ait_AutoencoderKL


USE_CUDA = detect_target().name() == "cuda"

access_token = True

def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("AIT output_{} shape: {}".format(i, y_shape))


def map_unet_params(pt_mod, dim):
    pt_params = dict(pt_mod.named_parameters())
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait


def map_vae_params(ait_module, pt_module, batch_size, seq_len):
    pt_params = dict(pt_module.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_module.named_parameters():
        ait_name = name.replace(".", "_")
        if name in pt_params:
            if (
                "conv" in name
                and "norm" not in name
                and name.endswith(".weight")
                and len(pt_params[name].shape) == 4
            ):
                mapped_pt_params[ait_name] = torch.permute(
                    pt_params[name], [0, 2, 3, 1]
                ).contiguous()
            else:
                mapped_pt_params[ait_name] = pt_params[name]
        elif name.endswith("attention.qkv.weight"):
            prefix = name[: -len("attention.qkv.weight")]
            q_weight = pt_params[prefix + "query.weight"]
            k_weight = pt_params[prefix + "key.weight"]
            v_weight = pt_params[prefix + "value.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            mapped_pt_params[ait_name] = qkv_weight
        elif name.endswith("attention.qkv.bias"):
            prefix = name[: -len("attention.qkv.bias")]
            q_bias = pt_params[prefix + "query.bias"]
            k_bias = pt_params[prefix + "key.bias"]
            v_bias = pt_params[prefix + "value.bias"]
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            mapped_pt_params[ait_name] = qkv_bias
        elif name.endswith("attention.proj.weight"):
            prefix = name[: -len("attention.proj.weight")]
            pt_name = prefix + "proj_attn.weight"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.proj.bias"):
            prefix = name[: -len("attention.proj.bias")]
            pt_name = prefix + "proj_attn.bias"
            mapped_pt_params[ait_name] = pt_params[pt_name]
        elif name.endswith("attention.cu_length"):
            cu_len = np.cumsum([0] + [seq_len] * batch_size).astype("int32")
            mapped_pt_params[ait_name] = torch.from_numpy(cu_len).cuda()
        else:
            pt_param = pt_module.get_parameter(name)
            mapped_pt_params[ait_name] = pt_param

    return mapped_pt_params


def map_clip_params(pt_mod, batch_size, seqlen, depth):

    params_pt = list(pt_mod.named_parameters())

    params_ait = {}
    pt_params = {}
    for key, arr in params_pt:
        pt_params[key.replace("text_model.", "")] = arr

    pt_params = dict(pt_mod.named_parameters())
    for key, arr in pt_params.items():
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("q_proj.weight"):
            ait_name = ait_name.replace("q_proj", "qkv")
            prefix = key[: -len("q_proj.weight")]
            q = pt_params[prefix + "q_proj.weight"]
            k = pt_params[prefix + "k_proj.weight"]
            v = pt_params[prefix + "v_proj.weight"]
            qkv_weight = torch.cat([q, k, v], dim=0)
            params_ait[ait_name] = qkv_weight
            continue
        elif name.endswith("q_proj.bias"):
            ait_name = ait_name.replace("q_proj", "qkv")
            prefix = key[: -len("q_proj.bias")]
            q = pt_params[prefix + "q_proj.bias"]
            k = pt_params[prefix + "k_proj.bias"]
            v = pt_params[prefix + "v_proj.bias"]
            qkv_bias = torch.cat([q, k, v], dim=0)
            params_ait[ait_name] = qkv_bias
            continue
        elif name.endswith("k_proj.weight"):
            continue
        elif name.endswith("k_proj.bias"):
            continue
        elif name.endswith("v_proj.weight"):
            continue
        elif name.endswith("v_proj.bias"):
            continue
        params_ait[ait_name] = arr

        if USE_CUDA:
            for i in range(depth):
                prefix = "encoder_layers_%d_self_attn_cu_length" % (i)
                cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
                params_ait[prefix] = torch.from_numpy(cu_len).cuda()

    return params_ait


def compile_unet(pipe,
    batch_size=2,
    hh=64,
    ww=64,
    dim=320,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    seqlen=64,
    text_dim=768,
    save_path="./tmp",
):

    ait_mod = ait_UNet2DConditionModel(#sample_size=seqlen, 
                                        cross_attention_dim=text_dim)
    ait_mod.name_parameter_tensor()

    # set AIT parameters
    pt_mod = pipe.unet
    pt_mod = pt_mod.eval()
    params_ait = map_unet_params(pt_mod, dim)

    latent_model_input_ait = Tensor(
        [batch_size, hh, ww, 4], name="input0", is_input=True
    )
    timesteps_ait = Tensor([batch_size], name="input1", is_input=True)
    text_embeddings_pt_ait = Tensor([batch_size, seqlen, text_dim], name="input2", is_input=True)

    Y = ait_mod(latent_model_input_ait, timesteps_ait, text_embeddings_pt_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(Y, target, save_path, "UNet2DConditionModel", constants=params_ait)


def compile_clip(pipe,
    batch_size=1,
    seqlen=64,
    dim=768,
    num_heads=12,
    hidden_size=768,
    vocab_size=49408,
    max_position_embeddings=77,
    use_fp16_acc=False,
    convert_conv_to_gemm=False,
    save_path="./tmp",
):
    mask_seq = 0
    causal = True
    depth = 12

    ait_mod = ait_CLIPTextTransformer(
        num_hidden_layers=depth,
        hidden_size=dim,
        num_attention_heads=num_heads,
        batch_size=batch_size,
        seq_len=seqlen,
        causal=causal,
        mask_seq=mask_seq,
    )
    ait_mod.name_parameter_tensor()

    pt_mod = pipe.text_encoder
    pt_mod = pt_mod.eval()
    params_ait = map_clip_params(pt_mod, batch_size, seqlen, depth)

    input_ids_ait = Tensor(
        [batch_size, seqlen], name="input0", dtype="int64", is_input=True
    )
    position_ids_ait = Tensor(
        [batch_size, seqlen], name="input1", dtype="int64", is_input=True
    )
    Y = ait_mod(input_ids=input_ids_ait, position_ids=position_ids_ait)
    mark_output(Y)

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(Y, target, save_path, "CLIPTextModel", constants=params_ait)


def compile_vae(pipe, 
    batch_size=1, height=64, width=64, use_fp16_acc=False, convert_conv_to_gemm=False, save_path="./tmp",
):
    in_channels = 3
    out_channels = 3
    down_block_types = [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ]
    up_block_types = [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ]
    block_out_channels = [128, 256, 512, 512]
    layers_per_block = 2
    act_fn = "silu"
    latent_channels = 4
    #sample_size = 512

    ait_vae = ait_AutoencoderKL(
        batch_size,
        height,
        width,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        act_fn=act_fn,
        latent_channels=latent_channels,
        #sample_size=sample_size,
    )
    ait_input = Tensor(
        shape=[batch_size, height, width, latent_channels],
        name="vae_input",
        is_input=True,
    )
    ait_vae.name_parameter_tensor()

    pt_mod = pipe.vae
    pt_mod = pt_mod.eval()
    params_ait = map_vae_params(ait_vae, pt_mod, batch_size, height * width)

    Y = ait_vae.decode(ait_input)
    mark_output(Y)
    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        save_path,
        "AutoencoderKL",
        constants=params_ait,
    )

    
    

def compile_diffusers(
    token, width, height, seqlen, batch_size, use_fp16_acc=True, convert_conv_to_gemm=True, pipe=None,
    save_path="./tmp",
):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    global access_token
    if token != "":
        access_token = token

    if pipe is None:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=access_token,
        ).to("cuda")

    ww = width // 8
    hh = height // 8

    # CLIP
    compile_clip(pipe,
        batch_size=batch_size,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        save_path=save_path,
        seqlen=seqlen,
    )
    # UNet
    compile_unet(pipe,
        batch_size=batch_size * 2,
        ww=ww,
        hh=hh,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        save_path=save_path,
        seqlen=seqlen,
    )
    # VAE
    compile_vae(pipe,
        batch_size=batch_size,
        width=ww,
        height=hh,
        use_fp16_acc=use_fp16_acc,
        convert_conv_to_gemm=convert_conv_to_gemm,
        save_path=save_path,
    )


@click.command()
@click.option("--token", default="", help="access token")
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option("--seqlen", default=77, help="Max number of text tokens")
@click.option("--batch-size", default=1, help="batch size")
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
def main(token, width, height, seqlen, batch_size, use_fp16_acc=True, convert_conv_to_gemm=True, pipe=None, save_path="./tmp",):
    compile_diffusers(token, width, height, seqlen, batch_size, use_fp16_acc, convert_conv_to_gemm, pipe, save_path)


if __name__ == "__main__":
    main()
