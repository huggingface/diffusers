import inspect
import os
from argparse import ArgumentParser

import numpy as np
import torch
from muse import MaskGiTUViT, VQGANModel
from muse import PipelineMuse as OldPipelineMuse
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import VQModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.unets.uvit_2d import UVit2DModel
from diffusers.pipelines.amused.pipeline_amused import AmusedPipeline
from diffusers.schedulers import AmusedScheduler


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

device = "cuda"


def main():
    args = ArgumentParser()
    args.add_argument("--model_256", action="store_true")
    args.add_argument("--write_to", type=str, required=False, default=None)
    args.add_argument("--transformer_path", type=str, required=False, default=None)
    args = args.parse_args()

    transformer_path = args.transformer_path
    subfolder = "transformer"

    if transformer_path is None:
        if args.model_256:
            transformer_path = "openMUSE/muse-256"
        else:
            transformer_path = (
                "../research-run-512-checkpoints/research-run-512-with-downsample-checkpoint-554000/unwrapped_model/"
            )
            subfolder = None

    old_transformer = MaskGiTUViT.from_pretrained(transformer_path, subfolder=subfolder)

    old_transformer.to(device)

    old_vae = VQGANModel.from_pretrained("openMUSE/muse-512", subfolder="vae")
    old_vae.to(device)

    vqvae = make_vqvae(old_vae)

    tokenizer = CLIPTokenizer.from_pretrained("openMUSE/muse-512", subfolder="text_encoder")

    text_encoder = CLIPTextModelWithProjection.from_pretrained("openMUSE/muse-512", subfolder="text_encoder")
    text_encoder.to(device)

    transformer = make_transformer(old_transformer, args.model_256)

    scheduler = AmusedScheduler(mask_token_id=old_transformer.config.mask_token_id)

    new_pipe = AmusedPipeline(
        vqvae=vqvae, tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, scheduler=scheduler
    )

    old_pipe = OldPipelineMuse(
        vae=old_vae, transformer=old_transformer, text_encoder=text_encoder, tokenizer=tokenizer
    )
    old_pipe.to(device)

    if args.model_256:
        transformer_seq_len = 256
        orig_size = (256, 256)
    else:
        transformer_seq_len = 1024
        orig_size = (512, 512)

    old_out = old_pipe(
        "dog",
        generator=torch.Generator(device).manual_seed(0),
        transformer_seq_len=transformer_seq_len,
        orig_size=orig_size,
        timesteps=12,
    )[0]

    new_out = new_pipe("dog", generator=torch.Generator(device).manual_seed(0)).images[0]

    old_out = np.array(old_out)
    new_out = np.array(new_out)

    diff = np.abs(old_out.astype(np.float64) - new_out.astype(np.float64))

    # assert diff diff.sum() == 0
    print("skipping pipeline full equivalence check")

    print(f"max diff: {diff.max()}, diff.sum() / diff.size {diff.sum() / diff.size}")

    if args.model_256:
        assert diff.max() <= 3
        assert diff.sum() / diff.size < 0.7
    else:
        assert diff.max() <= 1
        assert diff.sum() / diff.size < 0.4

    if args.write_to is not None:
        new_pipe.save_pretrained(args.write_to)


def make_transformer(old_transformer, model_256):
    args = dict(old_transformer.config)
    force_down_up_sample = args["force_down_up_sample"]

    signature = inspect.signature(UVit2DModel.__init__)

    args_ = {
        "downsample": force_down_up_sample,
        "upsample": force_down_up_sample,
        "block_out_channels": args["block_out_channels"][0],
        "sample_size": 16 if model_256 else 32,
    }

    for s in list(signature.parameters.keys()):
        if s in ["self", "downsample", "upsample", "sample_size", "block_out_channels"]:
            continue

        args_[s] = args[s]

    new_transformer = UVit2DModel(**args_)
    new_transformer.to(device)

    new_transformer.set_attn_processor(AttnProcessor())

    state_dict = old_transformer.state_dict()

    state_dict["cond_embed.linear_1.weight"] = state_dict.pop("cond_embed.0.weight")
    state_dict["cond_embed.linear_2.weight"] = state_dict.pop("cond_embed.2.weight")

    for i in range(22):
        state_dict[f"transformer_layers.{i}.norm1.norm.weight"] = state_dict.pop(
            f"transformer_layers.{i}.attn_layer_norm.weight"
        )
        state_dict[f"transformer_layers.{i}.norm1.linear.weight"] = state_dict.pop(
            f"transformer_layers.{i}.self_attn_adaLN_modulation.mapper.weight"
        )

        state_dict[f"transformer_layers.{i}.attn1.to_q.weight"] = state_dict.pop(
            f"transformer_layers.{i}.attention.query.weight"
        )
        state_dict[f"transformer_layers.{i}.attn1.to_k.weight"] = state_dict.pop(
            f"transformer_layers.{i}.attention.key.weight"
        )
        state_dict[f"transformer_layers.{i}.attn1.to_v.weight"] = state_dict.pop(
            f"transformer_layers.{i}.attention.value.weight"
        )
        state_dict[f"transformer_layers.{i}.attn1.to_out.0.weight"] = state_dict.pop(
            f"transformer_layers.{i}.attention.out.weight"
        )

        state_dict[f"transformer_layers.{i}.norm2.norm.weight"] = state_dict.pop(
            f"transformer_layers.{i}.crossattn_layer_norm.weight"
        )
        state_dict[f"transformer_layers.{i}.norm2.linear.weight"] = state_dict.pop(
            f"transformer_layers.{i}.cross_attn_adaLN_modulation.mapper.weight"
        )

        state_dict[f"transformer_layers.{i}.attn2.to_q.weight"] = state_dict.pop(
            f"transformer_layers.{i}.crossattention.query.weight"
        )
        state_dict[f"transformer_layers.{i}.attn2.to_k.weight"] = state_dict.pop(
            f"transformer_layers.{i}.crossattention.key.weight"
        )
        state_dict[f"transformer_layers.{i}.attn2.to_v.weight"] = state_dict.pop(
            f"transformer_layers.{i}.crossattention.value.weight"
        )
        state_dict[f"transformer_layers.{i}.attn2.to_out.0.weight"] = state_dict.pop(
            f"transformer_layers.{i}.crossattention.out.weight"
        )

        state_dict[f"transformer_layers.{i}.norm3.norm.weight"] = state_dict.pop(
            f"transformer_layers.{i}.ffn.pre_mlp_layer_norm.weight"
        )
        state_dict[f"transformer_layers.{i}.norm3.linear.weight"] = state_dict.pop(
            f"transformer_layers.{i}.ffn.adaLN_modulation.mapper.weight"
        )

        wi_0_weight = state_dict.pop(f"transformer_layers.{i}.ffn.wi_0.weight")
        wi_1_weight = state_dict.pop(f"transformer_layers.{i}.ffn.wi_1.weight")
        proj_weight = torch.concat([wi_1_weight, wi_0_weight], dim=0)
        state_dict[f"transformer_layers.{i}.ff.net.0.proj.weight"] = proj_weight

        state_dict[f"transformer_layers.{i}.ff.net.2.weight"] = state_dict.pop(f"transformer_layers.{i}.ffn.wo.weight")

    if force_down_up_sample:
        state_dict["down_block.downsample.norm.weight"] = state_dict.pop("down_blocks.0.downsample.0.norm.weight")
        state_dict["down_block.downsample.conv.weight"] = state_dict.pop("down_blocks.0.downsample.1.weight")

        state_dict["up_block.upsample.norm.weight"] = state_dict.pop("up_blocks.0.upsample.0.norm.weight")
        state_dict["up_block.upsample.conv.weight"] = state_dict.pop("up_blocks.0.upsample.1.weight")

    state_dict["mlm_layer.layer_norm.weight"] = state_dict.pop("mlm_layer.layer_norm.norm.weight")

    for i in range(3):
        state_dict[f"down_block.res_blocks.{i}.norm.weight"] = state_dict.pop(
            f"down_blocks.0.res_blocks.{i}.norm.norm.weight"
        )
        state_dict[f"down_block.res_blocks.{i}.channelwise_linear_1.weight"] = state_dict.pop(
            f"down_blocks.0.res_blocks.{i}.channelwise.0.weight"
        )
        state_dict[f"down_block.res_blocks.{i}.channelwise_norm.gamma"] = state_dict.pop(
            f"down_blocks.0.res_blocks.{i}.channelwise.2.gamma"
        )
        state_dict[f"down_block.res_blocks.{i}.channelwise_norm.beta"] = state_dict.pop(
            f"down_blocks.0.res_blocks.{i}.channelwise.2.beta"
        )
        state_dict[f"down_block.res_blocks.{i}.channelwise_linear_2.weight"] = state_dict.pop(
            f"down_blocks.0.res_blocks.{i}.channelwise.4.weight"
        )
        state_dict[f"down_block.res_blocks.{i}.cond_embeds_mapper.weight"] = state_dict.pop(
            f"down_blocks.0.res_blocks.{i}.adaLN_modulation.mapper.weight"
        )

        state_dict[f"down_block.attention_blocks.{i}.norm1.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.attn_layer_norm.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn1.to_q.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.attention.query.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn1.to_k.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.attention.key.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn1.to_v.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.attention.value.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn1.to_out.0.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.attention.out.weight"
        )

        state_dict[f"down_block.attention_blocks.{i}.norm2.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.crossattn_layer_norm.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn2.to_q.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.crossattention.query.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn2.to_k.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.crossattention.key.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn2.to_v.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.crossattention.value.weight"
        )
        state_dict[f"down_block.attention_blocks.{i}.attn2.to_out.0.weight"] = state_dict.pop(
            f"down_blocks.0.attention_blocks.{i}.crossattention.out.weight"
        )

        state_dict[f"up_block.res_blocks.{i}.norm.weight"] = state_dict.pop(
            f"up_blocks.0.res_blocks.{i}.norm.norm.weight"
        )
        state_dict[f"up_block.res_blocks.{i}.channelwise_linear_1.weight"] = state_dict.pop(
            f"up_blocks.0.res_blocks.{i}.channelwise.0.weight"
        )
        state_dict[f"up_block.res_blocks.{i}.channelwise_norm.gamma"] = state_dict.pop(
            f"up_blocks.0.res_blocks.{i}.channelwise.2.gamma"
        )
        state_dict[f"up_block.res_blocks.{i}.channelwise_norm.beta"] = state_dict.pop(
            f"up_blocks.0.res_blocks.{i}.channelwise.2.beta"
        )
        state_dict[f"up_block.res_blocks.{i}.channelwise_linear_2.weight"] = state_dict.pop(
            f"up_blocks.0.res_blocks.{i}.channelwise.4.weight"
        )
        state_dict[f"up_block.res_blocks.{i}.cond_embeds_mapper.weight"] = state_dict.pop(
            f"up_blocks.0.res_blocks.{i}.adaLN_modulation.mapper.weight"
        )

        state_dict[f"up_block.attention_blocks.{i}.norm1.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.attn_layer_norm.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn1.to_q.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.attention.query.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn1.to_k.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.attention.key.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn1.to_v.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.attention.value.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn1.to_out.0.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.attention.out.weight"
        )

        state_dict[f"up_block.attention_blocks.{i}.norm2.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.crossattn_layer_norm.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn2.to_q.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.crossattention.query.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn2.to_k.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.crossattention.key.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn2.to_v.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.crossattention.value.weight"
        )
        state_dict[f"up_block.attention_blocks.{i}.attn2.to_out.0.weight"] = state_dict.pop(
            f"up_blocks.0.attention_blocks.{i}.crossattention.out.weight"
        )

    for key in list(state_dict.keys()):
        if key.startswith("up_blocks.0"):
            key_ = "up_block." + ".".join(key.split(".")[2:])
            state_dict[key_] = state_dict.pop(key)

        if key.startswith("down_blocks.0"):
            key_ = "down_block." + ".".join(key.split(".")[2:])
            state_dict[key_] = state_dict.pop(key)

    new_transformer.load_state_dict(state_dict)

    input_ids = torch.randint(0, 10, (1, 32, 32), device=old_transformer.device)
    encoder_hidden_states = torch.randn((1, 77, 768), device=old_transformer.device)
    cond_embeds = torch.randn((1, 768), device=old_transformer.device)
    micro_conds = torch.tensor([[512, 512, 0, 0, 6]], dtype=torch.float32, device=old_transformer.device)

    old_out = old_transformer(input_ids.reshape(1, -1), encoder_hidden_states, cond_embeds, micro_conds)
    old_out = old_out.reshape(1, 32, 32, 8192).permute(0, 3, 1, 2)

    new_out = new_transformer(input_ids, encoder_hidden_states, cond_embeds, micro_conds)

    # NOTE: these differences are solely due to using the geglu block that has a single linear layer of
    # double output dimension instead of two different linear layers
    max_diff = (old_out - new_out).abs().max()
    total_diff = (old_out - new_out).abs().sum()
    print(f"Transformer max_diff: {max_diff} total_diff:  {total_diff}")
    assert max_diff < 0.01
    assert total_diff < 1500

    return new_transformer


def make_vqvae(old_vae):
    new_vae = VQModel(
        act_fn="silu",
        block_out_channels=[128, 256, 256, 512, 768],
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        in_channels=3,
        latent_channels=64,
        layers_per_block=2,
        norm_num_groups=32,
        num_vq_embeddings=8192,
        out_channels=3,
        sample_size=32,
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        mid_block_add_attention=False,
        lookup_from_codebook=True,
    )
    new_vae.to(device)

    # fmt: off

    new_state_dict = {}

    old_state_dict = old_vae.state_dict()

    new_state_dict["encoder.conv_in.weight"] = old_state_dict.pop("encoder.conv_in.weight")
    new_state_dict["encoder.conv_in.bias"]   = old_state_dict.pop("encoder.conv_in.bias")

    convert_vae_block_state_dict(old_state_dict, "encoder.down.0", new_state_dict, "encoder.down_blocks.0")
    convert_vae_block_state_dict(old_state_dict, "encoder.down.1", new_state_dict, "encoder.down_blocks.1")
    convert_vae_block_state_dict(old_state_dict, "encoder.down.2", new_state_dict, "encoder.down_blocks.2")
    convert_vae_block_state_dict(old_state_dict, "encoder.down.3", new_state_dict, "encoder.down_blocks.3")
    convert_vae_block_state_dict(old_state_dict, "encoder.down.4", new_state_dict, "encoder.down_blocks.4")

    new_state_dict["encoder.mid_block.resnets.0.norm1.weight"] = old_state_dict.pop("encoder.mid.block_1.norm1.weight")
    new_state_dict["encoder.mid_block.resnets.0.norm1.bias"]   = old_state_dict.pop("encoder.mid.block_1.norm1.bias")
    new_state_dict["encoder.mid_block.resnets.0.conv1.weight"] = old_state_dict.pop("encoder.mid.block_1.conv1.weight")
    new_state_dict["encoder.mid_block.resnets.0.conv1.bias"]   = old_state_dict.pop("encoder.mid.block_1.conv1.bias")
    new_state_dict["encoder.mid_block.resnets.0.norm2.weight"] = old_state_dict.pop("encoder.mid.block_1.norm2.weight")
    new_state_dict["encoder.mid_block.resnets.0.norm2.bias"]   = old_state_dict.pop("encoder.mid.block_1.norm2.bias")
    new_state_dict["encoder.mid_block.resnets.0.conv2.weight"] = old_state_dict.pop("encoder.mid.block_1.conv2.weight")
    new_state_dict["encoder.mid_block.resnets.0.conv2.bias"]   = old_state_dict.pop("encoder.mid.block_1.conv2.bias")
    new_state_dict["encoder.mid_block.resnets.1.norm1.weight"] = old_state_dict.pop("encoder.mid.block_2.norm1.weight")
    new_state_dict["encoder.mid_block.resnets.1.norm1.bias"]   = old_state_dict.pop("encoder.mid.block_2.norm1.bias")
    new_state_dict["encoder.mid_block.resnets.1.conv1.weight"] = old_state_dict.pop("encoder.mid.block_2.conv1.weight")
    new_state_dict["encoder.mid_block.resnets.1.conv1.bias"]   = old_state_dict.pop("encoder.mid.block_2.conv1.bias")
    new_state_dict["encoder.mid_block.resnets.1.norm2.weight"] = old_state_dict.pop("encoder.mid.block_2.norm2.weight")
    new_state_dict["encoder.mid_block.resnets.1.norm2.bias"]   = old_state_dict.pop("encoder.mid.block_2.norm2.bias")
    new_state_dict["encoder.mid_block.resnets.1.conv2.weight"] = old_state_dict.pop("encoder.mid.block_2.conv2.weight")
    new_state_dict["encoder.mid_block.resnets.1.conv2.bias"]   = old_state_dict.pop("encoder.mid.block_2.conv2.bias")
    new_state_dict["encoder.conv_norm_out.weight"]             = old_state_dict.pop("encoder.norm_out.weight")
    new_state_dict["encoder.conv_norm_out.bias"]               = old_state_dict.pop("encoder.norm_out.bias")
    new_state_dict["encoder.conv_out.weight"]                  = old_state_dict.pop("encoder.conv_out.weight")
    new_state_dict["encoder.conv_out.bias"]                    = old_state_dict.pop("encoder.conv_out.bias")
    new_state_dict["quant_conv.weight"]                        = old_state_dict.pop("quant_conv.weight")
    new_state_dict["quant_conv.bias"]                          = old_state_dict.pop("quant_conv.bias")
    new_state_dict["quantize.embedding.weight"]                = old_state_dict.pop("quantize.embedding.weight")
    new_state_dict["post_quant_conv.weight"]                   = old_state_dict.pop("post_quant_conv.weight")
    new_state_dict["post_quant_conv.bias"]                     = old_state_dict.pop("post_quant_conv.bias")
    new_state_dict["decoder.conv_in.weight"]                   = old_state_dict.pop("decoder.conv_in.weight")
    new_state_dict["decoder.conv_in.bias"]                     = old_state_dict.pop("decoder.conv_in.bias")
    new_state_dict["decoder.mid_block.resnets.0.norm1.weight"] = old_state_dict.pop("decoder.mid.block_1.norm1.weight")
    new_state_dict["decoder.mid_block.resnets.0.norm1.bias"]   = old_state_dict.pop("decoder.mid.block_1.norm1.bias")
    new_state_dict["decoder.mid_block.resnets.0.conv1.weight"] = old_state_dict.pop("decoder.mid.block_1.conv1.weight")
    new_state_dict["decoder.mid_block.resnets.0.conv1.bias"]   = old_state_dict.pop("decoder.mid.block_1.conv1.bias")
    new_state_dict["decoder.mid_block.resnets.0.norm2.weight"] = old_state_dict.pop("decoder.mid.block_1.norm2.weight")
    new_state_dict["decoder.mid_block.resnets.0.norm2.bias"]   = old_state_dict.pop("decoder.mid.block_1.norm2.bias")
    new_state_dict["decoder.mid_block.resnets.0.conv2.weight"] = old_state_dict.pop("decoder.mid.block_1.conv2.weight")
    new_state_dict["decoder.mid_block.resnets.0.conv2.bias"]   = old_state_dict.pop("decoder.mid.block_1.conv2.bias")
    new_state_dict["decoder.mid_block.resnets.1.norm1.weight"] = old_state_dict.pop("decoder.mid.block_2.norm1.weight")
    new_state_dict["decoder.mid_block.resnets.1.norm1.bias"]   = old_state_dict.pop("decoder.mid.block_2.norm1.bias")
    new_state_dict["decoder.mid_block.resnets.1.conv1.weight"] = old_state_dict.pop("decoder.mid.block_2.conv1.weight")
    new_state_dict["decoder.mid_block.resnets.1.conv1.bias"]   = old_state_dict.pop("decoder.mid.block_2.conv1.bias")
    new_state_dict["decoder.mid_block.resnets.1.norm2.weight"] = old_state_dict.pop("decoder.mid.block_2.norm2.weight")
    new_state_dict["decoder.mid_block.resnets.1.norm2.bias"]   = old_state_dict.pop("decoder.mid.block_2.norm2.bias")
    new_state_dict["decoder.mid_block.resnets.1.conv2.weight"] = old_state_dict.pop("decoder.mid.block_2.conv2.weight")
    new_state_dict["decoder.mid_block.resnets.1.conv2.bias"]   = old_state_dict.pop("decoder.mid.block_2.conv2.bias")

    convert_vae_block_state_dict(old_state_dict, "decoder.up.0", new_state_dict, "decoder.up_blocks.4")
    convert_vae_block_state_dict(old_state_dict, "decoder.up.1", new_state_dict, "decoder.up_blocks.3")
    convert_vae_block_state_dict(old_state_dict, "decoder.up.2", new_state_dict, "decoder.up_blocks.2")
    convert_vae_block_state_dict(old_state_dict, "decoder.up.3", new_state_dict, "decoder.up_blocks.1")
    convert_vae_block_state_dict(old_state_dict, "decoder.up.4", new_state_dict, "decoder.up_blocks.0")

    new_state_dict["decoder.conv_norm_out.weight"] = old_state_dict.pop("decoder.norm_out.weight")
    new_state_dict["decoder.conv_norm_out.bias"]   = old_state_dict.pop("decoder.norm_out.bias")
    new_state_dict["decoder.conv_out.weight"]      = old_state_dict.pop("decoder.conv_out.weight")
    new_state_dict["decoder.conv_out.bias"]        = old_state_dict.pop("decoder.conv_out.bias")

    # fmt: on

    assert len(old_state_dict.keys()) == 0

    new_vae.load_state_dict(new_state_dict)

    input = torch.randn((1, 3, 512, 512), device=device)
    input = input.clamp(-1, 1)

    old_encoder_output = old_vae.quant_conv(old_vae.encoder(input))
    new_encoder_output = new_vae.quant_conv(new_vae.encoder(input))
    assert (old_encoder_output == new_encoder_output).all()

    old_decoder_output = old_vae.decoder(old_vae.post_quant_conv(old_encoder_output))
    new_decoder_output = new_vae.decoder(new_vae.post_quant_conv(new_encoder_output))

    # assert (old_decoder_output == new_decoder_output).all()
    print("kipping vae decoder equivalence check")
    print(f"vae decoder diff {(old_decoder_output - new_decoder_output).float().abs().sum()}")

    old_output = old_vae(input)[0]
    new_output = new_vae(input)[0]

    # assert (old_output == new_output).all()
    print("skipping full vae equivalence check")
    print(f"vae full diff {(old_output - new_output).float().abs().sum()}")

    return new_vae


def convert_vae_block_state_dict(old_state_dict, prefix_from, new_state_dict, prefix_to):
    # fmt: off

    new_state_dict[f"{prefix_to}.resnets.0.norm1.weight"]             = old_state_dict.pop(f"{prefix_from}.block.0.norm1.weight")
    new_state_dict[f"{prefix_to}.resnets.0.norm1.bias"]               = old_state_dict.pop(f"{prefix_from}.block.0.norm1.bias")
    new_state_dict[f"{prefix_to}.resnets.0.conv1.weight"]             = old_state_dict.pop(f"{prefix_from}.block.0.conv1.weight")
    new_state_dict[f"{prefix_to}.resnets.0.conv1.bias"]               = old_state_dict.pop(f"{prefix_from}.block.0.conv1.bias")
    new_state_dict[f"{prefix_to}.resnets.0.norm2.weight"]             = old_state_dict.pop(f"{prefix_from}.block.0.norm2.weight")
    new_state_dict[f"{prefix_to}.resnets.0.norm2.bias"]               = old_state_dict.pop(f"{prefix_from}.block.0.norm2.bias")
    new_state_dict[f"{prefix_to}.resnets.0.conv2.weight"]             = old_state_dict.pop(f"{prefix_from}.block.0.conv2.weight")
    new_state_dict[f"{prefix_to}.resnets.0.conv2.bias"]               = old_state_dict.pop(f"{prefix_from}.block.0.conv2.bias")

    if f"{prefix_from}.block.0.nin_shortcut.weight" in old_state_dict:
        new_state_dict[f"{prefix_to}.resnets.0.conv_shortcut.weight"]     = old_state_dict.pop(f"{prefix_from}.block.0.nin_shortcut.weight")
        new_state_dict[f"{prefix_to}.resnets.0.conv_shortcut.bias"]       = old_state_dict.pop(f"{prefix_from}.block.0.nin_shortcut.bias")

    new_state_dict[f"{prefix_to}.resnets.1.norm1.weight"]             = old_state_dict.pop(f"{prefix_from}.block.1.norm1.weight")
    new_state_dict[f"{prefix_to}.resnets.1.norm1.bias"]               = old_state_dict.pop(f"{prefix_from}.block.1.norm1.bias")
    new_state_dict[f"{prefix_to}.resnets.1.conv1.weight"]             = old_state_dict.pop(f"{prefix_from}.block.1.conv1.weight")
    new_state_dict[f"{prefix_to}.resnets.1.conv1.bias"]               = old_state_dict.pop(f"{prefix_from}.block.1.conv1.bias")
    new_state_dict[f"{prefix_to}.resnets.1.norm2.weight"]             = old_state_dict.pop(f"{prefix_from}.block.1.norm2.weight")
    new_state_dict[f"{prefix_to}.resnets.1.norm2.bias"]               = old_state_dict.pop(f"{prefix_from}.block.1.norm2.bias")
    new_state_dict[f"{prefix_to}.resnets.1.conv2.weight"]             = old_state_dict.pop(f"{prefix_from}.block.1.conv2.weight")
    new_state_dict[f"{prefix_to}.resnets.1.conv2.bias"]               = old_state_dict.pop(f"{prefix_from}.block.1.conv2.bias")

    if f"{prefix_from}.downsample.conv.weight" in old_state_dict:
        new_state_dict[f"{prefix_to}.downsamplers.0.conv.weight"]         = old_state_dict.pop(f"{prefix_from}.downsample.conv.weight")
        new_state_dict[f"{prefix_to}.downsamplers.0.conv.bias"]           = old_state_dict.pop(f"{prefix_from}.downsample.conv.bias")

    if f"{prefix_from}.upsample.conv.weight" in old_state_dict:
        new_state_dict[f"{prefix_to}.upsamplers.0.conv.weight"]         = old_state_dict.pop(f"{prefix_from}.upsample.conv.weight")
        new_state_dict[f"{prefix_to}.upsamplers.0.conv.bias"]           = old_state_dict.pop(f"{prefix_from}.upsample.conv.bias")

    if f"{prefix_from}.block.2.norm1.weight" in old_state_dict:
        new_state_dict[f"{prefix_to}.resnets.2.norm1.weight"]             = old_state_dict.pop(f"{prefix_from}.block.2.norm1.weight")
        new_state_dict[f"{prefix_to}.resnets.2.norm1.bias"]               = old_state_dict.pop(f"{prefix_from}.block.2.norm1.bias")
        new_state_dict[f"{prefix_to}.resnets.2.conv1.weight"]             = old_state_dict.pop(f"{prefix_from}.block.2.conv1.weight")
        new_state_dict[f"{prefix_to}.resnets.2.conv1.bias"]               = old_state_dict.pop(f"{prefix_from}.block.2.conv1.bias")
        new_state_dict[f"{prefix_to}.resnets.2.norm2.weight"]             = old_state_dict.pop(f"{prefix_from}.block.2.norm2.weight")
        new_state_dict[f"{prefix_to}.resnets.2.norm2.bias"]               = old_state_dict.pop(f"{prefix_from}.block.2.norm2.bias")
        new_state_dict[f"{prefix_to}.resnets.2.conv2.weight"]             = old_state_dict.pop(f"{prefix_from}.block.2.conv2.weight")
        new_state_dict[f"{prefix_to}.resnets.2.conv2.bias"]               = old_state_dict.pop(f"{prefix_from}.block.2.conv2.bias")

    # fmt: on


if __name__ == "__main__":
    main()
