import argparse
import os
import re

import torch
from safetensors.torch import save_file
from tqdm import tqdm

def ema_scope(model):
    model_ema = LitEma(model)
    model_ema.store(model.parameters())
    model_ema.copy_to(model)
    try:
        yield None
    finally:
        model_ema.restore(model.parameters())

def replace1_string_ldm3d2sd(input_string):
    # Define the regular expression pattern
    pattern = r"\.(\d+)\.block"

    # Define the replacement string
    replacement = r"_blocks.\1.resnets"

    # Perform the replacement using re.sub()
    output_string = re.sub(pattern, replacement, input_string)
    return output_string


def replace_digits0(string):
    digits = "0123456789"
    result = ""
    digit_count = 0

    for char in string:
        if char in digits:
            if digit_count == 0:
                new_digit = abs(int(char) - 3)
            # elif digit_count == 1:
            # new_digit = abs(int(char) - 2)
            else:
                new_digit = int(char)

            result += str(new_digit)
            digit_count += 1
        else:
            result += char

    return result


##VAE##
###change name ldm3d keys to sd keys

# ldm3d_vae = torch.load("/home/estellea/LDM3D_checkpoint/ldm3d/vae/diffusion_pytorch_model_gabi_ckpt.bin")
# sd_hf_vae = torch.load("/home/estellea/LDM3D_checkpoint/stable-diffusion-v1-4/vae/diffusion_pytorch_model.bin")
# sd_compvis_vae = torch.load("/export/share/projects/mcai/checkpoints/stable-diffusion/autoencoders/kl-f8/model.ckpt")['state_dict']
# new_path_ldm3d = "/home/estellea/LDM3D_checkpoint/ldm3d-v1/vae/diffusion_pytorch_model.bin"
# ldm3d_vae_converted = torch.load(new_path)

# checker = 0
# for k, v in ldm3d_vae_converted.items():
#     if k not in sd_hf_vae:
#         checker = 1
#     if sd_hf_vae[k].shape != v.shape:
#         checker =1


def convert_vae_weights(ldm3d_vae, sd_hf_vae, new_path):
    print("Start converting VAE...")
    checker = []
    new_ldm3d_vae = {}
    for k, v in ldm3d_vae.items():
        if "first_stage" in k:
            new_k = k.replace("first_stage_model.", "")
            if new_k not in sd_hf_vae:
                if "attn" in new_k:
                    new_k = new_k.replace(".q.", ".query.")
                    new_k = new_k.replace(".k.", ".key.")
                    new_k = new_k.replace(".v.", ".value.")
                    new_k = new_k.replace(".norm.", ".group_norm.")
                    new_k = new_k.replace(".proj_out.", ".proj_attn.")
                    if "mid" in new_k:
                        new_k = new_k.replace("mid.attn_1", "mid_block.attentions.0")
                if "decoder" in k:
                    if "mid.block" in new_k:
                        new_k = new_k.replace("mid.block_", "mid_block.resnets.")
                        new_k = new_k.replace("resnets.1", "resnets.0")
                        new_k = new_k.replace("resnets.2", "resnets.1")
                    if ".up." in new_k:
                        new_k = new_k.replace(".block.", ".resnets.")
                        new_k = new_k.replace(".up.", ".up_blocks.")
                        new_k = replace_digits0(new_k)
                        new_k = new_k.replace(".upsample.", ".upsamplers.0.")
                    if "shortcut" in new_k:
                        new_k = new_k.replace("1.res", "3.res")
                        new_k = new_k.replace("0.res", "2.res")
                if "encoder" in k:
                    if "mid.block" in new_k:
                        new_k = new_k.replace("mid.block_", "mid_block.resnets.")
                        new_k = new_k.replace("resnets.1", "resnets.0")
                        new_k = new_k.replace("resnets.2", "resnets.1")
                    if ".down." in new_k:
                        new_k = new_k.replace(".block.", ".resnets.")
                        new_k = new_k.replace(".down.", ".down_blocks.")
                    if "downsample" in new_k:
                        new_k = new_k.replace(".down.", ".down_blocks.")
                        new_k = new_k.replace(".downsample.", ".downsamplers.0.")
                if "norm_out" in k:
                    new_k = new_k.replace("norm_out", "conv_norm_out")

                if "shortcut" in new_k:
                    new_k = new_k.replace("nin", "conv")
                if new_k in sd_hf_vae:
                    new_ldm3d_vae[new_k] = v
                    if new_ldm3d_vae[new_k].shape != sd_hf_vae[new_k].shape:
                        assert new_ldm3d_vae[new_k].numel() == sd_hf_vae[new_k].numel()
                        sd_hf_vae[new_k].shape
                        new_v = v.squeeze()
                        new_ldm3d_vae[new_k] = new_v
                    #    new_ldm3d_vae_ema[new_k] = v_ema.squeeze()
                    assert new_ldm3d_vae[new_k].shape == sd_hf_vae[new_k].shape
                else:
                    checker.append(k)
            else:
                new_ldm3d_vae[new_k] = v
              #  new_ldm3d_vae_ema[new_k] = v_ema
    assert len(checker) == 0
    assert len(new_ldm3d_vae) == len(sd_hf_vae), f"Size new: {len(new_ldm3d_vae)}, size to be: {len(sd_hf_vae)}"
    torch.save(new_ldm3d_vae, new_path)
    print(f"Model successfully saved on {new_path}")
    return new_ldm3d_vae


###UNET###
def get_first_number(string):
    pattern = r"\d+"  # Regular expression pattern to match one or more digits
    matches = re.findall(pattern, string)
    return int(matches[0])


def replace_digits(fullstring, substring, num_fix=0, reduce_r=None, reduce_q=None):
    d = get_first_number(fullstring)
    q = d // 3
    r = d % 3
    if reduce_r is not None:
        r -= reduce_r
    if reduce_q is not None:
        q -= reduce_q
    return fullstring.replace(f"{d}.{num_fix}", f"{q}.{substring}.{r}")


def convert_unet_weights(ldm3d_unet, sd_hf_unet, new_path):
    print("Start converting Unet...")
    checker = []
    new_ldm3d_unet = {}
    new_ldm3d_unet_ema = {}
    new_path_ema = new_path.replace(".bin", "_ema.bin")
    for k, v in tqdm(ldm3d_unet.items()):
        if "model.diffusion" in k:
            if k not in sd_hf_unet:
                new_k = k.replace("model.diffusion_model.", "")
                new_k = new_k.replace("input_blocks", "down_blocks")
                new_k = new_k.replace("middle_block", "mid_block")
                new_k = new_k.replace("output_blocks", "up_blocks")
                new_k = new_k.replace("emb_layers", "time_emb_proj")
            if "up" in new_k:
                if "emb" in new_k:
                    new_k = replace_digits(new_k, "resnets")
                    new_k = new_k.replace("proj.1", "proj")
                if "transformer_blocks" in new_k:
                    new_k = replace_digits(new_k, "attentions", num_fix=1)
                elif "layer" in new_k or "time_emb" in new_k or "skip" in new_k:
                    new_k = replace_digits(new_k, "resnets", num_fix=0, reduce_r=False)
                    new_k = new_k.replace("skip_connection", "conv_shortcut")
                    new_k = new_k.replace("time_emb_proj.1", "time_emb_proj")
                    new_k = new_k.replace("in_layers.0", "norm1")
                    new_k = new_k.replace("out_layers.0", "norm2")
                    new_k = new_k.replace("in_layers.2", "conv1")
                    new_k = new_k.replace("out_layers.3", "conv2")
                elif "conv" in new_k:
                    new_k = new_k.replace("2.1", "0.upsamplers.0")
                    new_k = new_k.replace("5.2", "1.upsamplers.0")
                    new_k = new_k.replace("8.2", "2.upsamplers.0")
                else:
                    new_k = replace_digits(new_k, "attentions", num_fix=1)
            elif "mid" in new_k:
                if "transformer_blocks" in new_k:
                    new_k = new_k.replace(".1.transformer_blocks.0", ".attentions.0.transformer_blocks.0")
                elif "layer" in new_k or "time_emb" in new_k or "skip" in new_k:
                    new_k = new_k.replace("mid_block", "mid_block.resnets")
                    new_k = new_k.replace("resnets.2", "resnets.1")
                    new_k = new_k.replace("skip_connection", "conv_shortcut")
                    new_k = new_k.replace("time_emb_proj.1", "time_emb_proj")
                    new_k = new_k.replace("in_layers.0", "norm1")
                    new_k = new_k.replace("out_layers.0", "norm2")
                    new_k = new_k.replace("in_layers.2", "conv1")
                    new_k = new_k.replace("out_layers.3", "conv2")
                else:
                    new_k = new_k.replace(".1.", ".attentions.0.")
            elif "down" in new_k:
                if "transformer_blocks" in new_k:
                    new_k = replace_digits(new_k, "attentions", num_fix=1, reduce_r=1)
                if "layer" in new_k or "time_emb" in new_k or "skip" in new_k:
                    new_k = replace_digits(new_k, "resnets", num_fix=0, reduce_r=1)
                    new_k = new_k.replace("skip_connection", "conv_shortcut")
                    new_k = new_k.replace("time_emb_proj.1", "time_emb_proj")
                    new_k = new_k.replace("in_layers.0", "norm1")
                    new_k = new_k.replace("out_layers.0", "norm2")
                    new_k = new_k.replace("in_layers.2", "conv1")
                    new_k = new_k.replace("out_layers.3", "conv2")
                else:
                    if "op" in new_k:
                        new_k = replace_digits(new_k, "downsamplers", num_fix=0, reduce_q=1)
                        new_k = new_k.replace("op", "conv")
                    elif new_k == "down_blocks.0.0.weight":
                        new_k = "conv_in.weight"
                    elif new_k == "down_blocks.0.0.bias":
                        new_k = "conv_in.bias"
                    else:
                        new_k = replace_digits(new_k, "attentions", num_fix=1, reduce_r=1)
            else:
                new_k = new_k.replace("time_embed.0", "time_embedding.linear_1")
                new_k = new_k.replace("time_embed.2", "time_embedding.linear_2")
                new_k = new_k.replace("out.0", "conv_norm_out")
                new_k = new_k.replace("out.2", "conv_out")
            if new_k in sd_hf_unet:
                assert sd_hf_unet[new_k].shape == v.shape
                v_ema = ldm3d_unet[f"model_ema.{k.replace('model.diff', 'diff').replace('.', '')}"]
                new_ldm3d_unet[new_k] = v
                new_ldm3d_unet_ema[new_k] = v_ema

            else:
                checker.append((new_k, v.shape))
    assert len(checker) == 0
    assert len(new_ldm3d_unet) == len(sd_hf_unet)
    assert len(new_ldm3d_unet_ema) == len(new_ldm3d_unet)
    torch.save(new_ldm3d_unet, new_path)
    torch.save(new_ldm3d_unet_ema, new_path_ema)
    print(f"Model successfully saved on {new_path} and {new_path_ema}")
    return new_ldm3d_unet, new_ldm3d_unet_ema


# print(len(sd_hf_unet), len(new_ldm3d_unet), len(temp))

# temp_ld = []
# for k, v in ldm3d_unet.items():
#     if "model.diffusion" in k:
#         if k not in sd_hf_unet:
#             new_k = k.replace("model.diffusion_model.", "")
#             new_k = new_k.replace("input_blocks", "down_blocks")
#             new_k = new_k.replace("middle_block", "mid_block")
#             new_k = new_k.replace("output_blocks", "up_blocks")
#             new_k = new_k.replace("emb_layers", "time_emb_proj")
#         if "down" not in new_k and "mid" not in new_k and "up" not in new_k:
#             new_k = new_k.replace("time_embed.0", "time_embedding.linear_1")
#             new_k = new_k.replace("time_embed.2", "time_embedding.linear_2")
#             new_k = new_k.replace("out.0", "conv_norm_out")
#             new_k = new_k.replace("out.2", "conv_out")
#             if new_k not in sd_hf_unet:
#                 temp_ld.append((new_k, v.shape))

# temp_sd = []
# for k, v in sd_hf_unet.items():
#     if "down" not in k and "up" not in k and "mid" not in k:
#         temp_sd.append((k, v.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--new_folder_name",
        default="/home/estellea/LDM3D_checkpoint/ldm3d-v2",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether or not using ema from the original ckpt"
    )
    args = parser.parse_args()
    os.makedirs(args.new_folder_name)
    os.makedirs(os.path.join(args.new_folder_name, "unet"))
    os.makedirs(os.path.join(args.new_folder_name, "vae"))

    new_path_unet = os.path.join(args.new_folder_name, "unet/diffusion_pytorch_model.bin")
    new_path_vae = os.path.join(args.new_folder_name, "vae/diffusion_pytorch_model.bin")
    # original_path = "/export/share/projects/mcai/checkpoints/stable-diffusion/logs/ldm3d_20k_dataset/sd_1_5/bs64/2023-06-05T02-34-23_depth-color-finetuning_aesthetics_sd20k_bs64_lre5_AE_rgbd_4ch_op1/checkpoints/epoch=000249.ckpt"

    old_format = torch.load(args.checkpoint_path)
    ldm3d_before_conversion = old_format["state_dict"]
    sd_hf_unet = torch.load(
        "/export/share/projects/mcai/ldm3d/ckpts/hf/ldm3d-v1/unet/before_conversion_diffusion_pytorch_model.bin"
    )
    sd_hf_vae = torch.load(
        "/export/share/projects/mcai/ldm3d/ckpts/hf/ldm3d-v1/vae/before_conversion_diffusion_pytorch_model.bin"
    )

    new_ldm3d_vae = convert_vae_weights(ldm3d_before_conversion, sd_hf_vae, new_path_vae)
    new_ldm3d_unet, new_ldm3d_unet_ema = convert_unet_weights(ldm3d_before_conversion, sd_hf_unet, new_path_unet)

    print("Saving safetensors")
    save_file(new_ldm3d_vae, os.path.join(args.new_folder_name, "vae/diffusion_pytorch_model.safetensors"))
    save_file(new_ldm3d_unet, os.path.join(args.new_folder_name, "unet/diffusion_pytorch_model.safetensors"))
    save_file(new_ldm3d_unet_ema, os.path.join(args.new_folder_name, "unet/diffusion_pytorch_model.safetensors"))

