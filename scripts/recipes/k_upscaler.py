import argparse

import huggingface_hub
import k_diffusion as K
import torch

from diffusers import UNet2DConditionModel
from diffusers.loaders.conversion import get_conversion


UPSCALER_REPO = "pcuenq/k-upscaler"


def unet_to_diffusers_checkpoint(model, checkpoint):
    return get_conversion("UNet2DConditionModel", dict(model.config)).to_diffusers(checkpoint)


def unet_model_from_original_config(original_config):
    in_channels = original_config["input_channels"] + original_config["unet_cond_dim"]
    out_channels = original_config["input_channels"] + (1 if original_config["has_variance"] else 0)

    block_out_channels = original_config["channels"]

    assert len(set(original_config["depths"])) == 1, (
        "UNet2DConditionModel currently do not support blocks with different number of layers"
    )
    layers_per_block = original_config["depths"][0]

    class_labels_dim = original_config["mapping_cond_dim"]
    cross_attention_dim = original_config["cross_cond_dim"]

    attn1_types = []
    attn2_types = []
    for s, c in zip(original_config["self_attn_depths"], original_config["cross_attn_depths"]):
        if s:
            a1 = "self"
            a2 = "cross" if c else None
        elif c:
            a1 = "cross"
            a2 = None
        else:
            a1 = None
            a2 = None
        attn1_types.append(a1)
        attn2_types.append(a2)

    unet = UNet2DConditionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=("KDownBlock2D", "KCrossAttnDownBlock2D", "KCrossAttnDownBlock2D", "KCrossAttnDownBlock2D"),
        mid_block_type=None,
        up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        act_fn="gelu",
        norm_num_groups=None,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=64,
        time_cond_proj_dim=class_labels_dim,
        resnet_time_scale_shift="scale_shift",
        time_embedding_type="fourier",
        timestep_post_act="gelu",
        conv_in_kernel=1,
        conv_out_kernel=1,
    )

    return unet


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_config_path = huggingface_hub.hf_hub_download(UPSCALER_REPO, "config_laion_text_cond_latent_upscaler_2.json")
    orig_weights_path = huggingface_hub.hf_hub_download(
        UPSCALER_REPO, "laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
    )
    print(f"loading original model configuration from {orig_config_path}")
    print(f"loading original model checkpoint from {orig_weights_path}")

    print("converting to diffusers unet")
    orig_config = K.config.load_config(open(orig_config_path))["model"]
    model = unet_model_from_original_config(orig_config)

    orig_checkpoint = torch.load(orig_weights_path, map_location=device)["model_ema"]
    converted_checkpoint = unet_to_diffusers_checkpoint(model, orig_checkpoint)

    model.load_state_dict(converted_checkpoint, strict=True)
    model.save_pretrained(args.dump_path)
    print(f"saving converted unet model in {args.dump_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
