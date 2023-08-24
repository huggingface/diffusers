import argparse
from safetensors.torch import save_file, load_file


# sdxl, hf
text_encoder_conversion_map = [
    ("te1", "text_encoder"),
    ("lora_", "lora_linear_layer.")
]

text_encoder_2_conversion_map = [
    ("te2", "text_encoder_2"),
    ("lora_", "lora_linear_layer."),
    ("_encoder_layers_", ".encoder.layers."),
    ("_self_attn_", ".self_attn."),
]

unet_conversion_map_layer = []
for i in range(3):
    # loop over downblocks/upblocks
    for j in range(2):
        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(4):
        if i < 2:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

unet_conversion_map_layer.append(("_", ".processor."))
unet_conversion_map_layer.append((".lora_", "_lora."))
unet_conversion_map_layer.append(("to_out_0", "to_out"))

common_conversion_map = [
    ("_", "."),
    (".lora_down.weight", "_lora_down_weight"),
    (".lora_up.weight", "_lora_up_weight"),
]


def convert(hf_ckpt):
    converted_ckpt = dict()

    for key in hf_ckpt:
        if key.startswith("text_encoder."):
            component_convention_map = text_encoder_conversion_map
        elif key.startswith("text_encoder_2."):
            component_convention_map = text_encoder_2_conversion_map
        elif key.startswith("unet."):
            component_convention_map = unet_conversion_map_layer
        else:
            raise RuntimeError(f"Unknown key: {key}")

        new_key = key
        for sd_part, hf_part in component_convention_map:
            new_key = new_key.replace(hf_part, sd_part)

        for sd_part, hf_part in common_conversion_map:
            new_key = new_key.replace(hf_part, sd_part)

        new_key = "lora_" + new_key

        converted_ckpt[new_key] = hf_ckpt[key]

    return converted_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")

    args = parser.parse_args()

    assert args.model_path is not None, "Must provide a model path!"

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"

    ckpt_sdxl_lora = load_file(args.model_path)
    state_dict = convert(ckpt_sdxl_lora)

    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
    save_file(state_dict, args.checkpoint_path)
