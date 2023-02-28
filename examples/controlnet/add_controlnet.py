#!/usr/bin/env python
import argparse
import os
from diffusers import UNet2DConditionModel, ControlNetModel

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Add controlnet to existing model by copying unet weights.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    device_map = 'cpu'

    try:
        config, unused_kwargs = ControlNetModel.load_config(
            args.pretrained_model_path,
            return_unused_kwargs=True,
            local_files_only=True,
            revision=args.revision,
            subfolder="controlnet",
            device_map=device_map,
        )
        print("Model already has controlnet. Doing nothing.")
        return
    except OSError:
        pass

    config, unused_kwargs = UNet2DConditionModel.load_config(
        args.pretrained_model_path,
        return_unused_kwargs=True,
        local_files_only=True,
        revision=args.revision,
        subfolder="unet",
        device_map=device_map,
    )

    config["_class_name"] = 'ControlNetModel'
    config.pop("conv_out_kernel", None)
    config.pop("out_channels", None)
    config.pop("up_block_types", None)
    controlnet = ControlNetModel.from_config(config, **unused_kwargs)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_path, subfolder="unet", revision=args.revision
    )

    unet = unet.state_dict()
    temp_state = controlnet.state_dict()

    for k in temp_state.keys():
        if k in unet:
            temp_state[k] = unet[k].clone()
        else:
            if 'controlnet' not in k:
                print('Not found in unet:', k)

    controlnet.load_state_dict(temp_state, strict=True)
    controlnet.save_pretrained(os.path.join(args.pretrained_model_path, 'controlnet'))
    print('Done')

if __name__ == "__main__":
    args = parse_args()
    main(args)
