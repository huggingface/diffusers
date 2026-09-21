import argparse

import torch

from diffusers import HunyuanDiT2DModel


def main(args):
    state_dict = torch.load(args.pt_checkpoint_path, map_location="cpu")

    if args.load_key != "none":
        try:
            state_dict = state_dict[args.load_key]
        except KeyError:
            raise KeyError(
                f"{args.load_key} not found in the checkpoint.Please load from the following keys:{state_dict.keys()}"
            )

    device = "cuda"
    model_config = HunyuanDiT2DModel.load_config("Tencent-Hunyuan/HunyuanDiT-Diffusers", subfolder="transformer")
    model_config["use_style_cond_and_image_meta_size"] = (
        args.use_style_cond_and_image_meta_size
    )  ### version <= v1.1: True; version >= v1.2: False

    model = HunyuanDiT2DModel.from_single_file(state_dict, config=model_config).to(device)

    from diffusers import HunyuanDiTPipeline

    if args.use_style_cond_and_image_meta_size:
        pipe = HunyuanDiTPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers", transformer=model, torch_dtype=torch.float32
        )
    else:
        pipe = HunyuanDiTPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", transformer=model, torch_dtype=torch.float32
        )
    pipe.to("cuda")
    pipe.to(dtype=torch.float32)

    if args.save:
        pipe.save_pretrained(args.output_checkpoint_path)

    # ### NOTE: HunyuanDiT supports both Chinese and English inputs
    prompt = "一个宇航员在骑马"
    # prompt = "An astronaut riding a horse"
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = pipe(
        height=1024, width=1024, prompt=prompt, generator=generator, num_inference_steps=25, guidance_scale=5.0
    ).images[0]

    image.save("img.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted pipeline or not."
    )
    parser.add_argument(
        "--pt_checkpoint_path", default=None, type=str, required=True, help="Path to the .pt pretrained model."
    )
    parser.add_argument(
        "--output_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output converted diffusers pipeline.",
    )
    parser.add_argument(
        "--load_key", default="none", type=str, required=False, help="The key to load from the pretrained .pt file"
    )
    parser.add_argument(
        "--use_style_cond_and_image_meta_size",
        type=bool,
        default=False,
        help="version <= v1.1: True; version >= v1.2: False",
    )

    args = parser.parse_args()
    main(args)
