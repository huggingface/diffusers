'''
Author: Juncfang
Date: 2023-02-05 08:28:56
LastEditTime: 2023-02-06 10:16:02
LastEditors: Juncfang
Description: 
FilePath: /diffusers_fork/personal_workspace/finetune/inference.py
 
'''
import os
import argparse
import torch
import json
from datetime import datetime
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default=None)
    parser.add_argument('--image_num', type=int, default=10)
    parser.add_argument('--base_seed', type=int, default=0)
    parser.add_argument('--guidance_scale', type=float, default=7)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--gpu_id', type=str, default='0')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16)
    pipe.to(f"cuda:{args.gpu_id}")
    seed = args.base_seed
    prompt_brief = args.prompt[:20] # will be truncated if lenght less than 20
    now_str = datetime.now().strftime("%Y-%m-%dT%H-%M")
    image_output_dir = os.path.join(args.output_dir, f"{now_str}-{prompt_brief}")
    os.makedirs(image_output_dir, exist_ok=True)
    for i in tqdm(range(args.image_num)):
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device='cuda').manual_seed(seed + i),
        ).images[0]
        image.save(f"{image_output_dir}/{i}.png")
    # write params
    params = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "base_seed": args.base_seed,
    }
    json_path = os.path.join(image_output_dir, "params.json")
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == '__main__':
    main()
