'''
Author: Juncfang
Date: 2023-02-05 08:28:56
LastEditTime: 2023-04-10 20:20:49
LastEditors: Juncfang
Description: 
FilePath: /diffusers_fork/personal_workspace/inference.py
 
'''
import os
import argparse
import torch
import json
import random
from datetime import datetime
from diffusers import StableDiffusionPipeline
from diffusers import UniPCMultistepScheduler, DDIMScheduler, PNDMScheduler, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import sys
_project_dir_ = os.path.dirname(os.path.dirname(__file__))
sys.path.append(_project_dir_)
from personal_workspace.code.cutoff_diffusers import cutoff_text_encoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default=None)
    parser.add_argument('--image_num', type=int, default=10)
    parser.add_argument('--base_seed', type=int, default=0)
    parser.add_argument('--sampler', type=int, default=0)
    parser.add_argument('--guidance_scale', type=float, default=7)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--sampler_method', type=str, default="", choices=["DDIM", "DDPM", "UniPC", "PNDM", "DF"], 
                        help="Please choice form the list, Otherwise use default sampler method ")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        torch_dtype=torch.float16)
    pipe.to(f"cuda:{args.gpu_id}")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
         subfolder="tokenizer",
        torch_dtype=torch.float16
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    text_encoder.to(f"cuda:{args.gpu_id}")
    # change sampler_method
    print(f"Using {args.sampler_method} as sampler method ...... ")
    if args.sampler_method == "UniPC":
        sampler_method = UniPCMultistepScheduler()
    elif args.sampler_method == "DDIM":
        sampler_method = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    elif args.sampler_method == "DDPM":
        sampler_method = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    elif args.sampler_method == "PNDM":
        # sampler_method = PNDMScheduler(
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     beta_start=0.00085,
        #     num_train_timesteps=1000,
        #     prediction_type="epsilon",
        #     set_alpha_to_one=False,
        #     skip_prk_steps=True,
        #     steps_offset=1,
        #     trained_betas=None
        # )
        sampler_method = PNDMScheduler()
    else:
        sampler_method = None
    if sampler_method is not None:
        pipe.scheduler = sampler_method
    
    seed_ = args.base_seed
    prompt_brief = args.prompt[:20] # will be truncated if lenght less than 20
    now_str = datetime.now().strftime("%Y-%m-%dT%H-%M")
    image_output_dir = os.path.join(args.output_dir, f"{now_str}-{prompt_brief}")
    os.makedirs(image_output_dir, exist_ok=True)
    for i in tqdm(range(args.image_num)):
        seed = seed_ + i  if seed_ != -1 else random.randint(0, 1000000)
        tensor = cutoff_text_encoder(
            [args.prompt], 
            text_encoder, 
            tokenizer, 
            targets=['red', 'blue', 'white', 'green', 'yellow', 'pink', 
                     'black', 'gray', 'orange', 'purple', 'cyan', 'brown']
            )
        tk = tokenizer(
            args.negative_prompt, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt").to(text_encoder.device)
        tensor_neg = text_encoder(tk["input_ids"])[0]
        image = pipe(
            prompt_embeds=tensor,
            negative_prompt_embeds=tensor_neg,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device='cuda').manual_seed(seed),
        ).images[0]
        image.save(f"{image_output_dir}/{i}_seed_{seed}.png")
    # write params
    params = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "base_seed": args.base_seed,
        "sampler_method": args.sampler_method
    }
    json_path = os.path.join(image_output_dir, "params.json")
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == '__main__':
    main()
