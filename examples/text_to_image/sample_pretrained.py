from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
from utils import save_image, concat_images_in_square_grid
import argparse

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="stabilityai/stable-diffusion-2", help='pretrained model')
    parser.add_argument('--prompts', nargs='+', type=str, help='edit prompt')
    parser.add_argument('--num_images', type=int, default=30, help='number of images')
    parser.add_argument('--output_dir', type=str, default="/scratch/mp5847/diffusers_ckpt/output", help='output directory')
    
    #create a store_true argument
    parser.add_argument('--create_metadata', action='store_true', help='if set, create json file with metadata')
    parser.add_argument('--create_grid', action='store_true', help='if set, create grid of images')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe_pretrained = StableDiffusionPipeline.from_pretrained(args.model_pretrained, torch_dtype=torch.float16)
    pipe_pretrained.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    print("Generating images of pretrained model")
    print("Edit prompt: ", args.prompts)

    if(args.create_metadata):
        metadata = []

    for p in args.prompts:
        for i in range(args.num_images):    
            while(True):
                nsfw = save_image(pipe_pretrained, p, os.path.join(args.output_dir, f"train/{p}_{i}.png"))
                
                #check if nsfw is a list
                if isinstance(nsfw, list):
                    nsfw = nsfw[0]

                if not nsfw:
                    break
            
            if(args.create_metadata):
                metadata.append({'file_name': f"train/{p}_{i}.png", 'text': p})
    
    if(args.create_grid):
        for p in args.prompts:
            concat_images_in_square_grid(args.output_dir, p, os.path.join(args.output_dir, f"grid {p}.png"))
    
    if(args.create_metadata):
        
        #save metadata to jsonl file
        with open(os.path.join(args.output_dir, 'metadata.jsonl'), 'w') as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")