from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
from utils import save_image, concat_images_in_square_grid, get_random_prompt, get_clip_score
import argparse

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="stabilityai/stable-diffusion-2", help='pretrained model')
    parser.add_argument('--prompts', nargs='+', type=str, help='edit prompt')
    parser.add_argument('--num_images', type=int, default=30, help='number of images')
    parser.add_argument('--output_dir', type=str, default="/scratch/mp5847/diffusers_ckpt/output", help='output directory')
    parser.add_argument('--clip_filtering_threshold', type=float, default=0.25, help='clip filtering threshold')
    parser.add_argument('--clip_filtering_tolerance', type=int, default=10, help='how many iterations before continue')
    
    #create a store_true argument
    parser.add_argument('--create_metadata', action='store_true', help='if set, create json file with metadata')
    parser.add_argument('--create_grid', action='store_true', help='if set, create grid of images')
    parser.add_argument('--random_prompt', action='store_true', help='if set, select random prompt from prompts.json file')
    parser.add_argument('--clip_filtering', action='store_true', help='filter images based on clip similarity')

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


    
    for i in range(args.num_images):   
        p = args.prompts[i]
        while(True):
            if(args.random_prompt):
                prompt = get_random_prompt(p)
                nsfw = save_image(pipe_pretrained, get_random_prompt(p), os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
            
                if(args.clip_filtering):
                    history = {}
                    tolerance = 0
                    while(tolerance < args.clip_filtering_tolerance):
                        clip_score = get_clip_score(p, os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
                        
                        print(f"Prompt: {prompt}, clip score: {clip_score}")
                        if(clip_score > args.clip_filtering_threshold):    
                            print("Accepting image") 
                            #delete all images in history
                            for k, v in history.items():
                                print(f"Deleting image with clip score {v}")
                                os.remove(k)
                            break
                        else:
                            print("Rejecting image")
                            #rename image with clip score and save it in history
                            history[os.path.join(args.output_dir, f"train/{prompt}_{i}_{clip_score}.png")] = clip_score
                            os.rename(os.path.join(args.output_dir, f"train/{prompt}_{i}.png"), os.path.join(args.output_dir, f"train/{prompt}_{i}_{clip_score}.png"))
                            
                            #generate new image
                            nsfw = save_image(pipe_pretrained, prompt, os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
                            tolerance += 1     

                    #if tolerance is reached, select the image with the highest clip score and rename it, delete the others
                    if(tolerance == args.clip_filtering_tolerance):
                        print("Tolerance reached")
                        max_clip_score = max(history.values())
                        for k, v in history.items():
                            if(v == max_clip_score):
                                print(f"Accepting image with clip score {v}")
                                os.rename(k, os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
                            else:
                                print(f"Deleting image with clip score {v}")
                                os.remove(k)

            else:
                prompt = p
                nsfw = save_image(pipe_pretrained, p, os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))

                if(args.clip_filtering):
                    history = {}
                    tolerance = 0
                    while(tolerance < args.clip_filtering_tolerance):
                        clip_score = get_clip_score("Kilian Eng style", os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
                        
                        print(f"Prompt: {prompt}, clip score: {clip_score}")
                        if(clip_score > args.clip_filtering_threshold):    
                            print("Accepting image") 
                            #delete all images in history
                            for k, v in history.items():
                                print(f"Deleting image with clip score {v}")
                                os.remove(k)
                            break
                        else:
                            print("Rejecting image")
                            #rename image with clip score and save it in history
                            history[os.path.join(args.output_dir, f"train/{prompt}_{i}_{clip_score}.png")] = clip_score
                            os.rename(os.path.join(args.output_dir, f"train/{prompt}_{i}.png"), os.path.join(args.output_dir, f"train/{prompt}_{i}_{clip_score}.png"))
                            
                            #generate new image
                            nsfw = save_image(pipe_pretrained, prompt, os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
                            tolerance += 1     

                    #if tolerance is reached, select the image with the highest clip score and rename it, delete the others
                    if(tolerance == args.clip_filtering_tolerance):
                        print("Tolerance reached")
                        max_clip_score = max(history.values())
                        print("Deleting all images")
                        for k, v in history.items():
                            if(v == max_clip_score):
                                print(f"Accepting image with clip score {v}")
                                os.rename(k, os.path.join(args.output_dir, f"train/{prompt}_{i}.png"))
                            else:
                                print(f"Deleting image with clip score {v}")
                                os.remove(k)
                            
                        print("Move on to next prompt")

            #check if nsfw is a list
            if isinstance(nsfw, list):
                nsfw = nsfw[0]

            if not nsfw:
                break
        
        if(args.create_metadata):
            metadata.append({'file_name': f"train/{prompt}_{i}.png", 'text': prompt})
        
    if(args.create_grid):
        for p in args.prompts:
            concat_images_in_square_grid(os.path.join(args.output_dir, "train"), p, os.path.join(args.output_dir, f"grid {p}.png"))
    
    if(args.create_metadata):
        
        #save metadata to jsonl file
        with open(os.path.join(args.output_dir, 'metadata.jsonl'), 'w') as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")