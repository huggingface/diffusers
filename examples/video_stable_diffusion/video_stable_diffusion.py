import os
import sys
import time
import torch
import clip
import random

from types import SimpleNamespace
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download

sys.path.extend(
    [
        os.path.join(sys.path[0], "deforum-stable-diffusion"),
        os.path.join(sys.path[0], "deforum-stable-diffusion", "src"),
    ]
)

from helpers.save_images import get_output_folder
from helpers.render import render_animation
from helpers.model_load import load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from configuration import Root, DeforumAnimArgs, DeforumArgs

REPO_ID = "feizhengcong/video-stable-diffusion"
SD_REPO_ID = "runwayml/stable-diffusion-v1-5"
# DPT_MODEL = "dpt_large-midas-2f21e586.pt"
# ADABIN_MODEL = "AdaBins_nyu.pt"


class VideoStableDiffusionPipeline(DiffusionPipeline):
    def __init__(self):
        super().__init__()

        root = Root()
        root = SimpleNamespace(**root)
        root.models_path = os.path.join(sys.path[0], root.models_path)
        root.output_path = os.path.join(sys.path[0], root.output_path)
        root.configs_path = os.path.join(sys.path[0], "deforum-stable-diffusion", root.configs_path)
        root.models_path, root.output_path = get_model_output_paths(root)

        # download stable diffusion v1.5 checkpoint to ./models
        hf_hub_download(repo_id=SD_REPO_ID, filename=root.model_checkpoint, cache_dir=root.models_path)

        root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=False)

        args_dict = DeforumArgs()
        anim_args_dict = DeforumAnimArgs()
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        args.outdir = get_output_folder(root.output_path, args.batch_name)
        args.timestring = time.strftime("%Y%m%d%H%M%S")
        args.strength = max(0.0, min(1.0, args.strength))

        if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
            root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
            if args.aesthetics_scale > 0:
                root.aesthetics_model = load_aesthetics_model(args, root)
        if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)
        if not args.use_init:
            args.init_image = None
        if args.sampler == "plms" and (args.use_init or anim_args.animation_mode != "None"):
            print("Init images aren't supported with PLMS yet, switching to KLMS")
            args.sampler = "klms"
        if args.sampler != "ddim":
            args.ddim_eta = 0

        if anim_args.animation_mode == "None":
            anim_args.max_frames = 1
        elif anim_args.animation_mode == "Video Input":
            args.use_init = True

        self.root = root
        self.args = args
        self.anim_args = anim_args

    def __call__(self, prompt, move, num_images):
        prompt_dict = self.__prompt_formulation__(prompt, num_images)
        self.__move_formulation__(move)
        render_animation(prompt_dict)

    def __prompt_formulation__(self, prompt, num_images):
        idx = 0
        prompt_dict = {}
        step = int(num_images / 50)
        for i in range(step):
            prompt_dict[str(idx)] = prompt
            idx += 50
        return prompt_dict

    def __move_formulation__(self, move):
        self.anim_args.translation_x = "0:(" + str(move["x"]) + ")"
        self.anim_args.translation_y = "0:(" + str(move["y"]) + ")"
        self.anim_args.translation_z = "0:(" + str(move["z"]) + ")"
