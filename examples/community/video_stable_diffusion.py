import os, sys
import torch

from types import SimpleNamespace
from diffusers import DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from huggingface_hub import hf_hub_url, cached_download, snapshot_download

REPO_ID = "feizhengcong/video-stable-diffusion"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DPT_MODEL = "dpt_large-midas-2f21e586.pt"
ADABIN_MODEL = "AdaBins_nyu.pt"

args_dict = {
    "models_path": "models",
    "outdir": "./outputs",
}


class VideoStableDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet: UNet2DConditionModel, scheduler: DDIMScheduler):
        super().__init__()

        snapshot_download(repo_id=REPO_ID)

        sys.path.extend(
            [
                "deforum-stable-diffusion/",
                "deforum-stable-diffusion/src",
            ]
        )

        from helpers.model_load import make_linear_decode, load_model, get_model_output_paths

        self.args = SimpleNamespace(**args_dict)

        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
        self.register_modules(pipe=pipe, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, prompt, num_images):

        prompt_dict = self.__prompt_formulation__(prompt, num_images)
        self.render_animation(prompt_dict)

    def __prompt_formulation__(self, prompt, num_images):
        idx = 0
        prompt_dict = {}
        step = int(num_images / 50)
        for i in range(step):
            prompt_dict[str(idx)] = prompt
            idx += 50
        return prompt_dict

    def __render_animation(self, animation_prompts):
        os.makedirs(self.args.outdir, exist_ok=True)
        print(f"Saving animation frames to {self.args.outdir}")

        pass
