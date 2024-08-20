message = """

Example Demo of Adaptive Mask Inpainting

Beyond the Contact: Discovering Comprehensive Affordance for 3D Objects from Pre-trained 2D Diffusion Models
Kim et al.
ECCV-2024 (Oral)


Please prepare the environment via

```
conda create --name ami python=3.9
conda activate ami

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install easydict
pip install diffusers==0.20.2 accelerate safetensors transformers
pip install setuptools==59.5.0
pip install opencv-python
```


Put the code inside the root of diffusers library (i.e., as '/home/username/diffusers/adaptive_mask_inpainting_example.py') and run the python code.




"""
print(message)


import numpy as np
import torch
from easydict import EasyDict
from PIL import Image


from diffusers import DDIMScheduler
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


from examples.community.adaptive_mask_inpainting import(
    download_file,
    AdaptiveMaskInpaintPipeline, 
    PointRendPredictor,
    MaskDilateScheduler,
    ProvokeScheduler,
)




if __name__ == "__main__":    
    """
    Download Necessary Files
    """
    download_file(
        url = "https://huggingface.co/datasets/jellyheadnadrew/adaptive-mask-inpainting-test-images/resolve/main/model_final_edd263.pkl?download=true",
        output_file = "model_final_edd263.pkl",
    )
    download_file(
        url = "https://huggingface.co/datasets/jellyheadnadrew/adaptive-mask-inpainting-test-images/resolve/main/pointrend_rcnn_R_50_FPN_3x_coco.yaml?download=true",
        output_file = "pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    )
    download_file(
        url = "https://huggingface.co/datasets/jellyheadnadrew/adaptive-mask-inpainting-test-images/resolve/main/input_img.png?download=true",
        output_file = "input_img.png"
    )
    download_file(
        url = "https://huggingface.co/datasets/jellyheadnadrew/adaptive-mask-inpainting-test-images/resolve/main/input_mask.png?download=true",
        output_file = "input_mask.png"
    )
    download_file(
        url = "https://huggingface.co/datasets/jellyheadnadrew/adaptive-mask-inpainting-test-images/resolve/main/Base-PointRend-RCNN-FPN.yaml?download=true",
        output_file = "Base-PointRend-RCNN-FPN.yaml"
    )
    download_file(
        url = "https://huggingface.co/datasets/jellyheadnadrew/adaptive-mask-inpainting-test-images/resolve/main/Base-RCNN-FPN.yaml?download=true",
        output_file = "Base-RCNN-FPN.yaml",
    )
    
    """ 
    Prepare Adaptive Mask Inpainting Pipeline
    """
    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_steps = 50
    
    # Scheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        clip_sample=False, 
        set_alpha_to_one=False
    )
    scheduler.set_timesteps(num_inference_steps=num_steps)

    ## load models as pipelines
    pipeline = AdaptiveMaskInpaintPipeline.from_pretrained(
        "Uminosachi/realisticVisionV51_v51VAE-inpainting", 
        scheduler=scheduler, 
        torch_dtype=torch.float16, 
        requires_safety_checker=False
    ).to(device)

    ## disable safety checker
    enable_safety_checker = False
    if not enable_safety_checker:
        pipeline.safety_checker = None

    # declare segmentation model used for mask adaptation
    use_visualizer = False
    assert not use_visualizer, \
    """
    If you plan to 'use_visualizer', USE WITH CAUTION. 
    It creates a directory of images and masks, which is used for merging into a video.
    The procedure involves deleting the directory of images, which means that 
    if you set the directory wrong you can have other important files blown away.
    """
    
    adaptive_mask_model = PointRendPredictor(
        pointrend_thres=0.2, 
        device="cuda" if torch.cuda.is_available() else "cpu", 
        use_visualizer=use_visualizer,
        config_pth="pointrend_rcnn_R_50_FPN_3x_coco.yaml",
        weights_pth="model_final_edd263.pkl",
    )
    pipeline.register_adaptive_mask_model(adaptive_mask_model)

    step_num = int(num_steps * 0.1)
    final_step_num = num_steps - step_num * 7
    # adaptive mask settings
    adaptive_mask_settings = EasyDict(
        dict(
            dilate_scheduler=MaskDilateScheduler(
                max_dilate_num=20,
                num_inference_steps=num_steps,
                schedule=[20] * step_num + [10] * step_num + [5] * step_num + [4] * step_num + [3] * step_num + [2] * step_num + [1] * step_num + [0] * final_step_num
            ),
            dilate_kernel=np.ones((3, 3), dtype=np.uint8),
            provoke_scheduler=ProvokeScheduler(
                num_inference_steps=num_steps,
                schedule=list(range(2, 10 + 1, 2)) + list(range(12, 40 + 1, 2)) + [45],
                is_zero_indexing=False,
            ),
        )
    )
    pipeline.register_adaptive_mask_settings(adaptive_mask_settings)
    
    """ 
    Run Adaptive Mask Inpainting 
    """
    default_mask_image = Image.open("./input_mask.png").convert("L")
    init_image = Image.open("./input_img.png").convert("RGB")
    
    
    seed = 46
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    image = pipeline(
        prompt="a man sitting on a couch",
        negative_prompt="worst quality, normal quality, low quality, bad anatomy, artifacts, blurry, cropped, watermark, greyscale, nsfw",
        image=init_image,
        default_mask_image=default_mask_image,
        guidance_scale=11.0,
        strength=0.98,
        use_adaptive_mask=True,
        generator=generator,
        enforce_full_mask_ratio=0.0,
        visualization_save_dir="./ECCV2024_adaptive_mask_inpainting_demo", # DON'T EVER CHANGE THIS!!!
        human_detection_thres=0.015,
    ).images[0]

    
    image.save(f'final_img.png')