### MotionCtrl SVD

[MotionCtrl](https://arxiv.org/abs/2312.03641) is a method that allows flexible control over object and camera movement in video diffusion models. The implementation here is only for [Stable Video Diffusion](https://wzhouxiff.github.io/projects/MotionCtrl/) as presented by the authors. You can find a more implementation-oriented description about it in [this](https://github.com/huggingface/diffusers/issues/6688#issuecomment-1913459070) comment. You can find example results, some useful discussion and MotionCtrl conversion script [here](https://github.com/huggingface/diffusers/pull/6844).

Paper: https://arxiv.org/abs/2312.03641
Project site: https://wzhouxiff.github.io/projects/MotionCtrl/
Colab: https://colab.research.google.com/drive/17xIdW-xWk4hCAIkGq0OfiJYUqwWSPSAz?usp=sharing
YouTube: Feature on [Two Minute Papers](https://youtu.be/2hfPVBDMB-o).

### Inference

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_gif, load_image

from pipeline_stable_video_motionctrl_diffusion import StableVideoMotionCtrlDiffusionPipeline
from unet_motionctrl import UNetSpatioTemporalConditionMotionCtrlModel

# Initialize pipeline
ckpt = "a-r-r-o-w/motionctrl-svd"
unet = UNetSpatioTemporalConditionMotionCtrlModel.from_pretrained(ckpt, subfolder="unet", torch_dtype=torch.float16)
pipe = StableVideoMotionCtrlDiffusionPipeline.from_pretrained(
    ckpt,
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Input image and camera pose
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
camera_pose = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.2, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.28750000000000003, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.37500000000000006, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.4625000000000001, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.55, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.6375000000000002, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7250000000000001, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.8125000000000002, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.9000000000000001, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.9875000000000003, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0750000000000002, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.1625000000000003, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.2500000000000002, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.3375000000000001, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.4250000000000003, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.5125000000000004, 0.0, 0.0, 1.0, 0.0],
]

# Set MotionCtrl scale
pipe.unet.set_motionctrl_scale(0.8)

# Generation (make sure num_frames == len(camera_pose))
num_frames = 16
frames = pipe(
    image=image,
    camera_pose=camera_pose,
    num_frames=num_frames,
    num_inference_steps=20,
    decode_chunk_size=2,
    motion_bucket_id=255,
    fps=15,
    min_guidance_scale=1,
    max_guidance_scale=3.5,
    generator=torch.Generator().manual_seed(42)
).frames[0]
export_to_gif(frames, f"animation.gif")
```

Note that `camera_pose` must be provided for inference. It represents the orientation and position of the camera, and is known as [Camera Projection Matrix](https://en.wikipedia.org/wiki/Camera_matrix). It must be a list of lists where the outer list has length equal to `num_frames` and inner list has length equal to `3x4 = 9`. For some general camera matrices and movements (left, right, up, down, clockwise, anticlockwise, zoom in/out, etc.), refer to [this](https://colab.research.google.com/drive/17xIdW-xWk4hCAIkGq0OfiJYUqwWSPSAz?usp=sharing) notebook.


