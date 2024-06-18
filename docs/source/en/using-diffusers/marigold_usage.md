<!--Copyright 2024 Marigold authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Marigold Pipelines for Computer Vision Tasks

[Marigold](../api/pipelines/marigold) is a novel diffusion-based dense prediction approach, and a set of pipelines for various computer vision tasks, such as monocular depth estimation.

This guide will show you how to use Marigold to obtain fast and high-quality predictions for images and videos.

Each pipeline supports one Computer Vision task, which takes an input RGB image as input and produces a *prediction* of the modality of interest, such as a depth map of the input image.
Currently, the following tasks are implemented:

| Pipeline                                                                                                                                    | Predicted Modalities                                                                                             |                                                                       Demos                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
| [MarigoldDepthPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py)     | [Depth](https://en.wikipedia.org/wiki/Depth_map), [Disparity](https://en.wikipedia.org/wiki/Binocular_disparity) | [Fast Demo (LCM)](https://huggingface.co/spaces/prs-eth/marigold-lcm), [Slow Original Demo (DDIM)](https://huggingface.co/spaces/prs-eth/marigold) |
| [MarigoldNormalsPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py) | [Surface normals](https://en.wikipedia.org/wiki/Normal_mapping)                                                  |                                   [Fast Demo (LCM)](https://huggingface.co/spaces/prs-eth/marigold-normals-lcm)                                    |

The original checkpoints can be found under the [PRS-ETH](https://huggingface.co/prs-eth/) Hugging Face organization.
These checkpoints are meant to work with diffusers pipelines and the [original codebase](https://github.com/prs-eth/marigold).
The original code can also be used to train new checkpoints.

| Checkpoint                                                                                    | Modality | Comment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|-----------------------------------------------------------------------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [prs-eth/marigold-v1-0](https://huggingface.co/prs-eth/marigold-v1-0)                         | Depth    | The first Marigold Depth checkpoint, which predicts *affine-invariant depth* maps. The performance of this checkpoint in benchmarks was studied in the original [paper](https://huggingface.co/papers/2312.02145). Designed to be used with the `DDIMScheduler` at inference, it requires at least 10 steps to get reliable predictions. Affine-invariant depth prediction has a range of values in each pixel between 0 (near plane) and 1 (far plane); both planes are chosen by the model as part of the inference process. See the `MarigoldImageProcessor` reference for visualization utilities. |
| [prs-eth/marigold-depth-lcm-v1-0](https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0)     | Depth    | The fast Marigold Depth checkpoint, fine-tuned from `prs-eth/marigold-v1-0`. Designed to be used with the `LCMScheduler` at inference, it requires as little as 1 step to get reliable predictions. The prediction reliability saturates at 4 steps and declines after that.                                                                                                                                                                                                                                                                                                                           |
| [prs-eth/marigold-normals-v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1)         | Normals  | A preview checkpoint for the Marigold Normals pipeline. Designed to be used with the `DDIMScheduler` at inference, it requires at least 10 steps to get reliable predictions. The surface normals predictions are unit-length 3D vectors with values in the range from -1 to 1. *This checkpoint will be phased out after the release of `v1-0` version.*                                                                                                                                                                                                                                              |
| [prs-eth/marigold-normals-lcm-v0-1](https://huggingface.co/prs-eth/marigold-normals-lcm-v0-1) | Normals  | The fast Marigold Normals checkpoint, fine-tuned from `prs-eth/marigold-normals-v0-1`. Designed to be used with the `LCMScheduler` at inference, it requires as little as 1 step to get reliable predictions. The prediction reliability saturates at 4 steps and declines after that. *This checkpoint will be phased out after the release of `v1-0` version.*                                                                                                                                                                                                                                       |
The examples below are mostly given for depth prediction, but they can be universally applied with other supported modalities.
We showcase the predictions using the same input image of Albert Einstein generated by Midjourney.
This makes it easier to compare visualizations of the predictions across various modalities and checkpoints.

<div class="flex gap-4" style="justify-content: center; width: 100%;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://marigoldmonodepth.github.io/images/einstein.jpg"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Example input image for all Marigold pipelines
    </figcaption>
  </div>
</div>

### Depth Prediction Quick Start

To get the first depth prediction, load `prs-eth/marigold-depth-lcm-v1-0` checkpoint into `MarigoldDepthPipeline` pipeline, put the image through the pipeline, and save the predictions:

```python
import diffusers
import torch

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(image)

vis = pipe.image_processor.visualize_depth(depth.prediction)
vis[0].save("einstein_depth.png")

depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
depth_16bit[0].save("einstein_depth_16bit.png")
```

The visualization function for depth [`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_depth`] applies one of [matplotlib's colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html) (`Spectral` by default) to map the predicted pixel values from a single-channel `[0, 1]` depth range into an RGB image.
With the `Spectral` colormap, pixels with near depth are painted red, and far pixels are assigned blue color.
The 16-bit PNG file stores the single channel values mapped linearly from the `[0, 1]` range into `[0, 65535]`.
Below are the raw and the visualized predictions; as can be seen, dark areas (mustache) are easier to distinguish in the visualization:

<div class="flex gap-4">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth_16bit.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Predicted depth (16-bit PNG)
    </figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Predicted depth visualization (Spectral)
    </figcaption>
  </div>
</div>

### Surface Normals Prediction Quick Start

Load `prs-eth/marigold-normals-lcm-v0-1` checkpoint into `MarigoldNormalsPipeline` pipeline, put the image through the pipeline, and save the predictions:

```python
import diffusers
import torch

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals.prediction)
vis[0].save("einstein_normals.png")
```

The visualization function for normals [`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_normals`] maps the three-dimensional prediction with pixel values in the range `[-1, 1]` into an RGB image.
The visualization function supports flipping surface normals axes to make the visualization compatible with other choices of the frame of reference.
Conceptually, each pixel is painted according to the surface normal vector in the frame of reference, where `X` axis points right, `Y` axis points up, and `Z` axis points at the viewer.
Below is the visualized prediction:

<div class="flex gap-4" style="justify-content: center; width: 100%;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_normals.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Predicted surface normals visualization
    </figcaption>
  </div>
</div>

In this example, the nose tip almost certainly has a point on the surface, in which the surface normal vector points straight at the viewer, meaning that its coordinates are `[0, 0, 1]`.
This vector maps to the RGB `[128, 128, 255]`, which corresponds to the violet-blue color.
Similarly, a surface normal on the cheek in the right part of the image has a large `X` component, which increases the red hue.
Points on the shoulders pointing up with a large `Y` promote green color.

### Speeding up inference

The above quick start snippets are already optimized for speed: they load the LCM checkpoint, use the `fp16` variant of weights and computation, and perform just one denoising diffusion step.
The `pipe(image)` call completes in 280ms on RTX 3090 GPU.
Internally, the input image is encoded with the Stable Diffusion VAE encoder, then the U-Net performs one denoising step, and finally, the prediction latent is decoded with the VAE decoder into pixel space.
In this case, two out of three module calls are dedicated to converting between pixel and latent space of LDM.
Because Marigold's latent space is compatible with the base Stable Diffusion, it is possible to speed up the pipeline call by more than 3x (85ms on RTX 3090) by using a [lightweight replacement of the SD VAE](../api/models/autoencoder_tiny):

```diff
  import diffusers
  import torch

  pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
  ).to("cuda")

+ pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
+     "madebyollin/taesd", torch_dtype=torch.float16
+ ).cuda()

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
  depth = pipe(image)
```

As suggested in [Optimizations](../optimization/torch2.0#torch.compile), adding `torch.compile` may squeeze extra performance depending on the target hardware:

```diff
  import diffusers
  import torch

  pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
  ).to("cuda")

+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

  image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
  depth = pipe(image)
```

## Qualitative Comparison with Depth Anything

With the above speed optimizations, Marigold delivers predictions with more details and faster than [Depth Anything](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything) with the largest checkpoint [LiheYoung/depth-anything-large-hf](https://huggingface.co/LiheYoung/depth-anything-large-hf):

<div class="flex gap-4">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_depth.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Marigold LCM fp16 with Tiny AutoEncoder
    </figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/einstein_depthanything_large.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Depth Anything Large
    </figcaption>
  </div>
</div>

## Maximizing Precision and Ensembling

Marigold pipelines have a built-in ensembling mechanism combining multiple predictions from different random latents.
This is a brute-force way of improving the precision of predictions, capitalizing on the generative nature of diffusion.
The ensembling path is activated automatically when the `ensemble_size` argument is set greater than `1`.
When aiming for maximum precision, it makes sense to adjust `num_inference_steps` simultaneously with `ensemble_size`.
The recommended values vary across checkpoints but primarily depend on the scheduler type.
The effect of ensembling is particularly well-seen with surface normals:

```python
import diffusers

model_path = "prs-eth/marigold-normals-v1-0"

model_paper_kwargs = {
	diffusers.schedulers.DDIMScheduler: {
		"num_inference_steps": 10,
		"ensemble_size": 10,
	},
	diffusers.schedulers.LCMScheduler: {
		"num_inference_steps": 4,
		"ensemble_size": 5,
	},
}

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(model_path).to("cuda")
pipe_kwargs = model_paper_kwargs[type(pipe.scheduler)]

depth = pipe(image, **pipe_kwargs)

vis = pipe.image_processor.visualize_normals(depth.prediction)
vis[0].save("einstein_normals.png")
```

<div class="flex gap-4">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_lcm_normals.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Surface normals, no ensembling
    </figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_normals.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Surface normals, with ensembling
    </figcaption>
  </div>
</div>

As can be seen, all areas with fine-grained structurers, such as hair, got more conservative and on average more correct predictions.
Such a result is more suitable for precision-sensitive downstream tasks, such as 3D reconstruction.

## Quantitative Evaluation

To evaluate Marigold quantitatively in standard leaderboards and benchmarks (such as NYU, KITTI, and other datasets), follow the evaluation protocol outlined in the paper: load the full precision fp32 model and use appropriate values for `num_inference_steps` and `ensemble_size`.
Optionally seed randomness to ensure reproducibility. Maximizing `batch_size` will deliver maximum device utilization.

```python
import diffusers
import torch

device = "cuda"
seed = 2024
model_path = "prs-eth/marigold-v1-0"

model_paper_kwargs = {
	diffusers.schedulers.DDIMScheduler: {
		"num_inference_steps": 50,
		"ensemble_size": 10,
	},
	diffusers.schedulers.LCMScheduler: {
		"num_inference_steps": 4,
		"ensemble_size": 10,
	},
}

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

generator = torch.Generator(device=device).manual_seed(seed)
pipe = diffusers.MarigoldDepthPipeline.from_pretrained(model_path).to(device)
pipe_kwargs = model_paper_kwargs[type(pipe.scheduler)]

depth = pipe(image, generator=generator, **pipe_kwargs)

# evaluate metrics
```

## Using Predictive Uncertainty

The ensembling mechanism built into Marigold pipelines combines multiple predictions obtained from different random latents.
As a side effect, it can be used to quantify epistemic (model) uncertainty; simply specify `ensemble_size` greater than 1 and set `output_uncertainty=True`.
The resulting uncertainty will be available in the `uncertainty` field of the output.
It can be visualized as follows:

```python
import diffusers
import torch

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to("cuda")

image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(
	image,
	ensemble_size=10,  # any number greater than 1; higher values yield higher precision
	output_uncertainty=True,
)

uncertainty = pipe.image_processor.visualize_uncertainty(depth.uncertainty)
uncertainty[0].save("einstein_depth_uncertainty.png")
```

<div class="flex gap-4">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_depth_uncertainty.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Depth uncertainty
    </figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_einstein_normals_uncertainty.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Surface normals uncertainty
    </figcaption>
  </div>
</div>

The interpretation of uncertainty is easy: higher values (white) correspond to pixels, where the model struggles to make consistent predictions.
Evidently, the depth model is the least confident around edges with discontinuity, where the object depth changes drastically.
The surface normals model is the least confident in fine-grained structures, such as hair, and dark areas, such as the collar.

## Frame-by-frame Video Processing with Temporal Consistency

Due to Marigold's generative nature, each prediction is unique and defined by the random noise sampled for the latent initialization.
This becomes an obvious drawback compared to traditional end-to-end dense regression networks, as exemplified in the following videos:

<div class="flex gap-4">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama.gif"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Input video</figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama_depth_independent.gif"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Marigold Depth applied to input video frames independently</figcaption>
  </div>
</div>

To address this issue, it is possible to pass `latents` argument to the pipelines, which defines the starting point of diffusion.
Empirically, we found that a convex combination of the very same starting point noise latent and the latent corresponding to the previous frame prediction give sufficiently smooth results, as implemented in the snippet below:

```python
import imageio
from PIL import Image
from tqdm import tqdm
import diffusers
import torch

device = "cuda"
path_in = "obama.mp4"
path_out = "obama_depth.gif"

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
).to(device)
pipe.vae = diffusers.AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", torch_dtype=torch.float16
).to(device)
pipe.set_progress_bar_config(disable=True)

with imageio.get_reader(path_in) as reader:
    size = reader.get_meta_data()['size']
    last_frame_latent = None
    latent_common = torch.randn(
        (1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))
    ).to(device=device, dtype=torch.float16)

    out = []
    for frame_id, frame in tqdm(enumerate(reader), desc="Processing Video"):
        frame = Image.fromarray(frame)
        latents = latent_common
        if last_frame_latent is not None:
            latents = 0.9 * latents + 0.1 * last_frame_latent

        depth = pipe(
			frame, match_input_resolution=False, latents=latents, output_latent=True
        )
        last_frame_latent = depth.latent
        out.append(pipe.image_processor.visualize_depth(depth.prediction)[0])

    diffusers.utils.export_to_gif(out, path_out, fps=reader.get_meta_data()['fps'])
```

Here, the diffusion process starts from the given computed latent.
The pipeline sets `output_latent=True` to access `out.latent` and computes its contribution to the next frame's latent initialization.
The result is much more stable now:

<div class="flex gap-4">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama_depth_independent.gif"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Marigold Depth applied to input video frames independently</figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama_depth_consistent.gif"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Marigold Depth with forced latents initialization</figcaption>
  </div>
</div>

## Marigold for ControlNet

A very common application for depth prediction with diffusion models comes in conjunction with ControlNet.
Depth crispness plays a crucial role in obtaining high-quality results from ControlNet.
As seen in comparisons with other methods above, Marigold excels at that task.
The snippet below demonstrates how to load an image, compute depth, and pass it into ControlNet in a compatible format:

```python
import torch
import diffusers

device = "cuda"
generator = torch.Generator(device=device).manual_seed(2024)
image = diffusers.utils.load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"
)

pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16, variant="fp16"
).to(device)

depth_image = pipe(image, generator=generator).prediction
depth_image = pipe.image_processor.visualize_depth(depth_image, color_map="binary")
depth_image[0].save("motorcycle_controlnet_depth.png")

controlnet = diffusers.ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
).to(device)
pipe = diffusers.StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnet
).to(device)
pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

controlnet_out = pipe(
    prompt="high quality photo of a sports bike, city",
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    image=depth_image,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.7,
    generator=generator,
).images
controlnet_out[0].save("motorcycle_controlnet_out.png")
```

<div class="flex gap-4">
  <div style="flex: 1 1 33%; max-width: 33%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Input image
    </figcaption>
  </div>
  <div style="flex: 1 1 33%; max-width: 33%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/motorcycle_controlnet_depth.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Depth in the format compatible with ControlNet
    </figcaption>
  </div>
  <div style="flex: 1 1 33%; max-width: 33%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/motorcycle_controlnet_out.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      ControlNet generation, conditioned on depth and prompt: "high quality photo of a sports bike, city"
    </figcaption>
  </div>
</div>

Hopefully, you will find Marigold useful for solving your downstream tasks, be it a part of a more broad generative workflow, or a perception task, such as 3D reconstruction.
