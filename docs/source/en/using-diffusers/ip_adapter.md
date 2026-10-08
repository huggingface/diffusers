<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IP-Adapter

[IP-Adapter](https://huggingface.co/papers/2308.06721) steers a pretrained diffusion model with a reference image while you keep a text prompt. It freezes the base model and adds a small set of image cross-attention layers, which makes it practical for matching a subject or style from a photo without fine-tuning.

```text
text prompt                         reference image
    |                                     |
 text encoder                        image encoder
    |                                     |
    v                                     v
 text cross-attn  (frozen)          IP-Adapter cross-attn
    \                                     /
     \                                   /
      +-------->  denoiser  <-----------+
                  (frozen base)
```

IP-Adapter checkpoints are typically ~100MB because they store adapter weights, not a full model. Load a base pipeline first, then load the adapter with [`~loaders.IPAdapterMixin.load_ip_adapter`].

> [!TIP]
> IP-Adapters are available to many models such as [Flux](../api/pipelines/flux#ip-adapter) and [Stable Diffusion 3](../api/pipelines/stable_diffusion/stable_diffusion_3), and more. The examples in this guide use Stable Diffusion and Stable Diffusion XL.

Use [`~loaders.IPAdapterMixin.set_ip_adapter_scale`] to control how strongly the IP-Adapter steers generation. `1.0` applies the adapter at full strength; `0.5` usually balances text and image prompts.

The examples below show IP-Adapter on common tasks.

<hfoptions id="usage">
<hfoption id="text-to-image">

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)
pipeline.set_ip_adapter_scale(0.8)
```

Pass an image to `ip_adapter_image` along with a text prompt to generate an image.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")
pipeline(
    prompt="a polar bear sitting in a chair drinking a milkshake",
    ip_adapter_image=image,
    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png" width="400" alt="IP-Adapter image"/>
    <figcaption style="text-align: center;">IP-Adapter image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner_2.png" width="400" alt="generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

</hfoption>
<hfoption id="image-to-image">

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)
pipeline.set_ip_adapter_scale(0.8)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png")
pipeline(
    prompt="best quality, high quality",
    image=image,
    ip_adapter_image=ip_image,
    strength=0.5,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png" width="300" alt="input image"/>
    <figcaption style="text-align: center;">input image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png" width="300" alt="IP-Adapter image"/>
    <figcaption style="text-align: center;">IP-Adapter image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_3.png" width="300" alt="generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

</hfoption>
<hfoption id="inpainting">

```py
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)
pipeline.set_ip_adapter_scale(0.6)

mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_mask.png")
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png")
pipeline(
    prompt="a cute gummy bear waving",
    image=image,
    mask_image=mask_image,
    ip_adapter_image=ip_image,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png" width="300" alt="input image"/>
    <figcaption style="text-align: center;">input image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png" width="300" alt="IP-Adapter image"/>
    <figcaption style="text-align: center;">IP-Adapter image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png" width="300" alt="generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

</hfoption>
<hfoption id="video">

The [`~DiffusionPipeline.enable_model_cpu_offload`] method is useful for reducing memory and it should be enabled after the IP-Adapter is loaded. Otherwise, the IP-Adapter's image encoder is also offloaded to the CPU and returns an error.

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from diffusers.utils import load_image

adapter = MotionAdapter.from_pretrained(
  "guoyww/animatediff-motion-adapter-v1-5-2",
  dtype=torch.float16
)
pipeline = AnimateDiffPipeline.from_pretrained(
  "emilianJR/epiCRealism",
  motion_adapter=adapter,
  dtype=torch.float16
)
scheduler = DDIMScheduler.from_pretrained(
    "emilianJR/epiCRealism",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.vae.enable_slicing()
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipeline.enable_model_cpu_offload()

ip_adapter_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png")
pipeline(
    prompt="A cute gummy bear waving",
    negative_prompt="bad quality, worse quality, low resolution",
    ip_adapter_image=ip_adapter_image,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=50,
).frames[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png" width="400" alt="IP-Adapter image"/>
    <figcaption style="text-align: center;">IP-Adapter image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gummy_bear.gif" width="400" alt="generated video"/>
    <figcaption style="text-align: center;">generated video</figcaption>
  </figure>
</div>

</hfoption>
</hfoptions>

## Checkpoint variants

Load Plus when detail from the reference image matters most. Load FaceID when you need InsightFace identity embeddings rather than CLIP image embeddings.

<hfoptions id="ipadapter-variants">
<hfoption id="IP-Adapter Plus">

```py
import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoPipelineForText2Image

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
)
```

</hfoption>
<hfoption id="IP-Adapter FaceID">

FaceID checkpoints use face embeddings from [InsightFace](https://github.com/deepinsight/insightface) instead of CLIP image embeddings. Use the [`DDIMScheduler`] or [`EulerDiscreteScheduler`] for FaceID models. Extract the face embeddings and pass them as a list of tensors to `ip_adapter_image_embeds`.

```py
# pip install insightface
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image
from insightface.app import FaceAnalysis

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    dtype=torch.float16,
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
  "h94/IP-Adapter-FaceID",
  subfolder=None,
  weight_name="ip-adapter-faceid_sd15.bin",
  image_encoder_folder=None
)
pipeline.set_ip_adapter_scale(0.6)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")

ref_images_embeds = []
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
faces = app.get(image)
image = torch.from_numpy(faces[0].normed_embedding)
ref_images_embeds.append(image.unsqueeze(0))
ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=torch.float16, device="cuda")

pipeline(
    prompt="A photo of a girl",
    ip_adapter_image_embeds=[id_embeds],
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
).images[0]
```

For FaceID Plus, load [`~transformers.CLIPVisionModelWithProjection`] as the image encoder.

```py
import torch
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoPipelineForText2Image

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    image_encoder=image_encoder,
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

pipeline.load_ip_adapter(
  "h94/IP-Adapter-FaceID",
  subfolder=None,
  weight_name="ip-adapter-faceid-plus_sd15.bin"
)
```

FaceID Plus and Plus v2 also need CLIP image embeddings on top of the InsightFace embeds. After you prepare the face embeddings, encode the reference image and assign the CLIP embeds to the image projection layer.

```py
import torch
from diffusers.utils import load_image

ip_adapter_images = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")
num_images = 1
clip_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=[ip_adapter_images],
    ip_adapter_image_embeds=None,
    device=torch.device("cuda"),
    num_images_per_prompt=num_images,
    do_classifier_free_guidance=True,
)[0]

pipeline.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
# set to True if using IP-Adapter FaceID Plus v2
pipeline.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False
```

</hfoption>
</hfoptions>

## Image embeddings

[`~StableDiffusionPipeline.prepare_ip_adapter_image_embeds`] encodes IP-Adapter images into embeddings you can save and reuse. Precompute them once instead of loading and encoding the same images every time you run the pipeline.

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=image,
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

torch.save(image_embeds, "image_embeds.ipadpt")
```

Reload the image embeddings by passing them to the `ip_adapter_image_embeds` parameter. Set `image_encoder_folder` to `None` because you don't need the image encoder anymore to generate the image embeddings.

> [!TIP]
> You can also load image embeddings from other sources such as ComfyUI.

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  image_encoder_folder=None,
  weight_name="ip-adapter_sdxl.bin"
)
pipeline.set_ip_adapter_scale(0.8)
image_embeds = torch.load("image_embeds.ipadpt")
pipeline(
    prompt="a polar bear sitting in a chair drinking a milkshake",
    ip_adapter_image_embeds=image_embeds,
    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    num_inference_steps=100,
).images[0]
```

## Masking

Binary masking enables assigning an IP-Adapter image to a specific area of the output image, making it useful for composing multiple IP-Adapter images. Each IP-Adapter image requires a binary mask.

Load the [`~image_processor.IPAdapterMaskProcessor`] to preprocess the image masks. For the best results, provide the output `height` and `width` to ensure masks with different aspect ratios are appropriately sized. If the input masks already match the aspect ratio of the generated image, you don't need to set the `height` and `width`.

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"

mask1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask1.png")
mask2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask2.png")

processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask1, mask2], height=1024, width=1024)
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_mask1.png" width="200" alt="mask 1"/>
    <figcaption style="text-align: center;">mask 1</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_mask2.png" width="200" alt="mask 2"/>
    <figcaption style="text-align: center;">mask 2</figcaption>
  </figure>
</div>

Provide both the IP-Adapter images and their scales as a list. Pass the preprocessed masks to `cross_attention_kwargs` in the pipeline.

```py
face_image1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")
face_image2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl2.png")

pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
)
pipeline.set_ip_adapter_scale([[0.7, 0.7]])

ip_images = [[face_image1, face_image2]]
masks = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]

pipeline(
  prompt="2 girls",
  ip_adapter_image=ip_images,
  negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
  cross_attention_kwargs={"ip_adapter_masks": masks}
).images[0]
```

<div style="display: flex; flex-direction: column; gap: 10px;">
  <div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
    <figure>
      <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_girl1.png" width="400" alt="IP-Adapter image 1"/>
      <figcaption style="text-align: center;">IP-Adapter image 1</figcaption>
    </figure>
    <figure>
      <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_girl2.png" width="400" alt="IP-Adapter image 2"/>
      <figcaption style="text-align: center;">IP-Adapter image 2</figcaption>
    </figure>
  </div>
  <div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
    <figure>
      <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_attention_mask_result_seed_0.png" width="400" alt="Generated image with mask"/>
      <figcaption style="text-align: center;">generated with mask</figcaption>
    </figure>
    <figure>
      <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_no_attention_mask_result_seed_0.png" width="400" alt="Generated image without mask"/>
      <figcaption style="text-align: center;">generated without mask</figcaption>
    </figure>
  </div>
</div>

## Recipes

Combine IP-Adapter with other adapters or pipelines, like multiple adapters, ControlNet, InstantStyle, or few-step LCM, when one reference image isn’t enough.
### Multiple IP-Adapters

Combine multiple IP-Adapters to generate images in more diverse styles. For example, you can use IP-Adapter Face to generate consistent faces and characters and IP-Adapter Plus to generate those faces in specific styles.

Load an image encoder with [`~transformers.CLIPVisionModelWithProjection`].

```py
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    dtype=torch.float16,
)
```

Load a base model, scheduler and the following IP-Adapters.

- [ip-adapter-plus_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) uses patch embeddings and a ViT-H image encoder
- [ip-adapter-plus-face_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) uses patch embeddings and a ViT-H image encoder but it is conditioned on images of cropped faces

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16,
    image_encoder=image_encoder,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
)
pipeline.set_ip_adapter_scale([0.7, 0.3])
# enable_model_cpu_offload to reduce memory usage
pipeline.enable_model_cpu_offload()
```

Load an image and a folder containing images of a certain style to apply.

```py
face_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png")
style_folder = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy"
style_images = [load_image(f"{style_folder}/img{i}.png") for i in range(10)]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png" width="400" alt="Face image"/>
    <figcaption style="text-align: center;">face image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_style_grid.png" width="400" alt="Style images"/>
    <figcaption style="text-align: center;">style images</figcaption>
  </figure>
</div>

Pass style and face images as a list to `ip_adapter_image`.

```py
generator = torch.Generator(device="cpu").manual_seed(0)

pipeline(
    prompt="wonderwoman",
    ip_adapter_image=[style_images, face_image],
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
).images[0]
```

<div style="display: flex; justify-content: center;">
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_multi_out.png" width="400" alt="Generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

### Structural control

For structural control, combine IP-Adapter with [ControlNet](../api/pipelines/controlnet) conditioned on depth maps, edge maps, pose estimations, and more.

The example below loads a [`ControlNetModel`] checkpoint conditioned on depth maps and combines it with an IP-Adapter.

```py
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
  "lllyasviel/control_v11f1p_sd15_depth",
  dtype=torch.float16
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models",
  weight_name="ip-adapter_sd15.bin"
)
```

Pass the depth map and IP-Adapter image to the pipeline.

```py
depth_map = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png")
ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png")
pipeline(
  prompt="best quality, high quality",
  image=depth_map,
  ip_adapter_image=ip_adapter_image,
  negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png" width="300" alt="IP-Adapter image"/>
    <figcaption style="text-align: center;">IP-Adapter image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png" width="300" alt="Depth map"/>
    <figcaption style="text-align: center;">depth map</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ipa-controlnet-out.png" width="300" alt="Generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

### Style and layout control

For style and layout control, combine IP-Adapter with [InstantStyle](https://huggingface.co/papers/2404.02733). InstantStyle separates *style* (color, texture, overall feel) from *content* and applies style only in style-specific blocks so content areas stay intact. That gives stronger style consistency and clearer layout control.

Activate the IP-Adapter only in selected layers with [`~loaders.IPAdapterMixin.set_ip_adapter_scale`]. The example below turns it on in the model's down `block_2` (layout) and up `block_0` (style).

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  dtype=torch.float16
).to("cuda")  # or "mps", "xpu", "cpu"
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter_sdxl.bin"
)

scale = {
    "down": {"block_2": [0.0, 1.0]},
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)
```

Load the style image and generate an image.

```py
style_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg")

pipeline(
    prompt="a cat, masterpiece, best quality, high quality",
    ip_adapter_image=style_image,
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    guidance_scale=5,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg" width="400" alt="Style image"/>
    <figcaption style="text-align: center;">style image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png" width="400" alt="Generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

The figures below compare InstantStyle's style-only activation (up `block_0`) with turning the IP-Adapter on in all layers. All-layers usually follows the image prompt more strongly and can reduce diversity. Prefer the style-only scale when you want InstantStyle's layout-preserving behavior.

> [!TIP]
> You don't need to specify all the layers in the `scale` dictionary. Layers not included are set to 0, which means the IP-Adapter is disabled.

```py
scale = {
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)

pipeline(
    prompt="a cat, masterpiece, best quality, high quality",
    ip_adapter_image=style_image,
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    guidance_scale=5,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_only.png" width="400" alt="Generated image (style only)"/>
    <figcaption style="text-align: center;">style-only (up block_0)</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_ip_adapter.png" width="400" alt="Generated image (all layers)"/>
    <figcaption style="text-align: center;">all layers</figcaption>
  </figure>
</div>

### Instant generation

Combine IP-Adapter with an [LCM](../api/pipelines/latent_consistency_models) LoRA for few-step generation.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
  "sd-dreambooth-library/herge-style",
  dtype=torch.float16
)
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models",
  weight_name="ip-adapter_sd15.bin"
)
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

pipeline.set_ip_adapter_scale(0.4)
ip_adapter_image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
pipeline(
    prompt="herge_style woman in armor, best quality, high quality",
    ip_adapter_image=ip_adapter_image,
    num_inference_steps=4,
    guidance_scale=1,
).images[0]
```

<div style="display: flex; justify-content: center;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_herge.png" width="400" alt="Generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>
