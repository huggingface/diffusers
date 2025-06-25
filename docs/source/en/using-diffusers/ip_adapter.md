<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IP-Adapter

[IP-Adapter](https://huggingface.co/papers/2308.06721) is a lightweight adapter designed to integrate image-based guidance with text-to-image diffusion models. The adapter uses an image encoder to extract image features that are passed to the newly added cross-attention layers in the UNet and fine-tuned. The original UNet model and the existing cross-attention layers corresponding to text features is frozen. Decoupling the cross-attention for image and text features enables more fine-grained and controllable generation.

IP-Adapter files are typically ~100MBs because they only contain the image embeddings. This means you need to load a model first, and then load the IP-Adapter with [`~loaders.IPAdapterMixin.load_ip_adapter`].

> [!TIP]
> IP-Adapters are available to many models such as [Flux](../api/pipelines/flux#ip-adapter) and [Stable Diffusion 3](../api/pipelines/stable_diffusion/stable_diffusion_3), and more. The examples in this guide use Stable Diffusion and Stable Diffusion XL.

Use the [`~loaders.IPAdapterMixin.set_ip_adapter_scale`] parameter to scale the influence of the IP-Adapter during generation. A value of `1.0` means the model is only conditioned on the image prompt, and `0.5` typically produces balanced results between the text and image prompt.

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")
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

Take a look at the examples below to learn how to use IP-Adapter for other tasks.

<hfoptions id="usage">
<hfoption id="image-to-image">

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")
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
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")
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

The [`~DiffusionPipeline.enable_model_cpu_offload`] method is useful for reducing memory and it should be enabled **after** the IP-Adapter is loaded. Otherwise, the IP-Adapter's image encoder is also offloaded to the CPU and returns an error.

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from diffusers.utils import load_image

adapter = MotionAdapter.from_pretrained(
  "guoyww/animatediff-motion-adapter-v1-5-2",
  torch_dtype=torch.float16
)
pipeline = AnimateDiffPipeline.from_pretrained(
  "emilianJR/epiCRealism",
  motion_adapter=adapter,
  torch_dtype=torch.float16
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
pipeline.enable_vae_slicing()
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

## Model variants

There are two variants of IP-Adapter, Plus and FaceID. The Plus variant uses patch embeddings and the ViT-H image encoder. FaceID variant uses face embeddings generated from InsightFace.

<hfoptions id="ipadapter-variants">
<hfoption id="IP-Adapter Plus">

```py
import torch
from transformers import CLIPVisionModelWithProjection, AutoPipelineForText2Image

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
)
```

</hfoption>
<hfoption id="IP-Adapter FaceID">

```py
import torch
from transformers import AutoPipelineForText2Image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter(
  "h94/IP-Adapter-FaceID",
  subfolder=None,
  weight_name="ip-adapter-faceid_sdxl.bin",
  image_encoder_folder=None
)
```

To use a IP-Adapter FaceID Plus model, load the CLIP image encoder as well as [`~transformers.CLIPVisionModelWithProjection`].

```py
from transformers import AutoPipelineForText2Image, CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    image_encoder=image_encoder,
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter(
  "h94/IP-Adapter-FaceID",
  subfolder=None,
  weight_name="ip-adapter-faceid-plus_sd15.bin"
)
```

</hfoption>
</hfoptions>

## Image embeddings

The `prepare_ip_adapter_image_embeds` generates image embeddings you can reuse if you're running the pipeline multiple times because you have more than one image. Loading and encoding multiple images each time you use the pipeline can be inefficient. Precomputing the image embeddings ahead of time, saving them to disk, and loading them when you need them is more efficient.

```py
import torch
from diffusers import AutoPipelineForText2Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")

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
    generator=generator,
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

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")

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

## Applications

The section below covers some popular applications of IP-Adapter.

### Face models

Face generation and preserving its details can be challenging. To help generate more accurate faces, there are checkpoints specifically conditioned on images of cropped faces. You can find the face models in the [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) repository or the [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) repository. The FaceID checkpoints use the FaceID embeddings from [InsightFace](https://github.com/deepinsight/insightface) instead of CLIP image embeddings.

We recommend using the [`DDIMScheduler`] or [`EulerDiscreteScheduler`] for face models.

<hfoptions id="usage">
<hfoption id="h94/IP-Adapter">

```py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipeline = StableDiffusionPipeline.from_pretrained(
  "stable-diffusion-v1-5/stable-diffusion-v1-5",
  torch_dtype=torch.float16,
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models", 
  weight_name="ip-adapter-full-face_sd15.bin"
)

pipeline.set_ip_adapter_scale(0.5)
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")

pipeline(
    prompt="A photo of Einstein as a chef, wearing an apron, cooking in a French restaurant",
    ip_adapter_image=image,
    negative_prompt="lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=100,
).images[0]
```

<div style="display: flex; gap: 10px; justify-content: space-around; align-items: flex-end;">
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png" width="400" alt="IP-Adapter image"/>
    <figcaption style="text-align: center;">IP-Adapter image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein.png" width="400" alt="generated image"/>
    <figcaption style="text-align: center;">generated image</figcaption>
  </figure>
</div>

</hfoption>
<hfoption id="h94/IP-Adapter-FaceID">

For FaceID models, extract the face embeddings and pass them as a list of tensors to `ip_adapter_image_embeds`.

```py
# pip install insightface
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image
from insightface.app import FaceAnalysis

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
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
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
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

The IP-Adapter FaceID Plus and Plus v2 models require CLIP image embeddings. Prepare the face embeddings and then extract and pass the CLIP embeddings to the hidden image projection layers.

```py
clip_embeds = pipeline.prepare_ip_adapter_image_embeds(
  [ip_adapter_images], None, torch.device("cuda"), num_images, True)[0]

pipeline.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
# set to True if using IP-Adapter FaceID Plus v2
pipeline.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False
```

</hfoption>
</hfoptions>

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
    torch_dtype=torch.float16,
)
```

Load a base model, scheduler and the following IP-Adapters.

- [ip-adapter-plus_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) uses patch embeddings and a ViT-H image encoder
- [ip-adapter-plus-face_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) uses patch embeddings and a ViT-H image encoder but it is conditioned on images of cropped faces

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
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

### Instant generation

[Latent Consistency Models (LCM)](../api/pipelines/latent_consistency_models) can generate images 4 steps or less, unlike other diffusion models which require a lot more steps, making it feel "instantaneous". IP-Adapters are compatible with LCM models to instantly generate images.

Load the IP-Adapter weights and load the LoRA weights with [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`].

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
  "sd-dreambooth-library/herge-style",
  torch_dtype=torch.float16
)

pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models",
  weight_name="ip-adapter_sd15.bin"
)
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
# enable_model_cpu_offload to reduce memory usage
pipeline.enable_model_cpu_offload()
```

Try using a lower IP-Adapter scale to condition generation more on the style you want to apply and remember to use the special token in your prompt to trigger its generation.

```py
pipeline.set_ip_adapter_scale(0.4)

prompt = "herge_style woman in armor, best quality, high quality"

ip_adapter_image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
pipeline(
    prompt=prompt,
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

### Structural control

For structural control, combine IP-Adapter with [ControlNet](../api/pipelines/controlnet) conditioned on depth maps, edge maps, pose estimations, and more.

The example below loads a [`ControlNetModel`] checkpoint conditioned on depth maps and combines it with a IP-Adapter.

```py
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
  "lllyasviel/control_v11f1p_sd15_depth",
  torch_dtype=torch.float16
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="models",
  weight_name="ip-adapter_sd15.bin"
)
```

Pass the depth map and IP-Adapter image to the pipeline.

```py
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

For style and layout control, combine IP-Adapter with [InstantStyle](https://huggingface.co/papers/2404.02733). InstantStyle separates *style* (color, texture, overall feel) and *content* from each other. It only applies the style in style-specific blocks of the model to prevent it from distorting other areas of an image. This generates images with stronger and more consistent styles and better control over the layout.

The IP-Adapter is only activated for specific parts of the model. Use the [`~loaders.IPAdapterMixin.set_ip_adapter_scale`] method to scale the influence of the IP-Adapter in different layers. The example below activates the IP-Adapter in the second layer of the models down `block_2` and up `block_0`. Down `block_2` is where the IP-Adapter injects layout information and up `block_0` is where style is injected.

```py
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.float16
).to("cuda")
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

You can also insert the IP-Adapter in all the model layers. This tends to generate images that focus more on the image prompt and may reduce the diversity of generated images. Only activate the IP-Adapter in up `block_0` or the style layer.

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
    <figcaption style="text-align: center;">style-layer generated image</figcaption>
  </figure>
  <figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_ip_adapter.png" width="400" alt="Generated image (IP-Adapter only)"/>
    <figcaption style="text-align: center;">all layers generated image</figcaption>
  </figure>
</div>