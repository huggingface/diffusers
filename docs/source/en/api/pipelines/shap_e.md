<!--Copyright 2025 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Shap-E

The Shap-E model was proposed in [Shap-E: Generating Conditional 3D Implicit Functions](https://huggingface.co/papers/2305.02463) by Alex Nichol and Heewoo Jun from [OpenAI](https://github.com/openai).

The abstract from the paper is:

*We present Shap-E, a conditional generative model for 3D assets. Unlike recent work on 3D generative models which produce a single output representation, Shap-E directly generates the parameters of implicit functions that can be rendered as both textured meshes and neural radiance fields. We train Shap-E in two stages: first, we train an encoder that deterministically maps 3D assets into the parameters of an implicit function; second, we train a conditional diffusion model on outputs of the encoder. When trained on a large dataset of paired 3D and text data, our resulting models are capable of generating complex and diverse 3D assets in a matter of seconds. When compared to Point-E, an explicit generative model over point clouds, Shap-E converges faster and reaches comparable or better sample quality despite modeling a higher-dimensional, multi-representation output space.*

The original codebase can be found at [openai/shap-e](https://github.com/openai/shap-e).

<Tip>

See the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## Text-to-3D

To generate a gif of a 3D object, pass a text prompt to the [`ShapEPipeline`]. The pipeline generates a list of image frames which are used to create the 3D object.

```py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = ["A firecracker", "A birthday cupcake"]

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images
```

이제 [`~utils.export_to_gif`] 함수를 사용해 이미지 프레임 리스트를 3D 오브젝트의 gif로 변환합니다.

```py
from diffusers.utils import export_to_gif

export_to_gif(images[0], "firecracker_3d.gif")
export_to_gif(images[1], "cake_3d.gif")
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/firecracker_out.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">prompt = "A firecracker"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/cake_out.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">prompt = "A birthday cupcake"</figcaption>
  </div>
</div>

## Image-to-3D

To generate a 3D object from another image, use the [`ShapEImg2ImgPipeline`]. You can use an existing image or generate an entirely new one. Let's use the [Kandinsky 2.1](../api/pipelines/kandinsky) model to generate a new image.

```py
from diffusers import DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

prompt = "A cheeseburger, white background"

image_embeds, negative_image_embeds = prior_pipeline(prompt, guidance_scale=1.0).to_tuple()
image = pipeline(
    prompt,
    image_embeds=image_embeds,
    negative_image_embeds=negative_image_embeds,
).images[0]

image.save("burger.png")
```

Pass the cheeseburger to the [`ShapEImg2ImgPipeline`] to generate a 3D representation of it.

```py
from PIL import Image
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif

pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, variant="fp16").to("cuda")

guidance_scale = 3.0
image = Image.open("burger.png").resize((256, 256))

images = pipe(
    image,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    frame_size=256,
).images

gif_path = export_to_gif(images[0], "burger_3d.gif")
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/burger_in.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">cheeseburger</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/burger_out.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">3D cheeseburger</figcaption>
  </div>
</div>

## Generate mesh

Shap-E is a flexible model that can also generate textured mesh outputs to be rendered for downstream applications. In this example, you'll convert the output into a `glb` file because the 🤗 Datasets library supports mesh visualization of `glb` files which can be rendered by the [Dataset viewer](https://huggingface.co/docs/hub/datasets-viewer#dataset-preview).

You can generate mesh outputs for both the [`ShapEPipeline`] and [`ShapEImg2ImgPipeline`] by specifying the `output_type` parameter as `"mesh"`:

```py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = "A birthday cupcake"

images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=64, frame_size=256, output_type="mesh").images
```

Use the [`~utils.export_to_ply`] function to save the mesh output as a `ply` file:

<Tip>

You can optionally save the mesh output as an `obj` file with the [`~utils.export_to_obj`] function. The ability to save the mesh output in a variety of formats makes it more flexible for downstream usage!

</Tip>

```py
from diffusers.utils import export_to_ply

ply_path = export_to_ply(images[0], "3d_cake.ply")
print(f"Saved to folder: {ply_path}")
```

Then you can convert the `ply` file to a `glb` file with the trimesh library:

```py
import trimesh

mesh = trimesh.load("3d_cake.ply")
mesh_export = mesh.export("3d_cake.glb", file_type="glb")
```

By default, the mesh output is focused from the bottom viewpoint but you can change the default viewpoint by applying a rotation transform:

```py
import trimesh
import numpy as np

mesh = trimesh.load("3d_cake.ply")
rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
mesh = mesh.apply_transform(rot)
mesh_export = mesh.export("3d_cake.glb", file_type="glb")
```

Upload the mesh file to your dataset repository to visualize it with the Dataset viewer!

<div class="flex justify-center">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/3D-cake.gif"/>
</div>


## ShapEPipeline
[[autodoc]] ShapEPipeline
	- all
	- __call__

## ShapEImg2ImgPipeline
[[autodoc]] ShapEImg2ImgPipeline
	- all
	- __call__

## ShapEPipelineOutput
[[autodoc]] pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput
