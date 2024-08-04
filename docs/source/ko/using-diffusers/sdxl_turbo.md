<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Diffusion XL Turbo

[[open-in-colab]]

SDXL Turbo는 adversarial time-distilled(적대적 시간 전이) [Stable Diffusion XL](https://huggingface.co/papers/2307.01952)(SDXL) 모델로, 단 한 번의 스텝만으로 추론을 실행할 수 있습니다.

이 가이드에서는 text-to-image와 image-to-image를 위한 SDXL-Turbo를 사용하는 방법을 설명합니다.

시작하기 전에 다음 라이브러리가 설치되어 있는지 확인하세요:

```py
# Colab에서 필요한 라이브러리를 설치하기 위해 주석을 제외하세요
#!pip install -q diffusers transformers accelerate
```

## 모델 체크포인트 불러오기

모델 가중치는 Hub의 별도 하위 폴더 또는 로컬에 저장할 수 있으며, 이 경우 [`~StableDiffusionXLPipeline.from_pretrained`] 메서드를 사용해야 합니다:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("cuda")
```

또한 [`~StableDiffusionXLPipeline.from_single_file`] 메서드를 사용하여 허브 또는 로컬에서 단일 파일 형식(`.ckpt` 또는 `.safetensors`)으로 저장된 모델 체크포인트를 불러올 수도 있습니다:

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
```

## Text-to-image

Text-to-image의 경우 텍스트 프롬프트를 전달합니다. 기본적으로 SDXL Turbo는 512x512 이미지를 생성하며, 이 해상도에서 최상의 결과를 제공합니다. `height` 및 `width` 매개 변수를 768x768 또는 1024x1024로 설정할 수 있지만 이 경우 품질 저하를 예상할 수 있습니다.

모델이 `guidance_scale` 없이 학습되었으므로 이를 0.0으로 설정해 비활성화해야 합니다. 단일 추론 스텝만으로도 고품질 이미지를 생성할 수 있습니다.
스텝 수를 2, 3 또는 4로 늘리면 이미지 품질이 향상됩니다.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sdxl-turbo-text2img.png" alt="generated image of a racoon in a robe"/>
</div>

## Image-to-image

Image-to-image 생성의 경우 `num_inference_steps * strength`가 1보다 크거나 같은지 확인하세요.
Image-to-image 파이프라인은 아래 예제에서 `0.5 * 2.0 = 1` 스텝과 같이 `int(num_inference_steps * strength)` 스텝으로 실행됩니다.

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# 체크포인트를 불러올 때 추가 메모리 소모를 피하려면 from_pipe를 사용하세요.
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
init_image = init_image.resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipeline(prompt, image=init_image, strength=0.5, guidance_scale=0.0, num_inference_steps=2).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sdxl-turbo-img2img.png" alt="Image-to-image generation sample using SDXL Turbo"/>
</div>

## SDXL Turbo 속도 훨씬 더 빠르게 하기

- PyTorch 버전 2 이상을 사용하는 경우 UNet을 컴파일합니다. 첫 번째 추론 실행은 매우 느리지만 이후 실행은 훨씬 빨라집니다.

```py
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

- 기본 VAE를 사용하는 경우, 각 생성 전후에 비용이 많이 드는 `dtype` 변환을 피하기 위해 `float32`로 유지하세요. 이 작업은 첫 생성 이전에 한 번만 수행하면 됩니다:

```py
pipe.upcast_vae()
```

또는, 커뮤니티 회원인 [`@madebyollin`](https://huggingface.co/madebyollin)이 만든 [16비트 VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)를 사용할 수도 있으며, 이는 `float32`로 업캐스트할 필요가 없습니다.