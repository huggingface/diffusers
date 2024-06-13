<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable diffusion XL

Stable Diffusion XL은 Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach에 의해 [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)에서 제안되었습니다.

논문 초록은 다음을 따릅니다:

*text-to-image의 latent diffusion 모델인 SDXL을 소개합니다. 이전 버전의 Stable Diffusion과 비교하면, SDXL은 세 배 더큰 규모의 UNet 백본을 포함합니다: 모델 파라미터의 증가는 많은 attention 블럭을 사용하고 더 큰 cross-attention context를 SDXL의 두 번째 텍스트 인코더에 사용하기 때문입니다. 다중 종횡비에 다수의 새로운 conditioning 방법을 구성했습니다. 또한 후에 수정하는 image-to-image 기술을 사용함으로써 SDXL에 의해 생성된 시각적 품질을 향상하기 위해 정제된 모델을 소개합니다. SDXL은 이전 버전의 Stable Diffusion보다 성능이 향상되었고, 이러한 black-box 최신 이미지 생성자와 경쟁력있는 결과를 달성했습니다.*

## 팁

- Stable Diffusion XL은 특히 786과 1024사이의 이미지에 잘 작동합니다.
- Stable Diffusion XL은 아래와 같이 학습된 각 텍스트 인코더에 대해 서로 다른 프롬프트를 전달할 수 있습니다. 동일한 프롬프트의 다른 부분을 텍스트 인코더에 전달할 수도 있습니다.
- Stable Diffusion XL 결과 이미지는 아래에 보여지듯이 정제기(refiner)를 사용함으로써 향상될 수 있습니다.

### 이용가능한 체크포인트:

- *Text-to-Image (1024x1024 해상도)*: [`StableDiffusionXLPipeline`]을 사용한 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- *Image-to-Image / 정제기(refiner) (1024x1024 해상도)*: [`StableDiffusionXLImg2ImgPipeline`]를 사용한 [stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

## 사용 예시

SDXL을 사용하기 전에 `transformers`, `accelerate`, `safetensors` 와 `invisible_watermark`를 설치하세요.
다음과 같이 라이브러리를 설치할 수 있습니다:

```sh
pip install transformers
pip install accelerate
pip install safetensors
pip install invisible-watermark>=0.2.0
```

### 워터마커

Stable Diffusion XL로 이미지를 생성할 때 워터마크가 보이지 않도록 추가하는 것을 권장하는데, 이는 다운스트림(downstream) 어플리케이션에서 기계에 합성되었는지를 식별하는데 도움을 줄 수 있습니다. 그렇게 하려면 [invisible_watermark 라이브러리](https://pypi.org/project/invisible-watermark/)를 통해 설치해주세요:


```sh
pip install invisible-watermark>=0.2.0
```

`invisible-watermark` 라이브러리가 설치되면 워터마커가 **기본적으로** 사용될 것입니다.

생성 또는 안전하게 이미지를 배포하기 위해 다른 규정이 있다면, 다음과 같이 워터마커를 비활성화할 수 있습니다:

```py
pipe = StableDiffusionXLPipeline.from_pretrained(..., add_watermarker=False)
```

### Text-to-Image

*text-to-image*를 위해 다음과 같이 SDXL을 사용할 수 있습니다:

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt=prompt).images[0]
```

### Image-to-image

*image-to-image*를 위해 다음과 같이 SDXL을 사용할 수 있습니다:

```py
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

init_image = load_image(url).convert("RGB")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images[0]
```

### 인페인팅

*inpainting*를 위해 다음과 같이 SDXL을 사용할 수 있습니다:

```py
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80).images[0]
```

### 이미지 결과물을 정제하기

[base 모델 체크포인트](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)에서, StableDiffusion-XL 또한 고주파 품질을 향상시키는 이미지를 생성하기 위해 낮은 노이즈 단계 이미지를 제거하는데 특화된 [refiner 체크포인트](huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)를 포함하고 있습니다. 이 refiner 체크포인트는 이미지 품질을 향상시키기 위해 base 체크포인트를 실행한 후 "두 번째 단계" 파이프라인에 사용될 수 있습니다.

refiner를 사용할 때, 쉽게 사용할 수 있습니다
- 1.) base 모델과 refiner을 사용하는데, 이는 *Denoisers의 앙상블*을 위한 첫 번째 제안된 [eDiff-I](https://research.nvidia.com/labs/dir/eDiff-I/)를 사용하거나
- 2.) base 모델을 거친 후 [SDEdit](https://arxiv.org/abs/2108.01073) 방법으로 단순하게 refiner를 실행시킬 수 있습니다.

**참고**: SD-XL base와 refiner를 앙상블로 사용하는 아이디어는 커뮤니티 기여자들이 처음으로 제안했으며, 이는 다음과 같은 `diffusers`를 구현하는 데도 도움을 주셨습니다.
- [SytanSD](https://github.com/SytanSD)
- [bghira](https://github.com/bghira)
- [Birch-san](https://github.com/Birch-san)
- [AmericanPresidentJimmyCarter](https://github.com/AmericanPresidentJimmyCarter)

#### 1.) Denoisers의 앙상블

base와 refiner 모델을 denoiser의 앙상블로 사용할 때, base 모델은 고주파 diffusion 단계를 위한 전문가의 역할을 해야하고, refiner는 낮은 노이즈 diffusion 단계를 위한 전문가의 역할을 해야 합니다.

2.)에 비해 1.)의 장점은 전체적으로 denoising 단계가 덜 필요하므로 속도가 훨씬 더 빨라집니다. 단점은 base 모델의 결과를 검사할 수 없다는 것입니다. 즉, 여전히 노이즈가 심하게 제거됩니다.

base 모델과 refiner를 denoiser의 앙상블로 사용하기 위해 각각 고노이즈(high-nosise) (*즉* base 모델)와 저노이즈 (*즉* refiner 모델)의 노이즈를 제거하는 단계를 거쳐야하는 타임스텝의 기간을 정의해야 합니다.
base 모델의 [`denoising_end`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.__call__.denoising_end)와 refiner 모델의 [`denoising_start`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline.__call__.denoising_start)를 사용해 간격을 정합니다.

`denoising_end`와 `denoising_start` 모두 0과 1사이의 실수 값으로 전달되어야 합니다.
전달되면 노이즈 제거의 끝과 시작은 모델 스케줄에 의해 정의된 이산적(discrete) 시간 간격의 비율로 정의됩니다.
노이즈 제거 단계의 수는 모델이 학습된 불연속적인 시간 간격과 선언된 fractional cutoff에 의해 결정되므로 '강도' 또한 선언된 경우 이 값이 '강도'를 재정의합니다.

예시를 들어보겠습니다.
우선, 두 개의 파이프라인을 가져옵니다. 텍스트 인코더와 variational autoencoder는 동일하므로 refiner를 위해 다시 불러오지 않아도 됩니다.

```py
from diffusers import DiffusionPipeline
import torch

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
```

이제 추론 단계의 수와 고노이즈에서 노이즈를 제거하는 단계(*즉* base 모델)를 거쳐 실행되는 지점을 정의합니다.

```py
n_steps = 40
high_noise_frac = 0.8
```

Stable Diffusion XL base 모델은 타임스텝 0-999에 학습되며 Stable Diffusion XL refiner는 포괄적인 낮은 노이즈 타임스텝인 0-199에 base 모델로 부터 파인튜닝되어, 첫 800 타임스텝 (높은 노이즈)에 base 모델을 사용하고 마지막 200 타입스텝 (낮은 노이즈)에서 refiner가 사용됩니다. 따라서, `high_noise_frac`는 0.8로 설정하고, 모든 200-999 스텝(노이즈 제거 타임스텝의 첫 80%)은 base 모델에 의해 수행되며 0-199 스텝(노이즈 제거 타임스텝의 마지막 20%)은 refiner 모델에 의해 수행됩니다.

기억하세요, 노이즈 제거 절차는 **높은 값**(높은 노이즈) 타임스텝에서 시작되고, **낮은 값** (낮은 노이즈) 타임스텝에서 끝납니다.

이제 두 파이프라인을 실행해봅시다. `denoising_end`과 `denoising_start`를 같은 값으로 설정하고 `num_inference_steps`는 상수로 유지합니다. 또한 base 모델의 출력은 잠재 공간에 있어야 한다는 점을 기억하세요:

```py
prompt = "A majestic lion jumping from a big stone at night"

image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
```

이미지를 살펴보겠습니다.

| 원래의 이미지 | Denoiser들의 앙상블 |
|---|---|
| ![lion_base](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_base.png) | ![lion_ref](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_refined.png)

동일한 40 단계에서 base 모델을 실행한다면, 이미지의 디테일(예: 사자의 눈과 코)이 떨어졌을 것입니다:

<Tip>

앙상블 방식은 사용 가능한 모든 스케줄러에서 잘 작동합니다!

</Tip>

#### 2.) 노이즈가 완전히 제거된 기본 이미지에서 이미지 출력을 정제하기

일반적인 [`StableDiffusionImg2ImgPipeline`] 방식에서, 기본 모델에서 생성된 완전히 노이즈가 제거된 이미지는 [refiner checkpoint](huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)를 사용해 더 향상시킬 수 있습니다.

이를 위해, 보통의 "base" text-to-image 파이프라인을 수행 후에 image-to-image 파이프라인으로써 refiner를 실행시킬 수 있습니다. base 모델의 출력을 잠재 공간에 남겨둘 수 있습니다.

```py
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil").images[0]
image = refiner(prompt=prompt, image=image[None, :]).images[0]
```

| 원래의 이미지 | 정제된 이미지 |
|---|---|
| ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/init_image.png) | ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_image.png) |

<Tip>

refiner는 또한 인페인팅 설정에 잘 사용될 수 있습니다. 아래에 보여지듯이 [`StableDiffusionXLInpaintPipeline`] 클래스를 사용해서 만들어보세요.

</Tip>

Denoiser 앙상블 설정에서 인페인팅에 refiner를 사용하려면 다음을 수행하면 됩니다:

```py
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
num_inference_steps = 75
high_noise_frac = 0.7

image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
).images[0]
```

일반적인 SDE 설정에서 인페인팅에 refiner를 사용하기 위해, `denoising_end`와 `denoising_start`를 제거하고 refiner의 추론 단계의 수를 적게 선택하세요.

### 단독 체크포인트 파일 / 원래의 파일 형식으로 불러오기

[`~diffusers.loaders.FromSingleFileMixin.from_single_file`]를 사용함으로써 원래의 파일 형식을 `diffusers` 형식으로 불러올 수 있습니다:

```py
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    "./sd_xl_base_1.0.safetensors", torch_dtype=torch.float16
)
pipe.to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    "./sd_xl_refiner_1.0.safetensors", torch_dtype=torch.float16
)
refiner.to("cuda")
```

### 모델 offloading을 통해 메모리 최적화하기

out-of-memory 에러가 난다면, [`StableDiffusionXLPipeline.enable_model_cpu_offload`]을 사용하는 것을 권장합니다.

```diff
- pipe.to("cuda")
+ pipe.enable_model_cpu_offload()
```

그리고

```diff
- refiner.to("cuda")
+ refiner.enable_model_cpu_offload()
```

### `torch.compile`로 추론 속도를 올리기

`torch.compile`를 사용함으로써 추론 속도를 올릴 수 있습니다. 이는 **ca.** 20% 속도 향상이 됩니다.

```diff
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
+ refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
```

### `torch < 2.0`일 때 실행하기

**참고** Stable Diffusion XL을 `torch`가 2.0 버전 미만에서 실행시키고 싶을 때, xformers 어텐션을 사용해주세요:

```sh
pip install xformers
```

```diff
+pipe.enable_xformers_memory_efficient_attention()
+refiner.enable_xformers_memory_efficient_attention()
```

## StableDiffusionXLPipeline

[[autodoc]] StableDiffusionXLPipeline
	- all
	- __call__

## StableDiffusionXLImg2ImgPipeline

[[autodoc]] StableDiffusionXLImg2ImgPipeline
	- all
	- __call__

## StableDiffusionXLInpaintPipeline

[[autodoc]] StableDiffusionXLInpaintPipeline
	- all
	- __call__

### 각 텍스트 인코더에 다른 프롬프트를 전달하기

Stable Diffusion XL는 두 개의 텍스트 인코더에 학습되었습니다. 기본 동작은 각 프롬프트에 동일한 프롬프트를 전달하는 것입니다. 그러나 [일부 사용자](https://github.com/huggingface/diffusers/issues/4004#issuecomment-1627764201)가 품질을 향상시킬 수 있다고 지적한 것처럼 텍스트 인코더마다 다른 프롬프트를 전달할 수 있습니다. 그렇게 하려면, `prompt_2`와 `negative_prompt_2`를 `prompt`와 `negative_prompt`에 전달해야 합니다. 그렇게 함으로써, 원래의 프롬프트들(`prompt`)과 부정 프롬프트들(`negative_prompt`)를 `텍스트 인코더`에 전달할 것입니다.(공식 SDXL 0.9/1.0의 [OpenAI CLIP-ViT/L-14](https://huggingface.co/openai/clip-vit-large-patch14)에서 볼 수 있습니다.) 그리고 `prompt_2`와 `negative_prompt_2`는 `text_encoder_2`에 전달됩니다.(공식 SDXL 0.9/1.0의 [OpenCLIP-ViT/bigG-14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)에서 볼 수 있습니다.)

```py
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

# OAI CLIP-ViT/L-14에 prompt가 전달됩니다
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# OpenCLIP-ViT/bigG-14에 prompt_2가 전달됩니다
prompt_2 = "monet painting"
image = pipe(prompt=prompt, prompt_2=prompt_2).images[0]
```