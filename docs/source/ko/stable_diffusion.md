<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 효과적이고 효율적인 Diffusion

[[open-in-colab]]

특정 스타일로 이미지를 생성하거나 원하는 내용을 포함하도록[`DiffusionPipeline`]을 설정하는 것은 까다로울 수 있습니다. 종종 만족스러운 이미지를 얻기까지 [`DiffusionPipeline`]을 여러 번 실행해야 하는 경우가 많습니다. 그러나 무에서 유를 창조하는 것은 특히 추론을 반복해서 실행하는 경우 계산 집약적인 프로세스입니다.

그렇기 때문에 파이프라인에서 *계산*(속도) 및 *메모리*(GPU RAM) 효율성을 극대화하여 추론 주기 사이의 시간을 단축하여 더 빠르게 반복할 수 있도록 하는 것이 중요합니다.

이 튜토리얼에서는 [`DiffusionPipeline`]을 사용하여 더 빠르고 효과적으로 생성하는 방법을 안내합니다.

[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 모델을 불러와서 시작합니다:

```python
from diffusers import DiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id)
```

예제 프롬프트는 "portrait of an old warrior chief" 이지만, 자유롭게 자신만의 프롬프트를 사용해도 됩니다:

```python
prompt = "portrait photo of a old warrior chief"
```

## 속도

<Tip>

💡 GPU에 액세스할 수 없는 경우 다음과 같은 GPU 제공업체에서 무료로 사용할 수 있습니다!. [Colab](https://colab.research.google.com/)

</Tip>

추론 속도를 높이는 가장 간단한 방법 중 하나는 Pytorch 모듈을 사용할 때와 같은 방식으로 GPU에 파이프라인을 배치하는 것입니다:

```python
pipeline = pipeline.to("cuda")
```

동일한 이미지를 사용하고 개선할 수 있는지 확인하려면 [`Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html)를 사용하고 [재현성](./using-diffusers/reusing_seeds)에 대한 시드를 설정하세요:

```python
import torch

generator = torch.Generator("cuda").manual_seed(0)
```

이제 이미지를 생성할 수 있습니다:

```python
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_1.png">
</div>

이 프로세스는 T4 GPU에서 약 30초가 소요되었습니다(할당된 GPU가 T4보다 나은 경우 더 빠를 수 있음). 기본적으로 [`DiffusionPipeline`]은 50개의 추론 단계에 대해 전체 `float32` 정밀도로 추론을 실행합니다. `float16`과 같은 더 낮은 정밀도로 전환하거나 추론 단계를 더 적게 실행하여 속도를 높일 수 있습니다.

`float16`으로 모델을 로드하고 이미지를 생성해 보겠습니다:


```python
import torch

pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_2.png">
</div>

이번에는 이미지를 생성하는 데 약 11초밖에 걸리지 않아 이전보다 3배 가까이 빨라졌습니다!

<Tip>

💡 파이프라인은 항상 `float16`에서 실행할 것을 강력히 권장하며, 지금까지 출력 품질이 저하되는 경우는 거의 없었습니다.

</Tip>

또 다른 옵션은 추론 단계의 수를 줄이는 것입니다. 보다 효율적인 스케줄러를 선택하면 출력 품질 저하 없이 단계 수를 줄이는 데 도움이 될 수 있습니다. 현재 모델과 호환되는 스케줄러는 `compatibles` 메서드를 호출하여 [`DiffusionPipeline`]에서 찾을 수 있습니다:

```python
pipeline.scheduler.compatibles
[
    diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
    diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
    diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
    diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
    diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
    diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
    diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
    diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
    diffusers.schedulers.scheduling_pndm.PNDMScheduler,
    diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
    diffusers.schedulers.scheduling_ddim.DDIMScheduler,
]
```

Stable Diffusion 모델은 일반적으로 약 50개의 추론 단계가 필요한 [`PNDMScheduler`]를 기본으로 사용하지만, [`DPMSolverMultistepScheduler`]와 같이 성능이 더 뛰어난 스케줄러는 약 20개 또는 25개의 추론 단계만 필요로 합니다. 새 스케줄러를 로드하려면 [`ConfigMixin.from_config`] 메서드를 사용합니다:

```python
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
```

`num_inference_steps`를 20으로 설정합니다:

```python
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_3.png">
</div>

추론시간을 4초로 단축할 수 있었습니다! ⚡️

## 메모리

파이프라인 성능 향상의 또 다른 핵심은 메모리 사용량을 줄이는 것인데, 초당 생성되는 이미지 수를 최대화하려고 하는 경우가 많기 때문에 간접적으로 더 빠른 속도를 의미합니다. 한 번에 생성할 수 있는 이미지 수를 확인하는 가장 쉬운 방법은 `OutOfMemoryError`(OOM)이 발생할 때까지 다양한 배치 크기를 시도해 보는 것입니다.

프롬프트 목록과 `Generators`에서 이미지 배치를 생성하는 함수를 만듭니다. 좋은 결과를 생성하는 경우 재사용할 수 있도록 각 `Generator`에 시드를 할당해야 합니다.

```python
def get_inputs(batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}
```

또한 각 이미지 배치를 보여주는 기능이 필요합니다:

```python
from PIL import Image


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
```

`batch_size=4`부터 시작해 얼마나 많은 메모리를 소비했는지 확인합니다:

```python
images = pipeline(**get_inputs(batch_size=4)).images
image_grid(images)
```

RAM이 더 많은 GPU가 아니라면 위의 코드에서 `OOM` 오류가 반환되었을 것입니다! 대부분의 메모리는 cross-attention 레이어가 차지합니다. 이 작업을 배치로 실행하는 대신 순차적으로 실행하면 상당한 양의 메모리를 절약할 수 있습니다. 파이프라인을 구성하여 [`~DiffusionPipeline.enable_attention_slicing`] 함수를 사용하기만 하면 됩니다:


```python
pipeline.enable_attention_slicing()
```

이제 `batch_size`를 8로 늘려보세요!

```python
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_5.png">
</div>

이전에는 4개의 이미지를 배치로 생성할 수도 없었지만, 이제는 이미지당 약 3.5초 만에 8개의 이미지를 배치로 생성할 수 있습니다! 이는 아마도 품질 저하 없이 T4 GPU에서 가장 빠른 속도일 것입니다.

## 품질

지난 두 섹션에서는 `fp16`을 사용하여 파이프라인의 속도를 최적화하고, 더 성능이 좋은 스케줄러를 사용하여 추론 단계의 수를 줄이고, attention slicing을 활성화하여 메모리 소비를 줄이는 방법을 배웠습니다. 이제 생성된 이미지의 품질을 개선하는 방법에 대해 집중적으로 알아보겠습니다.


### 더 나은 체크포인트

가장 확실한 단계는 더 나은 체크포인트를 사용하는 것입니다. Stable Diffusion 모델은 좋은 출발점이며, 공식 출시 이후 몇 가지 개선된 버전도 출시되었습니다. 하지만 최신 버전을 사용한다고 해서 자동으로 더 나은 결과를 얻을 수 있는 것은 아닙니다. 여전히 다양한 체크포인트를 직접 실험해보고, [negative prompts](https://minimaxir.com/2022/11/stable-diffusion-negative-prompt/) 사용 등 약간의 조사를 통해 최상의 결과를 얻어야 합니다.

이 분야가 성장함에 따라 특정 스타일을 연출할 수 있도록 세밀하게 조정된 고품질 체크포인트가 점점 더 많아지고 있습니다. [Hub](https://huggingface.co/models?library=diffusers&sort=downloads)와 [Diffusers Gallery](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery)를 둘러보고 관심 있는 것을 찾아보세요!


### 더 나은 파이프라인 구성 요소

현재 파이프라인 구성 요소를 최신 버전으로 교체해 볼 수도 있습니다. Stability AI의 최신 [autodecoder](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/vae)를 파이프라인에 로드하고 몇 가지 이미지를 생성해 보겠습니다:


```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipeline.vae = vae
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_6.png">
</div>

### 더 나은 프롬프트 엔지니어링

이미지를 생성하는 데 사용하는 텍스트 프롬프트는 *prompt engineering*이라고 할 정도로 매우 중요합니다. 프롬프트 엔지니어링 시 고려해야 할 몇 가지 사항은 다음과 같습니다:

- 생성하려는 이미지 또는 유사한 이미지가 인터넷에 어떻게 저장되어 있는가?
- 내가 원하는 스타일로 모델을 유도하기 위해 어떤 추가 세부 정보를 제공할 수 있는가?

이를 염두에 두고 색상과 더 높은 품질의 디테일을 포함하도록 프롬프트를 개선해 봅시다:


```python
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"
```

새로운 프롬프트로 이미지 배치를 생성합니다:

```python
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_7.png">
</div>

꽤 인상적입니다! `1`의 시드를 가진 `Generator`에 해당하는 두 번째 이미지에 피사체의 나이에 대한 텍스트를 추가하여 조금 더 조정해 보겠습니다:

```python
prompts = [
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
]

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25).images
image_grid(images)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/stable_diffusion_101/sd_101_8.png">
</div>

## 다음 단계

이 튜토리얼에서는 계산 및 메모리 효율을 높이고 생성된 출력의 품질을 개선하기 위해 [`DiffusionPipeline`]을 최적화하는 방법을 배웠습니다. 파이프라인을 더 빠르게 만드는 데 관심이 있다면 다음 리소스를 살펴보세요:

- [PyTorch 2.0](./optimization/torch2.0) 및 [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html)이 어떻게 추론 속도를 5~300% 향상시킬 수 있는지 알아보세요. A100 GPU에서는 추론 속도가 최대 50%까지 빨라질 수 있습니다!
- PyTorch 2를 사용할 수 없는 경우, [xFormers](./optimization/xformers)를 설치하는 것이 좋습니다. 메모리 효율적인 어텐션 메커니즘은 PyTorch 1.13.1과 함께 사용하면 속도가 빨라지고 메모리 소비가 줄어듭니다.
- 모델 오프로딩과 같은 다른 최적화 기법은 [이 가이드](./optimization/fp16)에서 다루고 있습니다.