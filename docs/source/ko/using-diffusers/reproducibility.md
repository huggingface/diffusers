<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 재현 가능한 파이프라인 생성하기

[[open-in-colab]]

재현성은 테스트, 결과 재현, 그리고 [이미지 퀄리티 높이기](resuing_seeds)에서 중요합니다.
그러나 diffusion 모델의 무작위성은 매번 모델이 돌아갈 때마다 파이프라인이 다른 이미지를 생성할 수 있도록 하는 이유로 필요합니다.
플랫폼 간에 정확하게 동일한 결과를 얻을 수는 없지만, 특정 허용 범위 내에서 릴리스 및 플랫폼 간에 결과를 재현할 수는 있습니다.
그럼에도 diffusion 파이프라인과 체크포인트에 따라 허용 오차가 달라집니다.

diffusion 모델에서 무작위성의 원천을 제어하거나 결정론적 알고리즘을 사용하는 방법을 이해하는 것이 중요한 이유입니다.

<Tip>

💡 Pytorch의 [재현성에 대한 선언](https://pytorch.org/docs/stable/notes/randomness.html)를 꼭 읽어보길 추천합니다:

> 완전하게 재현가능한 결과는 Pytorch 배포, 개별적인 커밋, 혹은 다른 플랫폼들에서 보장되지 않습니다.
> 또한, 결과는 CPU와 GPU 실행간에 심지어 같은 seed를 사용할 때도 재현 가능하지 않을 수 있습니다.

</Tip>

## 무작위성 제어하기

추론에서, 파이프라인은 노이즈를 줄이기 위해 가우시안 노이즈를 생성하거나 스케줄링 단계에 노이즈를 더하는 등의 랜덤 샘플링 실행에 크게 의존합니다,

[DDIMPipeline](https://huggingface.co/docs/diffusers/v0.18.0/en/api/pipelines/ddim#diffusers.DDIMPipeline)에서 두 추론 단계 이후의 텐서 값을 살펴보세요:

```python
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# 모델과 스케줄러를 불러오기
ddim = DDIMPipeline.from_pretrained(model_id)

# 두 개의 단계에 대해서 파이프라인을 실행하고 numpy tensor로 값을 반환하기
image = ddim(num_inference_steps=2, output_type="np").images
print(np.abs(image).sum())
```

위의 코드를 실행하면 하나의 값이 나오지만, 다시 실행하면 다른 값이 나옵니다. 무슨 일이 일어나고 있는 걸까요?

파이프라인이 실행될 때마다, [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html)은
단계적으로 노이즈 제거되는 가우시안 노이즈가 생성하기 위한 다른 랜덤 seed를 사용합니다.

그러나 동일한 이미지를 안정적으로 생성해야 하는 경우에는 CPU에서 파이프라인을 실행하는지 GPU에서 실행하는지에 따라 달라집니다.

### CPU

CPU에서 재현 가능한 결과를 생성하려면, PyTorch [Generator](https://pytorch.org/docs/stable/generated/torch.randn.html)로 seed를 고정합니다:

```python
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# 모델과 스케줄러 불러오기
ddim = DDIMPipeline.from_pretrained(model_id)

# 재현성을 위해 generator 만들기
generator = torch.Generator(device="cpu").manual_seed(0)

# 두 개의 단계에 대해서 파이프라인을 실행하고 numpy tensor로 값을 반환하기
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

이제 위의 코드를 실행하면 seed를 가진 `Generator` 객체가 파이프라인의 모든 랜덤 함수에 전달되므로 항상 `1491.1711` 값이 출력됩니다.

특정 하드웨어 및 PyTorch 버전에서 이 코드 예제를 실행하면 동일하지는 않더라도 유사한 결과를 얻을 수 있습니다.

<Tip>

💡 처음에는 시드를 나타내는 정수값 대신에 `Generator` 개체를 파이프라인에 전달하는 것이 약간 비직관적일 수 있지만,
`Generator`는 순차적으로 여러 파이프라인에 전달될 수 있는 \랜덤상태\이기 때문에 PyTorch에서 확률론적 모델을 다룰 때 권장되는 설계입니다.

</Tip>

### GPU

예를 들면, GPU 상에서 같은 코드 예시를 실행하면:

```python
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# 모델과 스케줄러 불러오기
ddim = DDIMPipeline.from_pretrained(model_id)
ddim.to("cuda")

# 재현성을 위한 generator 만들기
generator = torch.Generator(device="cuda").manual_seed(0)

# 두 개의 단계에 대해서 파이프라인을 실행하고 numpy tensor로 값을 반환하기
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

GPU가 CPU와 다른 난수 생성기를 사용하기 때문에 동일한 시드를 사용하더라도 결과가 같지 않습니다.

이 문제를 피하기 위해 🧨 Diffusers는 CPU에 임의의 노이즈를 생성한 다음 필요에 따라 텐서를 GPU로 이동시키는
[randn_tensor()](https://huggingface.co/docs/diffusers/v0.18.0/en/api/utilities#diffusers.utils.randn_tensor)기능을 가지고 있습니다.
`randn_tensor` 기능은 파이프라인 내부 어디에서나 사용되므로 파이프라인이 GPU에서 실행되더라도 **항상** CPU `Generator`를 통과할 수 있습니다.

이제 결과에 훨씬 더 다가왔습니다!

```python
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# 모델과 스케줄러 불러오기
ddim = DDIMPipeline.from_pretrained(model_id)
ddim.to("cuda")

#재현성을 위한 generator 만들기 (GPU에 올리지 않도록 조심한다!)
generator = torch.manual_seed(0)

# 두 개의 단계에 대해서 파이프라인을 실행하고 numpy tensor로 값을 반환하기
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

<Tip>

💡 재현성이 중요한 경우에는 항상 CPU generator를 전달하는 것이 좋습니다.
성능 손실은 무시할 수 없는 경우가 많으며 파이프라인이 GPU에서 실행되었을 때보다 훨씬 더 비슷한 값을 생성할 수 있습니다.

</Tip>

마지막으로 [UnCLIPPipeline](https://huggingface.co/docs/diffusers/v0.18.0/en/api/pipelines/unclip#diffusers.UnCLIPPipeline)과 같은
더 복잡한 파이프라인의 경우, 이들은 종종 정밀 오차 전파에 극도로 취약합니다.
다른 GPU 하드웨어 또는 PyTorch 버전에서 유사한 결과를 기대하지 마세요.
이 경우 완전한 재현성을 위해 완전히 동일한 하드웨어 및 PyTorch 버전을 실행해야 합니다.

## 결정론적 알고리즘

결정론적 알고리즘을 사용하여 재현 가능한 파이프라인을 생성하도록 PyTorch를 구성할 수도 있습니다.
그러나 결정론적 알고리즘은 비결정론적 알고리즘보다 느리고 성능이 저하될 수 있습니다.
하지만 재현성이 중요하다면, 이것이 최선의 방법입니다!

둘 이상의 CUDA 스트림에서 작업이 시작될 때 비결정론적 동작이 발생합니다.
이 문제를 방지하려면 환경 변수 [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)를 `:16:8`로 설정해서
런타임 중에 오직 하나의 버퍼 크리만 사용하도록 설정합니다.

PyTorch는 일반적으로 가장 빠른 알고리즘을 선택하기 위해 여러 알고리즘을 벤치마킹합니다.
하지만 재현성을 원하는 경우, 벤치마크가 매 순간 다른 알고리즘을 선택할 수 있기 때문에 이 기능을 사용하지 않도록 설정해야 합니다.
마지막으로, [torch.use_deterministic_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)에
`True`를 통과시켜 결정론적 알고리즘이 활성화 되도록 합니다.

```py
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

이제 동일한 파이프라인을 두번 실행하면 동일한 결과를 얻을 수 있습니다.

```py
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
import numpy as np

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
g = torch.Generator(device="cuda")

prompt = "A bear is playing a guitar on Times Square"

g.manual_seed(0)
result1 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

g.manual_seed(0)
result2 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

print("L_inf dist = ", abs(result1 - result2).max())
"L_inf dist =  tensor(0., device='cuda:0')"
```
